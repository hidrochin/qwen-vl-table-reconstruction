"""Local, in-process Qwen-VL inference for table reconstruction.

**No hosted API.** Both tracks run open weights on a GPU you control -- Lightning
(Track 1, public) or the company server (Track 2, private). This module is the
in-process path: Transformers or Unsloth for eager generation, or an in-process
vLLM engine for fast offline batch labelling. Serving a persistent vLLM endpoint
and hitting it over localhost is the separate ``vllm_client`` path.

Heavy imports are deferred into function bodies so this module imports cleanly on
a Mac with no CUDA -- data prep and evaluation stay runnable locally while only
generation needs the GPU box.

``max_pixels`` is the setting that matters most here. Qwen-VL tokenizes images at
dynamic resolution, and a full-page table can exceed 4,000 visual tokens, which
is enough to OOM a small card or make each step crawl. Capping it trades
fine-detail resolution for throughput; for table *structure* that trade is
usually favourable, but it is worth measuring rather than assuming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Model IDs live in the registry now (single source of truth across both tracks);
# re-exported here so ``from src.model.inference import MODEL_4B`` keeps working.
from src.model.registry import (  # noqa: F401
    MODEL_4B,
    MODEL_8B,
    MODEL_DOTS_OCR,
    MODEL_GLM_OCR,
    MODEL_MINERU,
    MODEL_PADDLEOCR_VL,
    MODEL_QWEN3VL_30B,
    MODEL_QWEN36_27B,
    MODEL_QWEN36_35B,
)
from src.model.prompts import INSTRUCTIONS, build_messages, clean_prediction

# 1024 visual tokens. Each token covers a 28x28 patch with 2x2 merging.
DEFAULT_MAX_PIXELS = 1024 * 28 * 28
DEFAULT_MIN_PIXELS = 256 * 28 * 28

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Prediction:
    uid: str
    raw: str
    html: str


def strip_thinking(raw: str) -> str:
    """Remove ``<think>...</think>`` spans a reasoning model emits before its answer.

    Qwen3.6 with thinking enabled prepends a reasoning trace; the table HTML is
    what follows. ``clean_prediction`` would trim to ``<table>`` anyway, but
    dropping the trace first avoids a stray ``<table`` *inside* the reasoning
    being mistaken for the answer.
    """
    return _THINK_RE.sub("", raw) if raw else raw


class TableReconstructor:
    """Wraps a Qwen-VL checkpoint for HTML table generation.

    ``backend``:
      * ``"transformers"`` -- HF eager generate (default).
      * ``"unsloth"`` -- Unsloth's patched fast path; use it wherever a run is
        compared against an Unsloth-trained adapter, so quantization is identical
        on both arms (the repo's load-parity rule).
      * ``"vllm"`` -- in-process vLLM engine for fast offline batch generation
        (e.g. teacher labelling). First run is a debugging pass, like every
        serving path here.

    Set ``load_in_4bit`` on a small card. Pass ``adapter_path`` to load a trained
    LoRA adapter on top of the base weights. ``thinking`` turns on the reasoning
    trace for models that support it (Qwen3.6) -- worth it for the schema-
    inference tables, off by default for plain structure.
    """

    def __init__(
        self,
        model_id: str = MODEL_4B,
        backend: str = "transformers",
        load_in_4bit: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        adapter_path: str | None = None,
        thinking: bool = False,
        use_unsloth: bool | None = None,
    ):
        # Back-compat: callers still pass use_unsloth=True; map it onto backend.
        if use_unsloth:
            backend = "unsloth"
        if backend not in ("transformers", "unsloth", "vllm"):
            raise ValueError(f"backend must be transformers|unsloth|vllm, got {backend!r}")
        self.model_id = model_id
        self.backend = backend
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.thinking = thinking
        self._model = None
        self._processor = None
        self._vllm = None
        if backend == "vllm":
            self._load_vllm()
        else:
            self._load_hf(load_in_4bit, adapter_path, use_unsloth=(backend == "unsloth"))

    def _load_hf(self, load_in_4bit: bool, adapter_path: str | None, use_unsloth: bool) -> None:
        import torch

        if use_unsloth:
            # Unsloth patches Transformers on import and must come first.
            from unsloth import FastVisionModel

            self._model, self._processor = FastVisionModel.from_pretrained(
                self.model_id,
                load_in_4bit=load_in_4bit,
                use_gradient_checkpointing="unsloth",
            )
            FastVisionModel.for_inference(self._model)
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            kwargs = {"dtype": "auto", "device_map": "auto"}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    ),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )

        if adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            self._model.eval()

    def _load_vllm(self) -> None:
        """Load an in-process vLLM engine + the processor (for the chat template).

        Deferred: vLLM is a GPU-box dependency and must not be needed to import
        this module. Result-object handling follows vLLM's documented offline
        multimodal API; confirm on the first real run.
        """
        from transformers import AutoProcessor
        from vllm import LLM

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self._vllm = LLM(
            model=self.model_id,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
        )

    def _apply_template(self, messages: list[dict]) -> str:
        """Render the chat template, toggling the thinking trace when asked.

        ``enable_thinking`` is a Qwen3.6-era kwarg; processors that don't accept
        it raise ``TypeError``, so fall back to the plain call.
        """
        try:
            return self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
        except TypeError:
            return self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def predict(
        self,
        image_path: str | Path,
        mode: str = "structure",
        max_new_tokens: int = 2048,
        instruction: str | None = None,
        ocr_layout: str | None = None,
    ) -> Prediction:
        """Generate HTML for one table image.

        ``mode`` is ``structure`` | ``full`` | ``schema`` (the logical-
        reconstruction prompt). ``instruction`` overrides the mode's default --
        pass a string from the notebook to iterate on prompts, because editing
        ``prompts.py`` mid-session has no effect once the module is imported.

        ``ocr_layout`` is the serialized OCR text+positions from
        ``src.ocr.layout.serialize_layout`` (stage 1). Provide it with the
        grounded or schema instruction so the model structures known text instead
        of re-reading pixels.
        """
        from PIL import Image

        if instruction is None and mode not in INSTRUCTIONS:
            raise ValueError(f"mode must be one of {sorted(INSTRUCTIONS)}")

        image = Image.open(image_path).convert("RGB")
        messages = build_messages(image, mode, instruction, ocr_layout)
        text = self._apply_template(messages)

        if self.backend == "vllm":
            raw = self._generate_vllm(text, image, max_new_tokens)
        else:
            raw = self._generate_hf(text, image, max_new_tokens)

        raw = strip_thinking(raw)
        return Prediction(uid=Path(image_path).stem, raw=raw, html=clean_prediction(raw))

    def _generate_hf(self, text: str, image, max_new_tokens: int) -> str:
        import torch

        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self._model.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: structure reconstruction has one right answer
            )
        trimmed = generated[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(trimmed, skip_special_tokens=True)

    def _generate_vllm(self, text: str, image, max_new_tokens: int) -> str:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self._vllm.generate(
            {"prompt": text, "multi_modal_data": {"image": image}},
            params,
        )
        return outputs[0].outputs[0].text

    def predict_many(
        self,
        image_paths: list[str | Path],
        mode: str = "structure",
        ocr_layouts: list[str] | None = None,
        **kwargs,
    ) -> list[Prediction]:
        """Sequential generation with progress. Batching VLMs across differing
        image sizes is fiddly and the eval set is small; clarity wins here.

        ``ocr_layouts`` (stage-1 grounding) is a per-image list aligned to
        ``image_paths`` -- each layout goes only to its own image, unlike the
        shared ``**kwargs``."""
        if ocr_layouts is not None and len(ocr_layouts) != len(image_paths):
            raise ValueError(
                f"ocr_layouts has {len(ocr_layouts)} entries, expected {len(image_paths)}"
            )
        preds = []
        for i, path in enumerate(image_paths, 1):
            per_image = {"ocr_layout": ocr_layouts[i - 1]} if ocr_layouts is not None else {}
            preds.append(self.predict(path, mode=mode, **per_image, **kwargs))
            if i % 10 == 0 or i == len(image_paths):
                print(f"  predicted {i}/{len(image_paths)}")
        return preds

    def close(self) -> None:
        """Release the weights and empty the CUDA cache.

        Call this before loading another model in the same session. A notebook
        that runs zero-shot, then trains, then loads the adapter holds three
        model copies at once otherwise -- reliably an OOM on a small card, and it
        happens *after* the training run rather than before it.
        """
        self._model = None
        self._processor = None
        self._vllm = None
        free_memory()


def predictions_dict(preds: list[Prediction]) -> dict[str, str]:
    """Shape ``predict_many`` output for ``evaluate_predictions`` (uid -> html)."""
    return {p.uid: p.html for p in preds}


def free_memory() -> None:
    """Drop unreferenced tensors and return VRAM to the allocator.

    Deleting a Python reference is not enough on its own -- PyTorch's caching
    allocator holds the blocks until ``empty_cache()``, and ``nvidia-smi`` will
    keep showing the memory as used.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def gpu_report() -> str:
    """One-line VRAM summary. Print it between stages to catch a leak early."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device"
        free, total = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        return (
            f"{props.name} | {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB used | "
            f"bf16={torch.cuda.is_bf16_supported()}"
        )
    except ImportError:
        return "torch not installed"


def visual_token_estimate(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> int:
    """Approximate the visual tokens an image will cost.

    Useful for spotting sequence-length blowups before a run OOMs rather than
    after 40 minutes of training.
    """
    pixels = min(width * height, max_pixels)
    return int(pixels / (28 * 28))
