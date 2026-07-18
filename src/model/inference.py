"""Qwen3-VL inference for table reconstruction.

Heavy imports are deferred into function bodies so this module imports cleanly on
a Mac with no CUDA -- data prep and evaluation stay runnable locally while only
generation needs the GPU box.

``max_pixels`` is the setting that matters most here. Qwen-VL tokenizes images at
dynamic resolution, and a full-page table can exceed 4,000 visual tokens, which
is enough to OOM a 16 GB card or make each step crawl. Capping it trades
fine-detail resolution for throughput; for table *structure* that trade is
usually favourable, but it is worth measuring rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.model.prompts import INSTRUCTIONS, build_messages, clean_prediction

# 1024 visual tokens. Each token covers a 28x28 patch with 2x2 merging.
DEFAULT_MAX_PIXELS = 1024 * 28 * 28
DEFAULT_MIN_PIXELS = 256 * 28 * 28

MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class Prediction:
    uid: str
    raw: str
    html: str


class TableReconstructor:
    """Wraps a Qwen3-VL checkpoint for HTML table generation.

    Set ``load_in_4bit`` on a 16 GB card. Pass ``adapter_path`` to load a trained
    LoRA adapter on top of the base weights.
    """

    def __init__(
        self,
        model_id: str = MODEL_4B,
        load_in_4bit: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        adapter_path: str | None = None,
        use_unsloth: bool = False,
    ):
        self.model_id = model_id
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._model = None
        self._processor = None
        self._load(load_in_4bit, adapter_path, use_unsloth)

    def _load(self, load_in_4bit: bool, adapter_path: str | None, use_unsloth: bool) -> None:
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

    def predict(
        self,
        image_path: str | Path,
        mode: str = "structure",
        max_new_tokens: int = 2048,
        instruction: str | None = None,
    ) -> Prediction:
        """Generate HTML for one table image.

        ``instruction`` overrides the mode's default prompt. Pass a string from
        the notebook to iterate on prompts -- editing ``prompts.py`` mid-session
        has no effect, because the module is already imported.
        """
        import torch
        from PIL import Image

        if instruction is None and mode not in INSTRUCTIONS:
            raise ValueError(f"mode must be one of {sorted(INSTRUCTIONS)}")

        image = Image.open(image_path).convert("RGB")
        messages = build_messages(image, mode, instruction)

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        raw = self._processor.decode(trimmed, skip_special_tokens=True)
        return Prediction(uid=Path(image_path).stem, raw=raw, html=clean_prediction(raw))

    def predict_many(
        self, image_paths: list[str | Path], mode: str = "structure", **kwargs
    ) -> list[Prediction]:
        """Sequential generation with progress. Batching VLMs across differing
        image sizes is fiddly and the eval set is small; clarity wins here."""
        preds = []
        for i, path in enumerate(image_paths, 1):
            preds.append(self.predict(path, mode=mode, **kwargs))
            if i % 10 == 0 or i == len(image_paths):
                print(f"  predicted {i}/{len(image_paths)}")
        return preds

    def close(self) -> None:
        """Release the weights and empty the CUDA cache.

        Call this before loading another model in the same session. A notebook
        that runs zero-shot, then trains, then loads the adapter holds three
        model copies at once otherwise -- reliably an OOM on a 16 GB card, and it
        happens *after* the training run rather than before it.
        """
        self._model = None
        self._processor = None
        free_memory()


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
