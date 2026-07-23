"""Single source of truth for model roles across both execution tracks.

Every model here is **open-weights and self-hostable** -- there is no hosted-API
path anywhere in this repo. The two tracks draw from this one registry:

* **Track 1 -- public** (`data="public"`): Lightning AI free GPU, FinTabNet + the
  hand-drawn sample. Reproducible Python drivers.
* **Track 2 -- private** (`data="private"`): the company server, the 20
  confidential invoices. Notebooks. Nothing leaves the network.

`data="both"` means the model is allowed on either side (the specialists, the
small trainable students). The `data` field is the confidential-boundary gate --
a `private` model must never be pulled into a Track-1 (cloud) driver, and the
public anchor must never touch the invoices.

Heavy frameworks are never imported here; this module is pure metadata and stays
importable on the Mac with nothing installed.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- reasoning VLMs (generative; run via inference.TableReconstructor or a local
#     vLLM endpoint through vllm_client) ------------------------------------------
MODEL_QWEN36_27B = "Qwen/Qwen3.6-27B"  # on-prem teacher: dense, vision, thinking
MODEL_QWEN36_35B = "Qwen/Qwen3.6-35B-A3B"  # prior teacher (MoE 35B/3B-active), swap-in
MODEL_QWEN3VL_30B = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # public anchor / ceiling
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"  # student / fallback anchor
MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"  # small trainable (Lightning ~2 h LoRA)

# --- document specialists (own pipelines: paddle_client / mineru_client) ---------
MODEL_MINERU = "opendatalab/MinerU2.5-Pro-2605-1.2B"  # best open Table-TEDS; stage-1
MODEL_PADDLEOCR_VL = "PaddlePaddle/PaddleOCR-VL-1.6"  # 0.9B overall-doc leader; stage-1
MODEL_DOTS_OCR = "rednote-hilab/dots.ocr"  # specialist alternate (1.7B)
MODEL_GLM_OCR = "zai-org/GLM-OCR"  # specialist alternate (MIT)


@dataclass(frozen=True)
class ModelSpec:
    """One model's role. ``data`` is the confidential-boundary gate; ``backend``
    says which code path runs it."""

    repo_id: str
    role: str
    data: str  # "public" (Track 1) | "private" (Track 2) | "both"
    hardware: str  # sizing note -- what it takes to run/serve
    backend: str  # "generative" (chat VLM) | "specialist" (doc-parse pipeline)
    trainable: bool  # LoRA-fine-tunable via Unsloth


# The one registry. Keys are the short names drivers and notebooks iterate over.
MODELS: dict[str, ModelSpec] = {
    "qwen3.6-27b": ModelSpec(
        MODEL_QWEN36_27B, "on-prem teacher (stronger)", "private",
        "L40 48GB FP8 / vLLM", "generative", True,
    ),
    "qwen3.6-35b-a3b": ModelSpec(
        MODEL_QWEN36_35B, "prior teacher (swap-in)", "private",
        "L40 48GB FP8 / vLLM", "generative", False,
    ),
    "qwen3-vl-30b-a3b": ModelSpec(
        MODEL_QWEN3VL_30B, "public anchor / ceiling", "public",
        "Lightning 24GB 4-bit", "generative", False,
    ),
    "qwen3-vl-8b": ModelSpec(
        MODEL_8B, "student / fallback anchor", "both",
        "Lightning 24GB / RTX 5000", "generative", True,
    ),
    "qwen3-vl-4b": ModelSpec(
        MODEL_4B, "small trainable student", "both",
        "Lightning ~2 h LoRA", "generative", True,
    ),
    "mineru-2.5-pro": ModelSpec(
        MODEL_MINERU, "stage-1 OCR+geometry / table-SOTA ceiling", "both",
        "local 1.2B (T4 / on-prem)", "specialist", False,
    ),
    "paddleocr-vl-1.6": ModelSpec(
        MODEL_PADDLEOCR_VL, "stage-1 OCR / overall-doc leader", "both",
        "local 0.9B (T4 / on-prem)", "specialist", False,
    ),
    "dots-ocr": ModelSpec(
        MODEL_DOTS_OCR, "specialist alternate", "both",
        "local 1.7B", "specialist", False,
    ),
    "glm-ocr": ModelSpec(
        MODEL_GLM_OCR, "specialist alternate", "both",
        "local", "specialist", False,
    ),
}


def _allowed(spec: ModelSpec, data: str | None) -> bool:
    """A model is allowed on a track if its ``data`` is ``both`` or matches."""
    return data is None or spec.data in (data, "both")


def models_for_data(data: str) -> dict[str, ModelSpec]:
    """Every model runnable under a data boundary. ``data`` in {"public","private"}.

    This is the gate a driver calls so a Track-1 (cloud) run can never enumerate a
    ``private`` model, and vice versa.
    """
    if data not in ("public", "private"):
        raise ValueError(f"data must be 'public' or 'private', got {data!r}")
    return {name: spec for name, spec in MODELS.items() if _allowed(spec, data)}


def generative_models(data: str | None = None) -> dict[str, ModelSpec]:
    """Chat-VLM candidates (the bake-off set), optionally gated by data boundary."""
    return {
        name: spec
        for name, spec in MODELS.items()
        if spec.backend == "generative" and _allowed(spec, data)
    }


def specialist_models(data: str | None = None) -> dict[str, ModelSpec]:
    """Document-parser specialists (stage-1 OCR + the table-structure ceiling)."""
    return {
        name: spec
        for name, spec in MODELS.items()
        if spec.backend == "specialist" and _allowed(spec, data)
    }


def trainable_models(data: str | None = None) -> dict[str, ModelSpec]:
    """Models we can LoRA-fine-tune (students + the dense teacher)."""
    return {
        name: spec
        for name, spec in MODELS.items()
        if spec.trainable and _allowed(spec, data)
    }
