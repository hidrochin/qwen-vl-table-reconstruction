"""Track 1 -- LoRA-train a small student on Lightning AI (PUBLIC FinTabNet).

Thin driver over ``src.train.lora``. Trains a 4B/8B student that fits a free
24 GB GPU; the 27B teacher trains on the company server, not here. Keep
``--mode`` and any custom instruction matched to what the bake-off / inference
uses, or the adapter learns from a prompt it will never see.

Run:
    python notebook/train.py --corpus data/corpus --model qwen3-vl-4b --mode structure
    python notebook/train.py --corpus data/corpus --mode schema --grounded --epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_manifest
from src.model.registry import MODELS, trainable_models
from src.train.lora import TrainConfig, train


def _resolve_model(key_or_repo: str) -> str:
    """Accept a registry short-name (validated as trainable) or a raw repo id."""
    if key_or_repo in MODELS:
        spec = MODELS[key_or_repo]
        if not spec.trainable:
            raise SystemExit(f"{key_or_repo} is not marked trainable in the registry")
        return spec.repo_id
    return key_or_repo


def _layouts(records, grounded: bool) -> dict[str, str] | None:
    if not grounded:
        return None
    from src.ocr.engine import run_ocr
    from src.ocr.layout import serialize_layout

    return {r.uid: serialize_layout(run_ocr(r.image_path)) for r in records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--model", default="qwen3-vl-4b", help="registry key or repo id")
    ap.add_argument("--mode", default="structure", choices=["structure", "full", "schema"])
    ap.add_argument("--out", type=Path, default=Path("outputs/lora"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grounded", action="store_true", help="OCR-ground the training prompt")
    args = ap.parse_args()

    print("trainable models:", ", ".join(trainable_models("public")))
    records = load_manifest(args.corpus / "train")
    print(f"loaded {len(records)} train tables from {args.corpus / 'train'}")

    cfg = TrainConfig(
        model_id=_resolve_model(args.model),
        mode=args.mode,
        output_dir=str(args.out),
        epochs=args.epochs,
    )
    train(records, cfg, ocr_layouts=_layouts(records, args.grounded))


if __name__ == "__main__":
    main()
