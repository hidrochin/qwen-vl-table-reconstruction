"""Track 1 -- score one model over the eval corpus (PUBLIC FinTabNet).

Runs a single generative model (base weights, or base + a trained LoRA adapter)
through the full metric suite -- TEDS-Struct, span recall, and the schema-
inference metrics (content placement, blank preservation, schema-column
accuracy) -- and optionally compares it to a previously saved run so a fine-tune
is judged against its baseline with the significance gate honoured.

Run:
    python notebook/eval.py --corpus data/corpus --model qwen3-vl-4b --mode structure
    python notebook/eval.py --corpus data/corpus --model qwen3-vl-4b \
        --adapter outputs/lora/adapter --mode schema --grounded \
        --baseline outputs/runs/qwen3-vl-4b.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_manifest
from src.eval.runner import compare_runs, evaluate_predictions, load_run, save_run
from src.model.inference import TableReconstructor, predictions_dict
from src.model.registry import MODELS


def _repo(key_or_repo: str) -> str:
    return MODELS[key_or_repo].repo_id if key_or_repo in MODELS else key_or_repo


def _layouts(records, grounded: bool) -> list[str] | None:
    if not grounded:
        return None
    from src.ocr.engine import run_ocr
    from src.ocr.layout import serialize_layout

    return [serialize_layout(run_ocr(r.image_path)) for r in records]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--model", default="qwen3-vl-4b", help="registry key or repo id")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir to load on top")
    ap.add_argument("--mode", default="structure", choices=["structure", "full", "schema"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--grounded", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--name", default=None, help="run name (defaults to the model key)")
    ap.add_argument("--baseline", type=Path, default=None, help="saved run JSON to compare against")
    args = ap.parse_args()

    records = load_manifest(args.corpus / "eval")
    if args.limit:
        records = records[: args.limit]
    image_paths = [r.image_path for r in records]

    # Load the same way the baseline was loaded -- keep quantization identical on
    # both arms (the repo's load-parity rule) when comparing base vs adapter.
    model = TableReconstructor(
        model_id=_repo(args.model),
        load_in_4bit=True,
        adapter_path=str(args.adapter) if args.adapter else None,
        thinking=(args.mode == "schema"),
    )
    preds = model.predict_many(image_paths, mode=args.mode, ocr_layouts=_layouts(records, args.grounded))

    name = args.name or (f"{args.model}+adapter" if args.adapter else args.model)
    results, summary = evaluate_predictions(records, predictions_dict(preds), name)
    save_run(results, summary, args.out)

    print(f"\n=== {name} over {summary.n} tables (mode={args.mode}) ===")
    print(f"TEDS-Struct         : {summary.mean_teds:.4f} [{summary.ci_low:.3f}, {summary.ci_high:.3f}]")
    print(f"span recall         : {summary.mean_span_recall:.4f}")
    print(f"content placement   : {summary.mean_content_placement:.4f}")
    print(f"blank preservation  : {summary.mean_blank_preservation:.4f}")
    print(f"schema col accuracy : {summary.schema_col_accuracy:.4f}")
    print(f"parse failures      : {summary.parse_failures}/{summary.n}")

    if args.baseline:
        _, base_summary = load_run(args.baseline)
        print("\n" + compare_runs(base_summary, summary))


if __name__ == "__main__":
    main()
