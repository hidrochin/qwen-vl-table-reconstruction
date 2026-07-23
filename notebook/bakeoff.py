"""Track 1 -- local-GPU bake-off on Lightning AI (PUBLIC data only).

No hosted API. Each candidate runs on the Lightning free GPU: generative VLMs
through ``TableReconstructor`` (the registry's public anchor + trainable
students), and the document specialists (MinerU2.5-Pro, PaddleOCR-VL) through
their own local pipelines to fix the table-structure ceiling. Everything is
scored through the existing TEDS-Struct + schema-inference + bootstrap pipeline,
then printed as a ranked ceiling table plus significance vs the public anchor.

Only ``data="public"`` models are enumerated -- the confidential teachers never
appear here, by construction (``registry.generative_models('public')``). This is
a cloud GPU, so it stays on FinTabNet + the hand-drawn sample.

Run:
    python notebook/bakeoff.py --corpus data/corpus --limit 40 --mode structure
    python notebook/bakeoff.py --corpus data/corpus --mode schema --grounded
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_manifest
from src.eval.runner import compare_runs, evaluate_predictions, save_run
from src.model.inference import TableReconstructor, free_memory, predictions_dict
from src.model.registry import generative_models, specialist_models

ANCHOR = "qwen3-vl-30b-a3b"  # the public-data ceiling everything is compared to

# Specialists that have a local client wired up. dots.ocr / GLM-OCR are in the
# registry as alternates but have no client yet, so the bake-off skips them.
_SPECIALIST_CLIENTS = {"mineru-2.5-pro", "paddleocr-vl-1.6"}


def _specialist(name: str):
    if name == "mineru-2.5-pro":
        from src.model.mineru_client import MinerUTableReconstructor

        return MinerUTableReconstructor()
    if name == "paddleocr-vl-1.6":
        from src.model.paddle_client import PaddleTableReconstructor

        return PaddleTableReconstructor()
    raise ValueError(f"no local client for specialist {name!r}")


def _maybe_layouts(image_paths: list[str], grounded: bool) -> list[str] | None:
    """Stage-1 OCR+geometry per image, or None when not grounding.

    Deferred import: OCR needs paddleocr, a GPU-box dependency.
    """
    if not grounded:
        return None
    from src.ocr.engine import run_ocr
    from src.ocr.layout import serialize_layout

    layouts = []
    for path in image_paths:
        layouts.append(serialize_layout(run_ocr(path)))
    return layouts


def run_bakeoff(
    corpus_dir: Path,
    limit: int | None,
    out_dir: Path,
    mode: str,
    grounded: bool,
    include_specialists: bool,
) -> None:
    records = load_manifest(corpus_dir / "eval")
    if limit:
        records = records[:limit]
    image_paths = [r.image_path for r in records]
    print(f"eval set: {len(records)} tables from {corpus_dir / 'eval'}  (mode={mode})\n")

    layouts = _maybe_layouts(image_paths, grounded)
    summaries = {}

    for name, spec in generative_models("public").items():
        print(f"=== {name}  ({spec.repo_id}, {spec.role}) ===")
        model = TableReconstructor(
            model_id=spec.repo_id,
            load_in_4bit=True,
            thinking=(mode == "schema"),  # reasoning helps schema inference
        )
        preds = model.predict_many(image_paths, mode=mode, ocr_layouts=layouts)
        results, summary = evaluate_predictions(records, predictions_dict(preds), name)
        save_run(results, summary, out_dir)
        summaries[name] = summary
        _print_row(summary)
        model.close()  # free VRAM before the next model -- 24 GB holds one at a time
        free_memory()

    if include_specialists:
        for name in specialist_models("public"):
            if name not in _SPECIALIST_CLIENTS:
                print(f"=== {name}: no local client yet, skipping ===\n")
                continue
            print(f"=== {name}  (specialist ceiling) ===")
            recon = _specialist(name)
            preds = recon.predict_many(image_paths)  # fixed parser, not prompt-steered
            results, summary = evaluate_predictions(records, predictions_dict(preds), name)
            save_run(results, summary, out_dir)
            summaries[name] = summary
            _print_row(summary)
            free_memory()

    _print_ranked(summaries)
    _print_significance(summaries)


def _print_row(s) -> None:
    print(
        f"  TEDS-Struct {s.mean_teds:.4f} [{s.ci_low:.3f}, {s.ci_high:.3f}]  "
        f"span-recall {s.mean_span_recall:.4f}  "
        f"placement {s.mean_content_placement:.3f}  "
        f"schema-cols {s.schema_col_accuracy:.3f}  "
        f"parse-fail {s.parse_failures}/{s.n}\n"
    )


def _print_ranked(summaries: dict) -> None:
    print("\n=== ranked (TEDS-Struct desc) ===")
    for name, s in sorted(summaries.items(), key=lambda kv: kv[1].mean_teds, reverse=True):
        print(f"  {name:<20} {s.mean_teds:.4f}  [{s.ci_low:.3f}, {s.ci_high:.3f}]")


def _print_significance(summaries: dict) -> None:
    if ANCHOR not in summaries:
        return
    print(f"\n=== vs anchor ({ANCHOR}) ===")
    for name, s in summaries.items():
        if name != ANCHOR:
            print("\n" + compare_runs(summaries[ANCHOR], s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--limit", type=int, default=None, help="cap eval tables for a fast pass")
    ap.add_argument("--mode", default="structure", choices=["structure", "full", "schema"])
    ap.add_argument("--grounded", action="store_true", help="stage-1 OCR grounding (needs paddleocr)")
    ap.add_argument("--no-specialists", action="store_true", help="skip MinerU / PaddleOCR-VL")
    args = ap.parse_args()
    run_bakeoff(
        args.corpus, args.limit, args.out, args.mode, args.grounded, not args.no_specialists
    )


if __name__ == "__main__":
    main()
