#!/usr/bin/env python
"""Run the staged reconstruction pipeline against a locally served vLLM endpoint.

Track-2 command-line companion to ``two-stage-reconstruct.ipynb``: same
``src.pipeline.StagedPipeline``, no notebook. On the company GPU box you serve the
teacher once (see ``RUNBOOK.md``)::

    vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 \\
        --no-enable-prefix-caching --limit-mm-per-prompt image=4

then reconstruct a table (nothing leaves the box -- the client hits loopback)::

    python run_pipeline.py data/invoices/0001.png --k 3
    python run_pipeline.py page1.png page2.png --out long_table.html   # one long table

It writes ``<image>_reconstructed.html`` and ``<image>_reconstructed.json`` (the
cell payload) beside the first image, prints the verifier's problems/flags and a
triage confidence, and with ``--draw`` saves a per-cell box overlay.

Only ``argparse``/``pathlib`` load at import time; the model client and pipeline
are imported inside ``main`` so ``--help`` works on any machine (the GPU deps are
needed only for a real run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.registry import MODEL_QWEN36_35B_FP8  # pure metadata, Mac-importable


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Staged logical table reconstruction over a served vLLM endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("images", nargs="+", help="one table image, or several page crops of ONE long table (reading order)")
    ap.add_argument("--model", default=MODEL_QWEN36_35B_FP8, help="served model id (must match `vllm serve`)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="the loopback vLLM endpoint")
    ap.add_argument("--k", type=int, default=1, help="cell-list vote drafts (k>1 votes for per-cell confidence)")
    ap.add_argument("--repair-rounds", type=int, default=1, help="max bounded, monotone repair rounds")
    ap.add_argument("--consensus", type=int, default=1, help="reading-pass consensus reads (extra tiling offsets)")
    ap.add_argument("--tile", type=int, default=1024, help="reading-pass tile size (px)")
    ap.add_argument("--max-side", type=int, default=1536, help="downscale the longer image edge before sending")
    ap.add_argument("--no-schema", action="store_true", help="skip schema discovery (Architecture-B baseline)")
    ap.add_argument("--no-thinking", action="store_true", help="disable the reasoning trace")
    ap.add_argument("--out", default=None, help="HTML output path (default: <first image>_reconstructed.html)")
    ap.add_argument("--draw", action="store_true", help="also save a per-cell box overlay PNG")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # GPU-box deps: imported here so `--help` / import works with nothing installed.
    from src.model.vllm_client import VLLMTableReconstructor
    from src.pipeline import StagedPipeline

    images = [str(Path(p)) for p in args.images]
    for p in images:
        if not Path(p).exists():
            print(f"error: image not found: {p}", file=sys.stderr)
            return 2

    client = VLLMTableReconstructor(
        model_id=args.model,
        base_url=args.base_url,
        max_side=args.max_side,
        thinking=not args.no_thinking,
    )
    # Extra reads shift the tiling grid so a fragment on a seam is read whole in
    # at least one; offsets ~10% of the tile keep them meaningfully different.
    offsets = tuple(int(i * args.tile * 0.1) for i in range(max(1, args.consensus)))
    pipe = StagedPipeline.from_client(
        client,
        k=args.k,
        use_schema_hint=not args.no_schema,
        max_repair_rounds=args.repair_rounds,
        tile=args.tile,
        consensus_offsets=offsets,
    )

    print(f"reading + reconstructing {len(images)} image(s) via {args.base_url} ...")
    result = pipe.run(images if len(images) > 1 else images[0])
    _report(result, images, args)
    return 0


def _report(result, images, args) -> None:
    import json

    payload = result.payload
    n_cells = len(payload.get("cells", []))
    cols = payload.get("columns", [])
    print("\n=== reconstruction ===")
    print(f"fragments read : {len(result.fragments)}")
    print(f"logical columns: {len(cols)}  " + ", ".join(str(c.get('name', '?')) for c in cols[:8]))
    print(f"cells          : {n_cells}")
    print(f"repair rounds  : {result.repair_rounds}")
    print(f"confidence     : {result.table_confidence}  (uncalibrated triage score)")
    if result.schema_hint:
        print("schema hint used:")
        print("  " + result.schema_hint.replace("\n", "\n  "))
    if result.report.problems:
        print(f"\nunresolved problems ({len(result.report.problems)}):")
        for p in result.report.problems[:15]:
            print("  -", p)
    if result.report.flags:
        print(f"\nflags for review ({len(result.report.flags)}):")
        for f in result.report.flags[:15]:
            print("  -", f)
    if result.notes:
        print("\nstage notes:")
        for n in result.notes:
            print("  .", n)

    out = Path(args.out) if args.out else Path(images[0]).with_name(
        Path(images[0]).stem + "_reconstructed.html"
    )
    out.write_text(result.html or "")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.json')}")

    if args.draw:
        try:
            from src.demo.boxes import boxed_cells, draw_cell_boxes

            overlay = draw_cell_boxes(images[0], result.html)
            box_path = out.with_name(out.stem + "_boxes.png")
            overlay.save(box_path)
            print(f"wrote {box_path}  ({len(boxed_cells(result.html))} cell boxes)")
        except Exception as exc:  # pragma: no cover - drawing is best-effort
            print(f"(box overlay skipped: {exc})", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
