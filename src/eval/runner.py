"""Evaluation orchestration: predict, score, aggregate, render.

Produces the one table the demo is built around -- TEDS-Struct by difficulty bin,
zero-shot versus fine-tuned -- which carries both "fine-tuning helps" and
"it handles the hard cases" at once.

Results are cached to JSON per run so the demo page can be rebuilt without
re-running generation. Nothing should need a GPU on demo day.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.loader import TableRecord
from src.eval.metrics import (
    blank_preservation,
    bootstrap_ci,
    content_placement,
    intervals_overlap,
    schema_match,
    span_recovery,
)
from src.eval.teds import teds_score


@dataclass
class Result:
    uid: str
    difficulty: str
    teds_struct: float
    span_recall: float
    span_f1: float
    n_spanning: int
    pred_html: str
    true_html: str
    image_path: str
    parse_failed: bool = False
    # Logical-reconstruction metrics (meaningful for schema/full modes; ~0 for a
    # structure-only run, which emits no cell text). Defaults keep load_run of
    # older run JSON working.
    content_placement: float = 0.0
    blank_preservation: float = 1.0
    schema_cols_correct: bool = False


@dataclass
class RunSummary:
    name: str
    n: int
    mean_teds: float
    ci_low: float
    ci_high: float
    mean_span_recall: float
    parse_failures: int
    by_bin: dict = field(default_factory=dict)
    mean_content_placement: float = 0.0
    mean_blank_preservation: float = 1.0
    schema_col_accuracy: float = 0.0


def evaluate_predictions(
    records: list[TableRecord],
    predictions: dict[str, str],
    run_name: str,
) -> tuple[list[Result], RunSummary]:
    """Score predictions (keyed by record uid) against ground truth."""
    results = []
    for rec in records:
        pred = predictions.get(rec.uid, "")
        recovery = span_recovery(pred, rec.html)
        placement = content_placement(pred, rec.html)
        blanks = blank_preservation(pred, rec.html)
        schema = schema_match(pred, rec.html)
        results.append(
            Result(
                uid=rec.uid,
                difficulty=rec.complexity.bin,
                teds_struct=teds_score(pred, rec.html, structure_only=True),
                span_recall=recovery.recall,
                span_f1=recovery.f1,
                n_spanning=rec.complexity.n_spanning,
                pred_html=pred,
                true_html=rec.html,
                image_path=rec.image_path,
                parse_failed=not pred.strip(),
                content_placement=placement.accuracy,
                blank_preservation=blanks.rate,
                schema_cols_correct=schema.cols_correct,
            )
        )
    return results, summarize_run(results, run_name)


def summarize_run(results: list[Result], name: str) -> RunSummary:
    if not results:
        return RunSummary(name, 0, 0.0, 0.0, 0.0, 0.0, 0)

    scores = [r.teds_struct for r in results]
    mean, low, high = bootstrap_ci(scores)
    n = len(results)

    by_bin = {}
    for label in ("easy", "medium", "hard"):
        subset = [r.teds_struct for r in results if r.difficulty == label]
        if subset:
            m, lo, hi = bootstrap_ci(subset)
            by_bin[label] = {"n": len(subset), "mean": m, "ci": [lo, hi]}

    return RunSummary(
        name=name,
        n=n,
        mean_teds=mean,
        ci_low=low,
        ci_high=high,
        mean_span_recall=sum(r.span_recall for r in results) / n,
        parse_failures=sum(r.parse_failed for r in results),
        by_bin=by_bin,
        mean_content_placement=sum(r.content_placement for r in results) / n,
        mean_blank_preservation=sum(r.blank_preservation for r in results) / n,
        schema_col_accuracy=sum(r.schema_cols_correct for r in results) / n,
    )


def compare_runs(baseline: RunSummary, candidate: RunSummary) -> str:
    """Format the headline comparison, flagging differences that are noise."""
    delta = candidate.mean_teds - baseline.mean_teds
    a = (baseline.mean_teds, baseline.ci_low, baseline.ci_high)
    b = (candidate.mean_teds, candidate.ci_low, candidate.ci_high)

    lines = [
        f"{'run':<22} {'n':>4} {'TEDS-Struct':>13} {'95% CI':>18} {'span recall':>12}",
        "-" * 74,
    ]
    for s in (baseline, candidate):
        lines.append(
            f"{s.name:<22} {s.n:>4} {s.mean_teds:>13.4f} "
            f"{'[' + format(s.ci_low, '.3f') + ', ' + format(s.ci_high, '.3f') + ']':>18} "
            f"{s.mean_span_recall:>12.4f}"
        )
    lines.append("-" * 74)
    lines.append(f"{'delta':<22} {'':>4} {delta:>+13.4f}")

    if intervals_overlap(a, b):
        lines.append(
            "\n  NOT SIGNIFICANT: confidence intervals overlap. At this sample size "
            "the\n  difference is indistinguishable from noise -- do not present it "
            "as a result."
        )
    else:
        direction = "improves over" if delta > 0 else "regresses against"
        lines.append(f"\n  {candidate.name} {direction} {baseline.name} (intervals disjoint).")

    # Schema-inference metrics (informative for schema/full runs; ~0 content
    # placement for structure-only runs, where the prediction has no cell text).
    lines.append(
        f"\n{'schema metric':<22} {'baseline':>10} {'candidate':>10}\n" + "-" * 44
    )
    for label, attr in (
        ("content placement", "mean_content_placement"),
        ("blank preservation", "mean_blank_preservation"),
        ("schema col accuracy", "schema_col_accuracy"),
    ):
        lines.append(
            f"{label:<22} {getattr(baseline, attr):>10.4f} {getattr(candidate, attr):>10.4f}"
        )

    if baseline.by_bin and candidate.by_bin:
        lines.append(f"\n{'bin':<10} {'n':>4} {'baseline':>10} {'candidate':>10} {'delta':>9}")
        lines.append("-" * 47)
        for label in ("easy", "medium", "hard"):
            if label in baseline.by_bin and label in candidate.by_bin:
                bm = baseline.by_bin[label]["mean"]
                cm = candidate.by_bin[label]["mean"]
                lines.append(
                    f"{label:<10} {baseline.by_bin[label]['n']:>4} "
                    f"{bm:>10.4f} {cm:>10.4f} {cm - bm:>+9.4f}"
                )
    return "\n".join(lines)


def save_run(results: list[Result], summary: RunSummary, out_dir: Path) -> Path:
    """Persist a run so the demo page can be rebuilt without a GPU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.name}.json"
    path.write_text(
        json.dumps(
            {"summary": asdict(summary), "results": [asdict(r) for r in results]}, indent=2
        )
    )
    return path


def load_run(path: Path) -> tuple[list[Result], RunSummary]:
    data = json.loads(Path(path).read_text())
    return (
        [Result(**r) for r in data["results"]],
        RunSummary(**data["summary"]),
    )


def to_cases(results: list[Result], limit: int | None = None, hardest_first: bool = True):
    """Convert results into renderer Cases, worst-scoring first by default.

    Showing the hard failures alongside the wins is deliberate: a demo reel with
    no failures invites the question of what was left out.
    """
    from src.demo.render import Case

    ordered = sorted(results, key=lambda r: r.teds_struct, reverse=not hardest_first)
    if limit:
        ordered = ordered[:limit]
    return [
        Case(
            uid=r.uid,
            image_path=r.image_path,
            pred_html=r.pred_html,
            true_html=r.true_html,
            score=r.teds_struct,
            difficulty=r.difficulty,
            n_spanning=r.n_spanning,
        )
        for r in ordered
    ]
