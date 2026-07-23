"""Metrics beyond TEDS-Struct.

TEDS-Struct is the right primary number but it is not a number anyone outside
this project has intuition for. Span recovery is: "the model found 47 of 52
merged cells." Both get reported.

Bootstrap confidence intervals are included because the eval set is ~100 tables.
At that size a 2-3 point TEDS difference is noise, and a fine-tune that looks
better without overlapping intervals being checked is a claim that will not
survive contact with a larger set later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.html_utils import extract_cells


@dataclass
class SpanRecovery:
    matched: int
    total: int
    spurious: int

    @property
    def recall(self) -> float:
        return self.matched / self.total if self.total else 1.0

    @property
    def precision(self) -> float:
        predicted = self.matched + self.spurious
        return self.matched / predicted if predicted else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def span_recovery(pred_html: str, true_html: str) -> SpanRecovery:
    """Match spanning cells by grid position and span shape.

    Position-aware on purpose: a rowspan of the right size in the wrong place is
    not a recovered merged cell, and counting it as one would flatter the model
    exactly where the task is hardest.
    """
    def spans(html: str) -> set[tuple[int, int, int, int]]:
        return {
            (c.row, c.col, c.rowspan, c.colspan)
            for c in extract_cells(html)
            if c.is_spanning
        }

    true_spans = spans(true_html)
    pred_spans = spans(pred_html)
    matched = len(true_spans & pred_spans)
    return SpanRecovery(
        matched=matched,
        total=len(true_spans),
        spurious=len(pred_spans - true_spans),
    )


# --- Logical-reconstruction metrics (the schema-inference task) -----------------
#
# TEDS-Struct and span recovery measure *structure*. The invoice tables in
# ``layout_description.md`` also demand that every printed fragment lands in the
# right logical cell and that blank cells stay blank -- neither of which
# structure-only scoring can see. These three metrics are position-aware (keyed by
# the occupancy-grid anchor from ``extract_cells``) and are meaningful only for
# text-emitting predictions (mode ``schema`` / ``full``); on a structure-only run
# the prediction has no cell text, so content placement reads ~0 by design.


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _anchor_text_map(html: str) -> dict[tuple[int, int], str]:
    """Map each cell's grid anchor (row, col) to its normalized text."""
    return {(c.row, c.col): _norm(c.text) for c in extract_cells(html)}


def _grid_dims(html: str) -> tuple[int, int]:
    """(rows, cols) of the occupancy grid -- the logical schema's shape."""
    cells = extract_cells(html)
    if not cells:
        return (0, 0)
    rows = max(c.row + c.rowspan for c in cells)
    cols = max(c.col + c.colspan for c in cells)
    return (rows, cols)


@dataclass
class ContentPlacement:
    matched: int
    total: int

    @property
    def accuracy(self) -> float:
        """Fraction of true text fragments placed in the correct logical cell.

        1.0 when the ground truth has no text cells (nothing to place)."""
        return self.matched / self.total if self.total else 1.0


@dataclass
class BlankPreservation:
    preserved: int
    total: int

    @property
    def rate(self) -> float:
        """Fraction of truly-blank cells left blank -- catches value-shifting.

        1.0 when the ground truth has no blank cells."""
        return self.preserved / self.total if self.total else 1.0


@dataclass
class SchemaMatch:
    true_cols: int
    pred_cols: int
    true_rows: int
    pred_rows: int

    @property
    def cols_correct(self) -> bool:
        """Right number of logical columns -- e.g. did it get the Optional column?"""
        return self.true_cols == self.pred_cols

    @property
    def exact(self) -> bool:
        return self.true_cols == self.pred_cols and self.true_rows == self.pred_rows


def content_placement(pred_html: str, true_html: str) -> ContentPlacement:
    """Position-aware fragment placement: right text in the right logical cell.

    A value transcribed correctly but dropped in the wrong cell does not count --
    that is exactly the failure (value-shifting across sparse columns) the invoice
    task is most exposed to.
    """
    true_map = _anchor_text_map(true_html)
    pred_map = _anchor_text_map(pred_html)
    true_text = {pos: t for pos, t in true_map.items() if t}
    matched = sum(1 for pos, t in true_text.items() if pred_map.get(pos, "") == t)
    return ContentPlacement(matched=matched, total=len(true_text))


def blank_preservation(pred_html: str, true_html: str) -> BlankPreservation:
    """How many truly-blank cells stayed blank (no neighbouring value shifted in)."""
    true_map = _anchor_text_map(true_html)
    pred_map = _anchor_text_map(pred_html)
    true_blanks = [pos for pos, t in true_map.items() if not t]
    preserved = sum(1 for pos in true_blanks if pred_map.get(pos, "") == "")
    return BlankPreservation(preserved=preserved, total=len(true_blanks))


def schema_match(pred_html: str, true_html: str) -> SchemaMatch:
    """Compare the inferred logical schema (grid shape) to ground truth.

    ``cols_correct`` is the headline: getting the logical column count right means
    the model resolved the variable/implicit columns (the Optional column).
    """
    true_rows, true_cols = _grid_dims(true_html)
    pred_rows, pred_cols = _grid_dims(pred_html)
    return SchemaMatch(
        true_cols=true_cols, pred_cols=pred_cols, true_rows=true_rows, pred_rows=pred_rows
    )


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the mean via percentile bootstrap."""
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    lower = means[int(tail * n_resamples)]
    upper = means[min(int((1.0 - tail) * n_resamples), n_resamples - 1)]
    return (sum(values) / n, lower, upper)


def intervals_overlap(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True if two bootstrap intervals overlap.

    Overlapping intervals mean the difference is not distinguishable from noise
    at this sample size -- say so rather than reporting the gap as a result.
    """
    return not (a[2] < b[1] or b[2] < a[1])
