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
