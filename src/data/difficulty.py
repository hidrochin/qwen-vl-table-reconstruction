"""Deterministic table-difficulty scoring.

Selects the hard tables the demo is built around — merged cells, nested headers,
irregular layouts — straight from ground-truth HTML, with no manual labeling.
Reported per-bin, this is what supports the claim "here is how the model does on
the hard cases specifically" rather than a single average.

Border presence needs the image and lives in :mod:`src.data.borders`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from src.data.html_utils import extract_cells, parse_table

# Saturation points: the value at which a component contributes its full weight.
_SPAN_RATIO_FULL = 0.15
_HEADER_DEPTH_FULL = 3
_HOLE_RATIO_FULL = 0.05
_CELL_COUNT_FULL = 150

_WEIGHTS = {"spans": 0.40, "header": 0.25, "irregular": 0.15, "density": 0.20}

# Absolute thresholds are a fallback only. Measured on 3,000 FinTabNet tables the
# score distribution is p10=0.02, p50=0.12, p90=0.23 -- a fixed 0.55 cutoff
# matched 1 table in 3,000. Corpus selection therefore ranks by score and takes
# the top N (see loader.select_hard_tables), and bins are assigned by percentile
# within the selected set. Use `summarize()` before trusting either.
#
# Known corpus quirk: FinTabNet's HTML uses no <thead>/<th>, so `header_depth` is
# 0 for every row there and the "header" component contributes nothing. That
# compresses scores uniformly, which leaves the *ranking* -- all that top-N
# selection depends on -- unaffected. PubTabNet does carry <thead> and activates
# the component.
HARD_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.30

# Percentile cut points within a selected corpus: top 30% hard, next 30% medium.
HARD_PERCENTILE = 0.70
MEDIUM_PERCENTILE = 0.40


@dataclass(frozen=True)
class TableComplexity:
    n_rows: int
    n_cols: int
    n_cells: int
    n_spanning: int
    max_rowspan: int
    max_colspan: int
    header_depth: int
    hole_ratio: float
    span_ratio: float
    score: float
    bin: str

    def as_dict(self) -> dict:
        return asdict(self)


def _header_depth(html: str) -> int:
    """Rows belonging to the header: <thead> rows, else leading all-<th> rows."""
    table = parse_table(html)
    if table is None:
        return 0

    thead = table.find(".//thead")
    if thead is not None:
        return sum(1 for _ in thead.iter("tr"))

    depth = 0
    for tr in table.iter("tr"):
        tags = [el.tag for el in tr if el.tag in ("td", "th")]
        if tags and all(t == "th" for t in tags):
            depth += 1
        else:
            break
    return depth


def score_table(html: str) -> TableComplexity:
    """Score a single table's structural difficulty in [0, 1]."""
    cells = extract_cells(html)
    if not cells:
        return TableComplexity(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, "empty")

    n_rows = max(c.row + c.rowspan for c in cells)
    n_cols = max(c.col + c.colspan for c in cells)
    n_cells = len(cells)
    spanning = [c for c in cells if c.is_spanning]

    covered = sum(c.rowspan * c.colspan for c in cells)
    grid_area = n_rows * n_cols
    # Cells can overlap in malformed tables, so clamp at zero.
    hole_ratio = max(0.0, (grid_area - covered) / grid_area) if grid_area else 0.0

    span_ratio = len(spanning) / n_cells
    header_depth = _header_depth(html)

    components = {
        "spans": min(span_ratio / _SPAN_RATIO_FULL, 1.0),
        "header": min(max(header_depth - 1, 0) / (_HEADER_DEPTH_FULL - 1), 1.0),
        "irregular": min(hole_ratio / _HOLE_RATIO_FULL, 1.0),
        "density": min(n_cells / _CELL_COUNT_FULL, 1.0),
    }
    score = sum(_WEIGHTS[k] * v for k, v in components.items())

    return TableComplexity(
        n_rows=n_rows,
        n_cols=n_cols,
        n_cells=n_cells,
        n_spanning=len(spanning),
        max_rowspan=max((c.rowspan for c in cells), default=1),
        max_colspan=max((c.colspan for c in cells), default=1),
        header_depth=header_depth,
        hole_ratio=round(hole_ratio, 4),
        span_ratio=round(span_ratio, 4),
        score=round(score, 4),
        bin=_bin_for(score),
    )


def _bin_for(score: float) -> str:
    if score >= HARD_THRESHOLD:
        return "hard"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "easy"


def is_hard(html: str) -> bool:
    """Filter predicate for corpus selection."""
    return score_table(html).bin == "hard"


def assign_bins_by_percentile(
    complexities: list[TableComplexity],
) -> list[TableComplexity]:
    """Re-bin a selected corpus by rank rather than absolute score.

    Absolute thresholds depend on a dataset's annotation conventions; rank does
    not. "Hard" here means "hardest 30% of what was selected", which is the
    comparison the demo actually reports.
    """
    if not complexities:
        return []
    ordered = sorted(c.score for c in complexities)
    n = len(ordered)
    hard_cut = ordered[min(int(HARD_PERCENTILE * n), n - 1)]
    med_cut = ordered[min(int(MEDIUM_PERCENTILE * n), n - 1)]

    rebinned = []
    for c in complexities:
        if c.bin == "empty":
            rebinned.append(c)
            continue
        label = "hard" if c.score >= hard_cut else "medium" if c.score >= med_cut else "easy"
        rebinned.append(replace(c, bin=label))
    return rebinned


def summarize(scores: list[float]) -> dict:
    """Corpus score distribution — use it to sanity-check the thresholds.

    If almost everything lands in one bin, the saturation constants above are
    miscalibrated for this dataset; move them before trusting per-bin results.
    """
    if not scores:
        return {}
    ordered = sorted(scores)

    def pct(p: float) -> float:
        return round(ordered[min(int(p * len(ordered)), len(ordered) - 1)], 4)

    bins = [_bin_for(s) for s in scores]
    return {
        "n": len(scores),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "bins": {b: bins.count(b) for b in ("easy", "medium", "hard", "empty")},
    }
