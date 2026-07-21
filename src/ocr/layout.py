"""Turn a bag of OCR words into a spatial layout the VLM can read.

This is the "preserve the structure in plain text" step -- a **deterministic
algorithm, not a model**. It clusters OCR boxes into rows and columns from pure
geometry, then serializes that either as an aligned text grid or as coordinate
lines. That serialization is what grounds the VLM in Option A: the model no
longer has to read or localize, only to decide the *structure* (which cells
merge, which rows are headers).

Two products come out of the same clustering:

* ``serialize_layout`` -- the grounding text passed alongside the image.
* ``to_grid_html`` -- a zero-model rectangular table. It has no merged cells
  (spans are exactly what the VLM adds), so it is a **floor** to measure the
  grounded VLM against, not an end in itself.

Pure Python, no torch/OpenCV, so it runs and is tested on the Mac. The two
thresholds are heuristics tuned for printed tables; expect to adjust
``row_tol`` / ``col_gap`` once real invoices are in hand -- they are arguments,
not constants, for that reason.
"""

from __future__ import annotations

from statistics import median

from src.ocr.engine import OcrWord


def _median_height(words: list[OcrWord]) -> float:
    heights = [w.h for w in words if w.h > 0]
    return median(heights) if heights else 1.0


def cluster_rows(words: list[OcrWord], row_tol: float = 0.6) -> list[list[OcrWord]]:
    """Group words into visual rows by vertical center.

    Greedy over words sorted top-to-bottom: a word joins the current row while
    its center stays within ``row_tol`` line-heights of the row's running mean,
    else it opens a new row. The running mean (rather than the first word's y)
    tolerates the slight slant of a photographed page. Each row is returned
    left-to-right.
    """
    if not words:
        return []
    tol = _median_height(words) * row_tol
    ordered = sorted(words, key=lambda w: w.cy)

    rows: list[list[OcrWord]] = []
    current = [ordered[0]]
    running_cy = ordered[0].cy
    for w in ordered[1:]:
        if abs(w.cy - running_cy) <= tol:
            current.append(w)
            running_cy = sum(x.cy for x in current) / len(current)
        else:
            rows.append(current)
            current = [w]
            running_cy = w.cy
    rows.append(current)

    for row in rows:
        row.sort(key=lambda w: w.x)
    return rows


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    ordered = sorted(intervals)
    merged: list[list[float]] = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def detect_column_separators(words: list[OcrWord], col_gap: float = 1.0) -> list[float]:
    """Find vertical-whitespace corridors that separate columns.

    Projects every word's horizontal span onto the x-axis, merges the spans,
    and treats each remaining gap wider than ``col_gap`` line-heights as a
    column boundary (returned as its midpoint x). Because it keys on whitespace
    that runs *through the table*, multi-word cells ("Total Amount") stay in one
    column -- the killer weakness of clustering on word left-edges instead.
    """
    if not words:
        return []
    min_gap = _median_height(words) * col_gap
    merged = _merge_intervals([(w.x, w.x + w.w) for w in words])

    separators: list[float] = []
    for left, right in zip(merged, merged[1:]):
        if right[0] - left[1] > min_gap:
            separators.append((left[1] + right[0]) / 2)
    return separators


def _column_index(cx: float, separators: list[float]) -> int:
    idx = 0
    for sep in separators:
        if cx > sep:
            idx += 1
        else:
            break
    return idx


def build_grid(
    words: list[OcrWord], row_tol: float = 0.6, col_gap: float = 1.0
) -> list[list[str]]:
    """Lay words onto a rectangular ``rows x columns`` grid of cell strings.

    Each cell holds the words assigned to it, left-to-right, space-joined.
    Empty where a row has no word in that column. This is the shared primitive
    behind both the text serialization and the HTML floor.
    """
    rows = cluster_rows(words, row_tol)
    if not rows:
        return []
    separators = detect_column_separators(words, col_gap)
    n_cols = len(separators) + 1

    grid: list[list[str]] = []
    for row in rows:
        cells: list[list[OcrWord]] = [[] for _ in range(n_cols)]
        for w in row:
            cells[_column_index(w.cx, separators)].append(w)
        grid.append([" ".join(x.text for x in sorted(c, key=lambda w: w.x)) for c in cells])
    return grid


def to_grid_text(words: list[OcrWord], row_tol: float = 0.6, col_gap: float = 1.0) -> str:
    """Pipe-delimited, column-aligned text grid -- the default grounding form.

    Reads like the table itself, so the VLM can see which values line up in a
    column without inferring it from pixels. Columns are padded to a common
    width per column for legibility.
    """
    grid = build_grid(words, row_tol, col_gap)
    if not grid:
        return ""
    n_cols = max(len(r) for r in grid)
    widths = [0] * n_cols
    for row in grid:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    for row in grid:
        padded = [row[i].ljust(widths[i]) if i < len(row) else " " * widths[i] for i in range(n_cols)]
        lines.append(" | ".join(padded).rstrip())
    return "\n".join(lines)


def to_coord_lines(words: list[OcrWord], row_tol: float = 0.6) -> str:
    """One line per word: text plus its box, in reading order.

    The alternative grounding form for when the model should reason from raw
    coordinates rather than a pre-aligned grid -- useful on irregular layouts
    where the column projection is unreliable.
    """
    lines = []
    for row in cluster_rows(words, row_tol):
        for w in row:
            lines.append(f"{w.text}\t(x={w.x:.0f}, y={w.y:.0f}, w={w.w:.0f}, h={w.h:.0f})")
    return "\n".join(lines)


def to_grid_html(words: list[OcrWord], row_tol: float = 0.6, col_gap: float = 1.0) -> str:
    """Zero-model rectangular HTML from geometry alone -- the floor to beat.

    No rowspan/colspan: those are precisely what a real table needs and what the
    VLM is there to add. Scoring this with TEDS-Struct tells you how much lift
    the model actually provides over naive gridding.
    """
    grid = build_grid(words, row_tol, col_gap)
    if not grid:
        return ""
    from xml.sax.saxutils import escape

    parts = ["<table>"]
    for row in grid:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def serialize_layout(
    words: list[OcrWord], style: str = "grid", row_tol: float = 0.6, col_gap: float = 1.0
) -> str:
    """Grounding text for the prompt. ``style`` is ``"grid"`` or ``"coords"``."""
    if style == "grid":
        return to_grid_text(words, row_tol, col_gap)
    if style == "coords":
        return to_coord_lines(words, row_tol)
    raise ValueError(f"style must be 'grid' or 'coords', got {style!r}")
