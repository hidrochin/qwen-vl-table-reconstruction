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

from dataclasses import dataclass, field
from statistics import median, pstdev

from src.data.valuetypes import NUMBER_KINDS, classify_kind, induce_kind
from src.ocr.engine import OcrWord

# Value kinds that mark a cell as carrying *data* rather than a label -- used to
# tell a sparse data row (holds a value) from a section row (holds only prose).
_VALUE_KINDS = NUMBER_KINDS | {"symbol", "date"}


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


# --- the geometric survey (pipeline_design.md §4.2, stage 3) -------------------
#
# The clustering above answers "which words share a row/column"; the survey adds
# the measurements schema discovery and verification reason over: how each track
# aligns (a general typographic fact -- numbers right-align, prose left-aligns --
# not a layout rule), the line pitch, which tracks are *anchors* (filled in nearly
# every row, so they seed logical rows), each row's type by its fill signature,
# and the fragments that straddle a column boundary. Nothing here encodes a
# specific layout; every signal is induced from this document's own geometry.

# Soft, family-level lexicon for summary rows. Used ONLY as a tie-break hint on a
# row already shaped like a data row -- never as a rule, and reconstruction never
# depends on it (see layout_description.md: summary lexicon is a soft signal).
SUMMARY_HINTS = frozenset({"total", "totals", "sum", "subtotal", "sub-total", "final", "balance", "grand total"})


@dataclass
class Track:
    """One vertical alignment track (a logical column, geometrically)."""

    index: int
    x1: float
    x2: float
    align: str  # 'left' | 'right' | 'center' -- the axis with least edge variance
    kind: str  # induced value kind over the track's body values
    fill: int  # number of rows carrying a value in this track
    n_rows: int  # rows considered, so fill/n_rows is the occupancy fraction

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def occupancy(self) -> float:
        return self.fill / self.n_rows if self.n_rows else 0.0


@dataclass
class RowInfo:
    """One visual row: its per-track cell text and its inferred type."""

    index: int
    cy: float
    cells: list[str]  # length == number of tracks; "" where the track is empty
    kind: str  # 'header' | 'section' | 'summary' | 'data' | 'empty'

    @property
    def filled(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c.strip()]


@dataclass
class Survey:
    """The full geometric survey -- the evidence schema discovery conditions on."""

    tracks: list[Track]
    rows: list[RowInfo]
    pitch: float
    anchors: list[int] = field(default_factory=list)  # anchor track indices
    contested: list[tuple[OcrWord, list[int]]] = field(default_factory=list)

    @property
    def n_cols(self) -> int:
        return len(self.tracks)


def estimate_pitch(words: list[OcrWord], row_tol: float = 0.6) -> float:
    """Document line pitch: the median gap between consecutive row centers.

    The line rhythm is what makes row segmentation adaptive instead of a fixed
    tolerance -- a wrapped line sits within a pitch of the one above, a real gap
    is a multiple of it. Falls back to the median glyph height for a one-row page.
    """
    rows = cluster_rows(words, row_tol)
    centers = sorted(sum(w.cy for w in r) / len(r) for r in rows)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 0]
    return median(gaps) if gaps else _median_height(words)


def _alignment(words: list[OcrWord]) -> str:
    """Which edge a column aligns on -- the axis of least variance."""
    if len(words) < 2:
        return "left"
    var = {
        "left": pstdev([w.x for w in words]),
        "right": pstdev([w.x + w.w for w in words]),
        "center": pstdev([w.cx for w in words]),
    }
    return min(var, key=var.get)


def _grid_words(words: list[OcrWord], separators: list[float], rows: list[list[OcrWord]]):
    """Bucket each row's words into columns -> rows x cols lists of ``OcrWord``."""
    n_cols = len(separators) + 1
    grid: list[list[list[OcrWord]]] = []
    for row in rows:
        cols: list[list[OcrWord]] = [[] for _ in range(n_cols)]
        for w in row:
            cols[_column_index(w.cx, separators)].append(w)
        grid.append(cols)
    return grid


def alignment_tracks(
    words: list[OcrWord], row_tol: float = 0.6, col_gap: float = 1.0
) -> list[Track]:
    """Vertical tracks with an induced alignment type and value kind.

    One :class:`Track` per whitespace-separated column: its x-extent, the edge it
    aligns on, the value kind induced from its own contents, and how many rows
    fill it. Alignment and kind are general signals (right-aligned numerics,
    left-aligned prose), never a layout assumption.
    """
    if not words:
        return []
    separators = detect_column_separators(words, col_gap)
    rows = cluster_rows(words, row_tol)
    grid = _grid_words(words, separators, rows)
    n_cols = len(separators) + 1

    col_words: list[list[OcrWord]] = [[] for _ in range(n_cols)]
    fill = [0] * n_cols
    for row_cols in grid:
        for i, cell in enumerate(row_cols):
            if cell:
                fill[i] += 1
                col_words[i].extend(cell)

    tracks: list[Track] = []
    for i, cw in enumerate(col_words):
        if not cw:
            continue
        tracks.append(
            Track(
                index=i,
                x1=min(w.x for w in cw),
                x2=max(w.x + w.w for w in cw),
                align=_alignment(cw),
                kind=induce_kind([w.text for w in cw]),
                fill=fill[i],
                n_rows=len(rows),
            )
        )
    return tracks


def discover_anchors(tracks: list[Track], min_occupancy: float = 0.8) -> list[int]:
    """Anchor tracks: those filled in nearly every row, so they seed logical rows.

    Discovered, not assumed -- the design forbids "the first column is always the
    label". If no track is dense enough (a genuinely sparse table), returns the
    single densest track so row seeding still has something to key on.
    """
    if not tracks:
        return []
    dense = [t.index for t in tracks if t.occupancy >= min_occupancy]
    if dense:
        return dense
    return [max(tracks, key=lambda t: t.occupancy).index]


def _has_value(cells: list[str], filled: list[int]) -> bool:
    """True if any filled cell carries value-typed (non-prose) content."""
    return any(classify_kind(cells[i]) in _VALUE_KINDS for i in filled)


def _classify_row(cells: list[str], anchors: list[int]) -> str:
    """Row type from its fill signature (header handled by ``survey`` position).

    A **section** row holds only prose in the leading/anchor track(s); a
    legitimately **sparse data** row is distinguished precisely by carrying
    value-typed content (design §4.2), so it is never mistyped as a section even
    when only one column is filled. A summary label wins over both.
    """
    filled = [i for i, c in enumerate(cells) if c.strip()]
    if not filled:
        return "empty"
    if cells[min(filled)].strip().lower() in SUMMARY_HINTS:
        return "summary"
    anchor_set = set(anchors) or {0}
    if set(filled) <= anchor_set and not _has_value(cells, filled):
        return "section"
    return "data"


def contested_fragments(
    words: list[OcrWord], col_gap: float = 1.0, margin_frac: float = 0.5
) -> list[tuple[OcrWord, list[int]]]:
    """Fragments sitting near a column boundary -- carried, not silently decided.

    A fragment is contested when its center is within ``margin_frac`` line-heights
    of a whitespace corridor: a small perturbation would flip its column, so the
    assignment is genuinely ambiguous between the two neighbouring tracks. (Note a
    fragment cannot *span* a corridor -- a corridor only exists where no fragment's
    x-interval reaches -- so proximity, not overlap, is the right signal.) The
    survey surfaces these with both candidate track indices for stage-7 to resolve
    semantically.
    """
    if not words:
        return []
    separators = detect_column_separators(words, col_gap)
    if not separators:
        return []
    margin = _median_height(words) * margin_frac
    out: list[tuple[OcrWord, list[int]]] = []
    for w in words:
        for k, sep in enumerate(separators):
            if abs(w.cx - sep) <= margin:
                out.append((w, [k, k + 1]))
                break
    return out


def survey(words: list[OcrWord], row_tol: float = 0.6, col_gap: float = 1.0) -> Survey:
    """Run the full geometric survey over a fragment set.

    Bundles tracks (typed + aligned), rows (typed by fill signature, with header
    rows identified by their position above the first data/section row), the line
    pitch, the discovered anchor tracks, and the contested fragments. This is the
    evidence dossier ``model/schema_infer.py`` reasons over.
    """
    tracks = alignment_tracks(words, row_tol, col_gap)
    anchors = discover_anchors(tracks)
    separators = detect_column_separators(words, col_gap)
    word_rows = cluster_rows(words, row_tol)
    grid = build_grid(words, row_tol, col_gap)

    rows: list[RowInfo] = []
    for i, (cells, wr) in enumerate(zip(grid, word_rows)):
        cy = sum(w.cy for w in wr) / len(wr) if wr else 0.0
        rows.append(RowInfo(index=i, cy=cy, cells=cells, kind=_classify_row(cells, anchors)))

    # Header rows label typed columns with prose, so a header sits ABOVE the first
    # row whose cells actually match the tracks' value kinds, and it fills several
    # tracks (a section row fills only the leading one). A header row can look like
    # "data" by fill alone -- this is what separates the two.
    def looks_body(row: RowInfo) -> bool:
        return any(
            t.kind in _VALUE_KINDS
            and t.index < len(row.cells)
            and classify_kind(row.cells[t.index]) == t.kind
            for t in tracks
        )

    first_body = next((i for i, r in enumerate(rows) if looks_body(r)), len(rows))
    for r in rows[:first_body]:
        if r.kind != "empty" and len(r.filled) >= 2 and not looks_body(r):
            r.kind = "header"

    # Re-induce each track's kind AND alignment from BODY words only (design §4.2):
    # a header label ("Number", "Type") must count toward neither the column's
    # value kind nor its measured alignment.
    body_words: dict[int, list[OcrWord]] = {}
    for r, wr in zip(rows, word_rows):
        if r.kind in ("header", "empty"):
            continue
        for w in wr:
            body_words.setdefault(_column_index(w.cx, separators), []).append(w)
    for t in tracks:
        bw = body_words.get(t.index)
        if bw:
            t.kind = induce_kind([w.text for w in bw])
            t.align = _alignment(bw)

    return Survey(
        tracks=tracks,
        rows=rows,
        pitch=estimate_pitch(words, row_tol),
        anchors=anchors,
        contested=contested_fragments(words, col_gap),
    )
