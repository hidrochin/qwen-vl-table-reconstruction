"""Layout-independent verification of a reconstructed cell list.

The reconstruction design (``pipeline_design.md`` §4.4) closes the loop with
checks that use *only general document truths* -- never a layout fact -- and that
were **held out of construction**, so they are genuine evidence rather than the
model grading itself. This module is that stage.

Two output channels, split by what the finding can *do*:

* ``problems`` -- machine-found errors the model can fix by rearranging its own
  output (a value in the wrong column, a cell that overlaps another, a fragment
  dropped). These are ``build_repair_suffix``-compatible strings, fed straight
  into the existing bounded, monotone repair loop in ``cells.py``.
* ``flags`` -- findings that need pixels or a human, not another guess: ink in a
  cell the table left blank (missed reading or a value shifted away -> targeted
  re-read), an arithmetic identity that nearly holds (a likely misread), and the
  model's own ``ocr_missed`` claims. Feeding these to the repair loop would make
  the model hallucinate a fix, so they drive the recall net and triage instead.

The checks each toggle off independently, which is exactly ablation #4 in the
design ("verification families off one at a time"). Heavy imports (PIL for the
ink check) are deferred, so the module imports on the Mac; the ink check simply
skips when no image is supplied or PIL is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from src.data.valuetypes import NUMBER_KINDS, induce_kind, kind_accepts, parse_number
from src.model.cells import build_repair_suffix, collect_ocr_missed, validate_cells

# Tunable, documented thresholds -- general heuristics, not layout constants.
_TYPE_DOMINANCE = 0.66  # a column must be >=2/3 one kind before an outlier is an error
_ARITH_EXACT = 0.005  # |total - sum(rest)| / scale below this = identity holds (silent)
_ARITH_NEAR = 0.08  # ...between exact and this = a near-miss worth flagging
_INK_DARK = 128  # L-channel value below which a pixel counts as ink
_INK_FRAC = 0.02  # fraction of dark pixels above which a region "has ink"


@dataclass
class VerificationReport:
    """Result of ``verify``: repairable ``problems`` and soft ``flags``."""

    problems: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing is repairable (flags may still be present)."""
        return not self.problems

    @property
    def n_findings(self) -> int:
        """Total findings -- an input to per-table confidence aggregation."""
        return len(self.problems) + len(self.flags)

    def repair_suffix(self, previous_json: str) -> str | None:
        """A ``build_repair_suffix`` string for the repair loop, or None if clean."""
        return build_repair_suffix(previous_json, self.problems) if self.problems else None


# --- occupancy grid over the payload ------------------------------------------


def _geom(cell: dict) -> tuple[int, int, int, int]:
    try:
        return (
            max(0, int(cell["r"])),
            max(0, int(cell["c"])),
            max(1, int(cell.get("rs", 1))),
            max(1, int(cell.get("cs", 1))),
        )
    except (KeyError, TypeError, ValueError):
        return (0, 0, 1, 1)


def _grid(payload: dict):
    """anchors {(r,c): cell}, occupied {(r,c)}, and grid dims -- the cells-JSON
    semantics where a blank is a hole (see ``cells.cells_to_html``)."""
    cells = [c for c in payload.get("cells", []) if isinstance(c, dict)]
    anchors: dict[tuple[int, int], dict] = {}
    occupied: set[tuple[int, int]] = set()
    for cell in sorted(cells, key=lambda c: _geom(c)[:2]):
        r, c, rs, cs = _geom(cell)
        if (r, c) in occupied:
            continue
        anchors[(r, c)] = cell
        occupied.update((r + dr, c + dc) for dr in range(rs) for dc in range(cs))
    n_rows = max((r + rs for r, _, rs, _ in map(_geom, cells)), default=0)
    n_cols = max(
        len(payload.get("columns") or []),
        max((c + cs for _, c, _, cs in map(_geom, cells)), default=0),
    )
    return anchors, occupied, n_rows, n_cols


def _box(cell: dict):
    b = cell.get("b")
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        return tuple(float(v) for v in b)
    return None


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _column_x(anchors: dict) -> dict[int, tuple[float, float]]:
    """Per-column horizontal interval, from single-column cells that carry a box."""
    xs: dict[int, list[tuple[float, float]]] = {}
    for (_, c), cell in anchors.items():
        _, _, _, cs = _geom(cell)
        box = _box(cell)
        if cs == 1 and box:
            xs.setdefault(c, []).append((box[0], box[2]))
    return {c: (min(a for a, _ in v), max(b for _, b in v)) for c, v in xs.items() if v}


def _column_centers(anchors: dict) -> dict[int, float]:
    """Median horizontal center per column, from single-column cells with a box.

    Median (not min/max) so a single shifted cell cannot widen its own column's
    range and thereby hide its shift -- the failure a naive interval would miss.
    """
    cxs: dict[int, list[float]] = {}
    for (_, c), cell in anchors.items():
        _, _, _, cs = _geom(cell)
        box = _box(cell)
        if cs == 1 and box:
            cxs.setdefault(c, []).append((box[0] + box[2]) / 2)
    return {c: median(v) for c, v in cxs.items() if v}


def _row_y(anchors: dict) -> dict[int, tuple[float, float]]:
    """Per-row vertical interval, from single-row cells that carry a box."""
    ys: dict[int, list[tuple[float, float]]] = {}
    for (r, _), cell in anchors.items():
        _, _, rs, _ = _geom(cell)
        box = _box(cell)
        if rs == 1 and box:
            ys.setdefault(r, []).append((box[1], box[3]))
    return {r: (min(a for a, _ in v), max(b for _, b in v)) for r, v in ys.items() if v}


# --- the check families -------------------------------------------------------


def check_geometry(payload: dict) -> list[str]:
    """Columns must be left-to-right ordered, and a cell's box must sit nearer its
    own column's center than any other's. Displaced ink betrays a shifted cell
    geometrically -- the design's "shifted-cell detector"."""
    anchors, _, _, _ = _grid(payload)
    col_center = _column_centers(anchors)
    problems: list[str] = []

    known = sorted(col_center)
    for a, b in zip(known, known[1:]):
        if col_center[a] >= col_center[b]:
            problems.append(
                f"column {a} sits at or right of column {b} horizontally "
                f"(centers {col_center[a]:.0f} >= {col_center[b]:.0f}) -- column order "
                "disagrees with the boxes"
            )

    if len(col_center) >= 2:
        for (r, c), cell in anchors.items():
            _, _, _, cs = _geom(cell)
            box = _box(cell)
            if cs != 1 or not box or c not in col_center:
                continue
            center = (box[0] + box[2]) / 2
            nearest = min(col_center, key=lambda k: abs(col_center[k] - center))
            if nearest != c:
                problems.append(
                    f"cell r={r} c={c} {_norm(cell.get('t', ''))[:30]!r}: its box center "
                    f"x={center:.0f} is closest to column {nearest}, not its column {c} (shifted?)"
                )
    return problems


def check_type_coherence(payload: dict) -> list[str]:
    """Every body value should fit its column's *induced* kind. Only columns with
    a clearly dominant discriminative kind flag outliers, so an ambiguous or
    prose column never over-flags."""
    anchors, _, _, _ = _grid(payload)
    by_col: dict[int, list[dict]] = {}
    for (_, c), cell in anchors.items():
        if not cell.get("h"):
            by_col.setdefault(c, []).append(cell)

    problems: list[str] = []
    for c, cells in by_col.items():
        values = [_norm(cell.get("t", "")) for cell in cells]
        values = [v for v in values if v]
        if len(values) < 3:
            continue
        kind = induce_kind(values)
        if kind not in NUMBER_KINDS and kind not in ("symbol", "date"):
            continue  # text/other columns accept anything
        accepted = sum(1 for v in values if kind_accepts(kind, v))
        if accepted / len(values) < _TYPE_DOMINANCE:
            continue  # no dominant kind -> not confident enough to call outliers
        for cell in cells:
            v = _norm(cell.get("t", ""))
            if v and not kind_accepts(kind, v):
                problems.append(
                    f"cell r={cell.get('r')} c={c} {v[:30]!r} does not fit column {c}'s "
                    f"induced kind '{kind}' -- likely the wrong column"
                )
    return problems


def check_arithmetic(payload: dict) -> list[str]:
    """Discover column totals and flag near-misses (soft).

    For a numeric column, if its largest value nearly equals the sum of the rest
    but not exactly, that is the signature of a shifted/misread value. An exact
    identity is silent; a column with no total-like value is silent. Conservative
    by design (returns *flags*, not repair problems) because a single table
    cannot validate an identity 'widely'."""
    anchors, _, _, _ = _grid(payload)
    by_col: dict[int, list[tuple[int, float]]] = {}
    for (r, c), cell in anchors.items():
        if cell.get("h"):
            continue
        val = parse_number(cell.get("t", ""))
        if val is not None:
            by_col.setdefault(c, []).append((r, val))

    flags: list[str] = []
    for c, rows in by_col.items():
        vals = [v for _, v in rows]
        if len(vals) < 4:  # too few values: max ~= sum(rest) happens by chance
            continue
        ordered = sorted(vals, key=abs, reverse=True)
        total, second = ordered[0], ordered[1]
        # A real total exceeds any single component with margin; without this a
        # plain numeric column (e.g. 123/234/345) coincidentally trips the band.
        if abs(total) < 1.2 * abs(second):
            continue
        rest = sum(vals) - total
        scale = max(abs(total), abs(rest), 1.0)
        diff = abs(total - rest) / scale
        if _ARITH_EXACT < diff <= _ARITH_NEAR:
            flags.append(
                f"column {c}: {total:g} looks like a total but the other values sum to "
                f"{rest:g} (off {diff * 100:.1f}%) -- check for a misread or shifted value"
            )
    return flags


def check_ink_in_blank(payload: dict, images) -> list[str]:
    """Every grid hole's implied region must be ink-free (soft).

    Ink where the table says blank means either a missed reading (recall failure)
    or a value shifted out of that cell -- both need pixels, not a model guess, so
    findings are flags that drive a targeted re-read. Skips silently if PIL is
    absent, no image is given, or a region cannot be resolved."""
    pages = _page_images(images)
    if not pages:
        return []
    anchors, occupied, n_rows, n_cols = _grid(payload)
    col_x = _column_x(anchors)
    row_y = _row_y(anchors)
    page_of_row = _dominant_page(anchors)

    flags: list[str] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in occupied or r not in row_y or c not in col_x:
                continue
            img = pages.get(page_of_row.get(r, 1))
            if img is None:
                continue
            x1, x2 = col_x[c]
            y1, y2 = row_y[r]
            if _has_ink(img, (x1, y1, x2, y2)):
                flags.append(
                    f"ink found in blank cell r={r} c={c} (region x={x1:.0f}..{x2:.0f}, "
                    f"y={y1:.0f}..{y2:.0f}) -- a value was missed or shifted away; re-read it"
                )
    return flags


def _dominant_page(anchors: dict) -> dict[int, int]:
    pages: dict[int, int] = {}
    for (r, _), cell in anchors.items():
        p = cell.get("p")
        if isinstance(p, int) and r not in pages:
            pages[r] = p
    return pages


def _page_images(images):
    """Normalize ``images`` to {page: PIL L-image}. Accepts a path/PIL, a list
    (page order), or a {page: ...} dict. Returns {} on any import/IO failure."""
    if images is None:
        return {}
    try:
        from PIL import Image
    except ImportError:
        return {}

    def load(x):
        try:
            if isinstance(x, (str,)) or hasattr(x, "__fspath__"):
                return Image.open(x).convert("L")
            if hasattr(x, "convert"):
                return x.convert("L")
        except Exception:
            return None
        return None

    if isinstance(images, dict):
        out = {int(k): load(v) for k, v in images.items()}
    elif isinstance(images, (list, tuple)):
        out = {i: load(v) for i, v in enumerate(images, 1)}
    else:
        out = {1: load(images)}
    return {k: v for k, v in out.items() if v is not None}


def _has_ink(img, box) -> bool:
    """True if the crop has more than ``_INK_FRAC`` dark pixels. Defensive."""
    try:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return False
        crop = img.crop((x1, y1, x2, y2))
        hist = crop.histogram()
        dark = sum(hist[: _INK_DARK])
        area = (x2 - x1) * (y2 - y1)
        return area > 0 and dark / area > _INK_FRAC
    except Exception:
        return False


def check_schema_echo(payload: dict) -> list[str]:
    """Every declared column should be used by some cell. A column declared but
    never filled is either a dropped column's values or an over-declared schema --
    both are schema-inference errors worth surfacing."""
    anchors, _, _, n_cols = _grid(payload)
    declared = len(payload.get("columns") or [])
    if not declared:
        return []
    used = {c for (_, c), cell in anchors.items() for c in range(_geom(cell)[1], _geom(cell)[1] + _geom(cell)[3])}
    problems = []
    for c in range(declared):
        if c not in used:
            name = ""
            cols = payload.get("columns") or []
            if c < len(cols) and isinstance(cols[c], dict):
                name = str(cols[c].get("name", ""))
            problems.append(
                f"column {c} {name!r} is declared but no cell is placed in it -- "
                "drop the column or place the values it should hold"
            )
    return problems


def verify(
    payload: dict | None,
    *,
    fragments: list | None = None,
    words: list | None = None,
    images=None,
    n_pages: int = 1,
    check_structure: bool = True,
    check_geom: bool = True,
    check_type: bool = True,
    check_schema: bool = True,
    check_arith: bool = True,
    check_ink: bool = True,
) -> VerificationReport:
    """Run the layout-independent checks and collect a ``VerificationReport``.

    ``check_structure`` runs ``cells.validate_cells`` (shape, occupancy, boxes,
    pages, and -- given ``fragments`` -- per-fragment identity, else the
    ``words`` token-bag). Geometry, type coherence, and schema-echo go to
    ``problems`` (repairable); arithmetic near-misses, ink-in-blank, and the
    model's ``ocr_missed`` claims go to ``flags``. Each family toggles for the
    ablation ladder.
    """
    report = VerificationReport()
    if not isinstance(payload, dict):
        report.problems.append("output is not a JSON object with 'columns' and 'cells'")
        return report

    if check_structure:
        report.problems.extend(
            validate_cells(payload, n_pages=n_pages, words=words, fragments=fragments)
        )
    if check_geom:
        report.problems.extend(check_geometry(payload))
    if check_type:
        report.problems.extend(check_type_coherence(payload))
    if check_schema:
        report.problems.extend(check_schema_echo(payload))
    if check_arith:
        report.flags.extend(check_arithmetic(payload))
    if check_ink:
        report.flags.extend(check_ink_in_blank(payload, images))
    for missed in collect_ocr_missed(payload):
        report.flags.append(
            f"ocr_missed claim {_norm(missed.get('t', ''))[:40]!r} at {missed.get('b')} "
            f"page {missed.get('p')} -- re-read to confirm"
        )
    return report
