"""Structured JSON cell-list output -- the accuracy-first alternative to HTML.

The output format is free (the requirement is layout accuracy, not HTML), and a
flat cell list beats an HTML string on exactly the failures observed so far:

* **Boxes cannot be dropped.** ``cells_json_schema()`` marks ``b`` (bbox) and
  ``p`` (page) required on every cell, so with vLLM ``guided_json`` the model
  *cannot* emit a cell without them -- the "bboxes only on the first call"
  failure is impossible on this path rather than merely discouraged.
* **Blanks cannot be back-filled.** Only non-empty cells are listed; a blank is
  a hole in the grid, not a token the model can shift a neighbour into.
* **Schema before cells.** ``columns`` comes first in the object, so the model
  must commit to the logical column list (names + value kinds) before placing a
  single value -- the two-pass "infer schema, then assign fragments" idea in one
  generation, enforced by field order.
* **Machine-checkable.** ``validate_cells`` finds overlapping spans, cells past
  the declared column count, bad boxes, and (given the stage-1 OCR words)
  invented or dropped values -- feeding an optional single repair round-trip.

``cells_to_html`` converts the payload deterministically to the same HTML
dialect the rest of the repo speaks (``data-bbox``/``data-page`` included), so
TEDS-Struct, the schema metrics, and the box renderer all work unchanged.

Typical served-endpoint use::

    ins = build_cells_instruction()
    pred = client.predict(paths, instruction=ins, guided_json=cells_json_schema())
    payload = parse_cells(pred.raw)
    problems = validate_cells(payload, n_pages=len(paths), words=ocr_words)
    if problems:  # one repair round-trip, still a fresh single-turn request
        pred = client.predict(
            paths,
            instruction=ins + "\n\n" + build_repair_suffix(pred.raw, problems),
            guided_json=cells_json_schema(),
        )
        payload = parse_cells(pred.raw)
    html = cells_to_html(payload)  # -> existing eval / demo stack

Pure Python + stdlib ``json``; imports on the Mac with nothing installed.
"""

from __future__ import annotations

import json
import re

from src.data.valuetypes import COLUMN_KINDS  # single source of truth for kinds
from src.model.prompts import (  # shared prompt text -- edit it once, in prompts.py
    _CROPPED_TABLE_NOTE,
    _IGNORE_RULE,
    _SCHEMA_HINT_TEMPLATE,
    _SCHEMA_PRINCIPLE_ITEMS,
    number_principles,
)

_CELLS_INTRO = (
    "Reconstruct the latent LOGICAL table of the printed data table as a JSON "
    "cell list -- recover the logical table, not its visual appearance. If OCR "
    "text with positions is provided below the image(s), take each cell's text "
    "from the OCR (do not re-transcribe from pixels) and use the positions to "
    "place cells; otherwise read the values from the image."
)

# JSON-output replacement for the HTML-specific final principle in prompts.py.
_CELLS_HEADER_PRINCIPLE = (
    'Mark every header cell with "h": true and preserve every parent/child '
    "relationship with rs/cs spans. Preserve merged cells in headers, section "
    "rows, and summary rows."
)

_CELLS_OUTPUT = (
    "Output ONE JSON object and nothing else -- no markdown fences, no "
    "explanation:\n"
    '{"columns": [{"name": "<column name>", '
    '"kind": "<text|numeric|symbol|amount|percent|date|other>"}, ...],\n'
    ' "cells": [{"r": <row, 0-based>, "c": <column, 0-based>, "rs": <rowspan>, '
    '"cs": <colspan>, "h": <true for a header cell>, "t": "<cell text>", '
    '"b": [x1, y1, x2, y2], "p": <image number, 1-based>}, ...]}\n'
    "List columns FIRST: commit to the full logical schema -- including implicit "
    "or optional columns the body supports -- before any cell; name each column "
    "by its meaning and give its value kind. Then list ONLY non-empty cells, in "
    "reading order. A blank cell is simply absent from the list: never emit a "
    "blank cell and never move a neighbouring value into one. r and c index the "
    "logical grid (header rows are rows 0..n like any other row); rs and cs are "
    "the spans. b is the pixel box enclosing that cell's content in the image it "
    "appears on -- (x1,y1) top-left, (x2,y2) bottom-right; for OCR-grounded "
    "cells the union of the fragments assigned to that cell; keep every "
    "coordinate inside the image. p is which image the cell sits on (1 unless "
    "several images are given). Every cell MUST carry b and p, in every "
    "response."
)


# Fragment-grounded variant of the cell list (the reading-pass path). The model
# is handed a numbered fragment list (``src/ocr/read.py``) and references
# fragments by id instead of retyping text: text is looked up from the fragment,
# so inventing a value is impossible and dropping one is set arithmetic (the
# design's "hallucination impossible, drops computable"). See ``read.py``.
_CELLS_FRAGMENT_INTRO = (
    "Reconstruct the latent LOGICAL table of the printed data table as a JSON "
    "cell list -- recover the logical table, not its visual appearance. You are "
    "given a numbered list of OCR text fragments, each with an id and a position. "
    "Build the table by REFERENCING fragment ids: every non-empty logical cell "
    "lists the id(s) of the fragment(s) that fall in it, and the cell's text is "
    "looked up from those fragments. Do not retype any fragment's text."
)

_CELLS_FRAGMENT_OUTPUT = (
    "Output ONE JSON object and nothing else -- no markdown fences, no "
    "explanation:\n"
    '{"columns": [{"name": "<column name>", '
    '"kind": "<text|numeric|symbol|amount|percent|date|other>"}, ...],\n'
    ' "cells": [{"r": <row, 0-based>, "c": <column, 0-based>, "rs": <rowspan>, '
    '"cs": <colspan>, "h": <true for a header cell>, '
    '"f": [<fragment id>, ...]}, ...],\n'
    ' "ignored": [<fragment id>, ...],\n'
    ' "ocr_missed": [{"t": "<printed text you see but no fragment covers>", '
    '"b": [x1, y1, x2, y2], "p": <image number>}, ...]}\n'
    "List columns FIRST: commit to the full logical schema -- including implicit "
    "or optional columns the body supports -- before any cell. Then list ONLY "
    'non-empty cells, in reading order; each cell\'s "f" holds the fragment id(s) '
    "it is built from (left-to-right). A blank cell is simply absent from the "
    "list: never emit a blank cell and never move a fragment into one. EVERY "
    'fragment id must be accounted for exactly once: in some cell\'s "f", or in '
    '"ignored" if it is not part of the printed data table (stamps, QR/barcode '
    "text, handwriting, signatures). If you can see printed table text that no "
    'fragment covers, list it under "ocr_missed" so it can be re-read -- never '
    "invent a fragment id for it."
)


def build_cells_instruction(schema: str | None = None, with_fragments: bool = False) -> str:
    """Assemble the JSON cell-list instruction.

    Same skeleton as ``build_schema_instruction`` -- intro, cropped-table note,
    the shared reconstruction principles (with a JSON-output final item),
    ignore rule, optional header hint -- but the output contract is the cell
    list. ``schema`` is the same optional hint-to-verify; do not pass one unless
    the document family's header is known stable.

    ``with_fragments`` switches to the reading-pass-grounded contract: the model
    references OCR fragment ids (``f``) rather than retyping text, and accounts
    for every fragment (``ignored`` / ``ocr_missed``). Pair it with
    ``cells_json_schema(with_fragments=True)`` and pass the numbered fragment
    list as the OCR block; text is filled in later by ``fill_cells_from_fragments``.
    """
    principles = number_principles(_SCHEMA_PRINCIPLE_ITEMS[:-1] + (_CELLS_HEADER_PRINCIPLE,))
    intro = _CELLS_FRAGMENT_INTRO if with_fragments else _CELLS_INTRO
    output = _CELLS_FRAGMENT_OUTPUT if with_fragments else _CELLS_OUTPUT
    parts = [intro, _CROPPED_TABLE_NOTE, principles, _IGNORE_RULE]
    if schema:
        parts.append(_SCHEMA_HINT_TEMPLATE.format(schema=schema.strip()))
    parts.append(output)
    return "\n\n".join(parts)


CELLS_INSTRUCTION = build_cells_instruction()


def cells_json_schema(with_fragments: bool = False) -> dict:
    """JSON Schema for vLLM ``guided_json`` -- the structural guarantee.

    By default ``b`` and ``p`` are required on every cell, so a decode that drops
    boxes is not merely off-prompt, it is impossible. ``t`` has ``minLength: 1``
    because blank cells must be absent, not empty.

    With ``with_fragments=True`` the grounded contract applies: each cell must
    carry ``f`` (>=1 fragment id) and need not carry ``t``/``b``/``p`` (those are
    filled from the referenced fragments). Optional top-level ``ignored`` (ids the
    model excludes as non-table) and ``ocr_missed`` (text lacking a fragment)
    make the fragment partition explicit and drive the recall net.
    """
    cell_props: dict = {
        "r": {"type": "integer", "minimum": 0},
        "c": {"type": "integer", "minimum": 0},
        "rs": {"type": "integer", "minimum": 1},
        "cs": {"type": "integer", "minimum": 1},
        "h": {"type": "boolean"},
        "t": {"type": "string", "minLength": 1},
        "b": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "p": {"type": "integer", "minimum": 1},
    }
    if with_fragments:
        cell_props["f"] = {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 1,
        }
        cell_required = ["r", "c", "rs", "cs", "h", "f"]
    else:
        cell_required = ["r", "c", "rs", "cs", "h", "t", "b", "p"]

    schema: dict = {
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": list(COLUMN_KINDS)},
                    },
                    "required": ["name", "kind"],
                    "additionalProperties": False,
                },
            },
            "cells": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": cell_props,
                    "required": cell_required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["columns", "cells"],
        "additionalProperties": False,
    }
    if with_fragments:
        schema["properties"]["ignored"] = {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        }
        schema["properties"]["ocr_missed"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "string"},
                    "b": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "p": {"type": "integer", "minimum": 1},
                },
                "required": ["t", "b", "p"],
                "additionalProperties": False,
            },
        }
    return schema


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_cells(raw: str) -> dict | None:
    """Recover the payload from model output, tolerating the usual wrappers.

    Guided decoding yields clean JSON; the ungrounded/eager paths may add a
    thinking trace, markdown fences, or prose. Mirrors ``clean_prediction``:
    strip the trace, prefer a fenced block that mentions ``"cells"``, then trim
    to the outermost ``{...}``. Returns None when nothing parses -- score that
    as a failure rather than guessing.
    """
    if not raw:
        return None
    text = _THINK_RE.sub("", raw).strip()

    if "```" in text:
        for block in text.split("```")[1::2]:
            body = block.split("\n", 1)[-1] if block[:12].lower().startswith("json") else block
            if '"cells"' in body:
                text = body
                break

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _norm(text: str) -> str:
    return " ".join(text.split())


def _frag_id(fr) -> int:
    return fr["id"] if isinstance(fr, dict) else fr.id


def _frag_text(fr) -> str:
    return fr.get("text", "") if isinstance(fr, dict) else fr.text


def validate_cells(
    payload: dict | None,
    n_pages: int = 1,
    words: list | None = None,
    fragments: list | None = None,
) -> list[str]:
    """Machine-check a payload; returns human-readable violations (empty = ok).

    Checks: payload shape, cells outside the declared column range, overlapping
    span claims (the occupancy rule browsers use, as in
    ``html_utils.extract_cells``), degenerate or missing boxes, and page numbers
    beyond ``n_pages``.

    Grounding checks (mutually exclusive, strongest first):

    * ``fragments`` (reading-pass ``Fragment``/``OcrWord``-likes with ``.id`` and
      ``.text``) -- *per-fragment identity*. Every fragment id must be accounted
      for exactly once: referenced by one cell's ``f`` or listed in ``ignored``.
      A reference to an unknown id is an **invented** value (impossible to sneak
      past), an id referenced twice is **duplicated**, and an id neither used nor
      ignored is **dropped**. This is exact set arithmetic, not the fuzzy
      token-bag below.
    * ``words`` (anything with ``.text``) -- the older token-bag heuristic for the
      non-grounded path: flags OCR text in no cell (dropped) and cell tokens in
      no OCR fragment (invented). Used only when ``fragments`` is not given.

    The list is written to be fed back verbatim in a repair round-trip
    (``build_repair_suffix``). ``ocr_missed`` claims are *not* included here (they
    are recall triggers, not repairable identity errors) -- see
    ``collect_ocr_missed``.
    """
    if not isinstance(payload, dict):
        return ["output is not a JSON object with 'columns' and 'cells'"]
    columns = payload.get("columns")
    cells = payload.get("cells")
    if not isinstance(columns, list) or not columns:
        return ["'columns' is missing or empty"]
    if not isinstance(cells, list) or not cells:
        return ["'cells' is missing or empty"]

    problems: list[str] = []
    n_cols = len(columns)
    occupied: dict[tuple[int, int], str] = {}

    for cell in cells:
        if not isinstance(cell, dict):
            problems.append(f"cell entry is not an object: {cell!r}")
            continue
        label = f"cell r={cell.get('r')} c={cell.get('c')} t={_norm(str(cell.get('t', '')))[:30]!r}"
        try:
            r, c = int(cell["r"]), int(cell["c"])
            rs, cs = int(cell.get("rs", 1)), int(cell.get("cs", 1))
        except (KeyError, TypeError, ValueError):
            problems.append(f"{label}: r/c/rs/cs are not all integers")
            continue
        if r < 0 or c < 0 or rs < 1 or cs < 1:
            problems.append(f"{label}: negative position or span < 1")
            continue
        if c + cs > n_cols:
            problems.append(
                f"{label}: extends to column {c + cs - 1} but only {n_cols} columns are declared"
            )

        for dr in range(rs):
            for dc in range(cs):
                key = (r + dr, c + dc)
                if key in occupied:
                    problems.append(
                        f"{label}: overlaps grid position {key} already claimed by {occupied[key]}"
                    )
                else:
                    occupied[key] = label

        # On the grounded path a cell carries only 'f' (fragment ids); its text,
        # box and page are filled from the fragments later, so the t/b/p shape
        # checks do not apply -- identity is checked against the fragments instead.
        if fragments is not None and isinstance(cell.get("f"), list) and cell.get("f"):
            continue

        if not _norm(str(cell.get("t", ""))):
            problems.append(f"{label}: empty text -- blank cells must be omitted, not emitted")
        bbox = cell.get("b")
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(v, (int, float)) for v in bbox)
        ):
            problems.append(f"{label}: 'b' is not four numbers")
        elif bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            problems.append(f"{label}: degenerate bbox {bbox} (x2<=x1 or y2<=y1)")
        page = cell.get("p")
        if not isinstance(page, int) or page < 1 or page > n_pages:
            problems.append(f"{label}: 'p'={page!r} outside 1..{n_pages}")

    if fragments is not None:
        problems.extend(_check_fragment_identity(cells, fragments, payload))
    elif words:
        cell_texts = [
            _norm(str(c.get("t", ""))) for c in cells if isinstance(c, dict)
        ]
        joined = " ".join(cell_texts)
        cell_token_bag = set(joined.split())
        ocr_token_bag = {tok for w in words for tok in _norm(w.text).split()}
        for w in words:
            frag = _norm(w.text)
            if frag and frag not in joined:
                problems.append(f"OCR fragment {frag[:40]!r} was not assigned to any cell (dropped)")
        for tok in sorted(cell_token_bag - ocr_token_bag):
            problems.append(f"cell text token {tok[:40]!r} appears in no OCR fragment (invented)")

    return problems


def _check_fragment_identity(cells: list, fragments: list, payload: dict) -> list[str]:
    """Exact fragment partition check for the grounded path (see ``validate_cells``)."""
    problems: list[str] = []
    frag_by_id = {_frag_id(fr): fr for fr in fragments}
    ids = set(frag_by_id)
    used: dict[int, str] = {}

    for cell in cells:
        if not isinstance(cell, dict):
            continue
        label = f"cell r={cell.get('r')} c={cell.get('c')}"
        refs = cell.get("f")
        if not isinstance(refs, list) or not refs:
            problems.append(f"{label}: no fragment references 'f' -- every cell must cite its fragment id(s)")
            continue
        for fid in refs:
            if fid not in ids:
                problems.append(f"{label}: references unknown fragment id {fid} (invented value)")
            elif fid in used:
                problems.append(f"{label}: fragment id {fid} already used by {used[fid]} (duplicated)")
            else:
                used[fid] = label

    ignored = payload.get("ignored") or []
    ignored_set: set[int] = set()
    for fid in ignored:
        if fid not in ids:
            problems.append(f"'ignored' lists unknown fragment id {fid}")
        elif fid in used:
            problems.append(f"fragment id {fid} is both used by {used[fid]} and in 'ignored'")
        else:
            ignored_set.add(fid)

    for fid in sorted(ids - set(used) - ignored_set):
        txt = _norm(_frag_text(frag_by_id[fid]))[:40]
        problems.append(
            f"fragment id {fid} {txt!r} is neither used by a cell nor ignored (dropped)"
        )
    return problems


def collect_ocr_missed(payload: dict | None) -> list[dict]:
    """Return the payload's ``ocr_missed`` claims (text the model saw but no
    fragment covered). These are recall triggers -- feed each box to a targeted
    re-read (``ocr.read.reread_regions``), not to the repair loop.
    """
    if not isinstance(payload, dict):
        return []
    return [m for m in (payload.get("ocr_missed") or []) if isinstance(m, dict)]


def fill_cells_from_fragments(payload: dict | None, fragments: list) -> dict | None:
    """Populate ``t``/``b``/``p`` on fragment-referencing cells from the fragments.

    On the grounded path a cell carries ``f`` (fragment ids) instead of retyped
    text. This looks each cell's fragments up and sets ``t`` to their
    reading-order, space-joined text, ``b`` to the union pixel box
    (``[x1,y1,x2,y2]``), and ``p`` to the fragment page -- so the rest of the
    stack (``validate_cells`` structural checks, ``cells_to_html``, the metrics)
    consumes the payload unchanged. Cells that already carry ``t`` are left as-is
    (a mixed or non-grounded payload passes through untouched). Returns a new
    payload; the input is not mutated.
    """
    if not isinstance(payload, dict):
        return payload
    frag_by_id = {_frag_id(fr): fr for fr in fragments}
    out_cells = []
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            out_cells.append(cell)
            continue
        refs = cell.get("f")
        if _norm(str(cell.get("t", ""))) or not isinstance(refs, list) or not refs:
            out_cells.append(cell)
            continue
        frs = [frag_by_id[fid] for fid in refs if fid in frag_by_id]
        if not frs:
            out_cells.append(cell)
            continue
        frs.sort(key=lambda fr: fr.x)
        new = dict(cell)
        new["t"] = " ".join(_norm(fr.text) for fr in frs if _norm(fr.text))
        new["b"] = [
            min(fr.x for fr in frs),
            min(fr.y for fr in frs),
            max(fr.x + fr.w for fr in frs),
            max(fr.y + fr.h for fr in frs),
        ]
        new["p"] = getattr(frs[0], "page", cell.get("p", 1))
        out_cells.append(new)
    result = dict(payload)
    result["cells"] = out_cells
    return result


def _cell_key(cell: dict) -> tuple[str, ...] | str:
    """Content identity of a cell for voting: fragment ids if grounded, else text."""
    refs = cell.get("f")
    if isinstance(refs, list) and refs:
        return tuple(str(x) for x in refs)
    return _norm(str(cell.get("t", ""))).casefold()


def vote_cells(payloads: list[dict]) -> tuple[dict, dict[tuple[int, int], float]]:
    """Majority-vote k cell-list payloads into one consensus payload + confidence.

    Diverse drafts (different conditioning, not just temperature) decorrelate a
    single model's failures; voting keeps what most drafts agree on. Columns: the
    most common ``(name, kind)`` list wins. Cells: for each grid anchor ``(r, c)``
    a majority of drafts fill, the most common cell *content* (fragment ids, or
    text) wins and one matching cell object is emitted. Per-anchor confidence is
    the agreement fraction (votes for the winner / number of drafts) -- the
    entropy the design turns into per-cell confidence.

    Returns ``(payload, {(r, c): confidence})``. A single payload votes to
    itself with confidence 1.0 everywhere.
    """
    payloads = [p for p in payloads if isinstance(p, dict) and p.get("cells")]
    if not payloads:
        return ({"columns": [], "cells": []}, {})
    k = len(payloads)

    from collections import Counter

    col_votes = Counter(
        tuple((str(col.get("name", "")), str(col.get("kind", "")))
              for col in (p.get("columns") or []) if isinstance(col, dict))
        for p in payloads
    )
    best_cols = col_votes.most_common(1)[0][0]
    columns = [{"name": name, "kind": kind} for name, kind in best_cols]

    # anchor -> list of (content_key, cell_object), one entry per draft that fills it
    anchor_cells: dict[tuple[int, int], list[tuple]] = {}
    for p in payloads:
        seen: set[tuple[int, int]] = set()
        for cell in p.get("cells", []):
            if not isinstance(cell, dict):
                continue
            try:
                anchor = (int(cell["r"]), int(cell["c"]))
            except (KeyError, TypeError, ValueError):
                continue
            if anchor in seen:  # one vote per draft per anchor
                continue
            seen.add(anchor)
            anchor_cells.setdefault(anchor, []).append((_cell_key(cell), cell))

    cells: list[dict] = []
    confidence: dict[tuple[int, int], float] = {}
    for anchor, entries in anchor_cells.items():
        if len(entries) * 2 <= k:  # not a majority of drafts -> drop as noise
            continue
        key_votes = Counter(key for key, _ in entries)
        winning_key, votes = key_votes.most_common(1)[0]
        cell = next(c for key, c in entries if key == winning_key)
        cells.append(cell)
        confidence[anchor] = votes / k

    cells.sort(key=lambda c: (c.get("r", 0), c.get("c", 0)))
    return ({"columns": columns, "cells": cells}, confidence)


def build_repair_suffix(previous_json: str, problems: list[str]) -> str:
    """Suffix for one repair round-trip -- append to the SAME instruction.

    The repair request is still a fresh single-turn call (same images, same
    instruction, same guided schema); this suffix shows the model its previous
    answer and the machine-found violations so it fixes those and nothing else.
    """
    issues = "\n".join(f"- {p}" for p in problems)
    return (
        "A previous attempt at this exact task produced the JSON below, and "
        "automatic validation found the listed problems. Produce a corrected "
        "full JSON object: fix ONLY these problems and keep every other column "
        "and cell identical.\n\n"
        f"Previous attempt:\n{previous_json.strip()}\n\nProblems:\n{issues}"
    )


def cells_to_html(payload: dict | None) -> str:
    """Deterministic payload -> HTML, in the dialect the rest of the repo speaks.

    Non-empty cells become ``<th>``/``<td>`` with ``rowspan``/``colspan``,
    ``data-bbox="x1,y1,x2,y2"`` and (when any cell sits past page 1)
    ``data-page``; grid holes become empty ``<td>`` so blanks stay blank. The
    result feeds ``html_utils.extract_cells``, TEDS-Struct, the schema metrics,
    and ``demo.boxes.draw_cell_boxes`` unchanged. A cell whose anchor lands on
    a grid position already claimed by a span is skipped -- ``validate_cells``
    reports it; conversion stays total.
    """
    if not isinstance(payload, dict):
        return ""
    cells = [c for c in payload.get("cells", []) if isinstance(c, dict)]
    if not cells:
        return ""
    from xml.sax.saxutils import escape, quoteattr

    def geom(cell: dict) -> tuple[int, int, int, int]:
        try:
            return (
                max(0, int(cell["r"])),
                max(0, int(cell["c"])),
                max(1, int(cell.get("rs", 1))),
                max(1, int(cell.get("cs", 1))),
            )
        except (KeyError, TypeError, ValueError):
            return (0, 0, 1, 1)

    geoms = {id(c): geom(c) for c in cells}
    n_rows = max(r + rs for r, _, rs, _ in geoms.values())
    n_cols = max(
        len(payload.get("columns") or []),
        max(c + cs for _, c, _, cs in geoms.values()),
    )
    multi_page = any(isinstance(c.get("p"), int) and c["p"] > 1 for c in cells)

    anchors: dict[tuple[int, int], dict] = {}
    occupied: set[tuple[int, int]] = set()
    for cell in sorted(cells, key=lambda c: geoms[id(c)][:2]):
        r, c, rs, cs = geoms[id(cell)]
        if (r, c) in occupied:
            continue  # overlap: first claim wins, validate_cells flags it
        anchors[(r, c)] = cell
        occupied.update((r + dr, c + dc) for dr in range(rs) for dc in range(cs))

    parts = ["<table>"]
    for r in range(n_rows):
        parts.append("<tr>")
        c = 0
        while c < n_cols:
            cell = anchors.get((r, c))
            if cell is not None:
                _, _, rs, cs = geoms[id(cell)]
                tag = "th" if cell.get("h") else "td"
                attrs = ""
                if rs > 1:
                    attrs += f' rowspan="{rs}"'
                if cs > 1:
                    attrs += f' colspan="{cs}"'
                bbox = cell.get("b")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        attrs += ' data-bbox="{},{},{},{}"'.format(
                            *(int(round(float(v))) for v in bbox)
                        )
                    except (TypeError, ValueError):
                        pass
                if multi_page and isinstance(cell.get("p"), int):
                    attrs += f" data-page={quoteattr(str(cell['p']))}"
                parts.append(f"<{tag}{attrs}>{escape(_norm(str(cell.get('t', ''))))}</{tag}>")
                c += cs
            elif (r, c) in occupied:
                c += 1  # covered by a span from above/left
            else:
                parts.append("<td></td>")  # a real blank, kept blank
                c += 1
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)
