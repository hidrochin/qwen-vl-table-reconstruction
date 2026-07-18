"""Shared HTML/table primitives.

Model output is frequently malformed, so every parser here goes through
``lxml.html``, which recovers from unclosed tags rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import lxml.html

CELL_TAGS = ("td", "th")


@dataclass(frozen=True)
class Cell:
    """A table cell with its declared spans and 0-indexed grid anchor."""

    row: int
    col: int
    rowspan: int
    colspan: int
    tag: str
    text: str

    @property
    def is_spanning(self) -> bool:
        return self.rowspan > 1 or self.colspan > 1


def parse_table(html: str) -> lxml.html.HtmlElement | None:
    """Return the first <table> element, or None if nothing parseable is found."""
    if not html or not html.strip():
        return None
    try:
        fragment = lxml.html.fromstring(html)
    except Exception:
        return None
    if fragment.tag == "table":
        return fragment
    found = fragment.find(".//table")
    if found is not None:
        return found
    # Some outputs omit <table> and start straight at <tr>. Wrap and retry.
    if fragment.tag in ("tr", "tbody", "thead") or fragment.find(".//tr") is not None:
        try:
            return lxml.html.fromstring(f"<table>{html}</table>")
        except Exception:
            return None
    return None


def _span(el: lxml.html.HtmlElement, attr: str) -> int:
    """Read a span attribute, tolerating junk like ``colspan="two"`` or ``"0"``."""
    try:
        return max(1, int(el.get(attr, 1)))
    except (TypeError, ValueError):
        return 1


def extract_cells(html: str) -> list[Cell]:
    """Lay cells onto a grid using the HTML table model.

    Walks rows left to right, skipping columns already claimed by a rowspan from
    above — the same occupancy rule browsers use. Without this, any table
    containing a rowspan reports the wrong column count.
    """
    table = parse_table(html)
    if table is None:
        return []

    occupied: set[tuple[int, int]] = set()
    cells: list[Cell] = []

    for row_idx, tr in enumerate(table.iter("tr")):
        col = 0
        for el in tr:
            if el.tag not in CELL_TAGS:
                continue
            while (row_idx, col) in occupied:
                col += 1
            rowspan, colspan = _span(el, "rowspan"), _span(el, "colspan")
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied.add((row_idx + dr, col + dc))
            cells.append(
                Cell(
                    row=row_idx,
                    col=col,
                    rowspan=rowspan,
                    colspan=colspan,
                    tag=el.tag,
                    text=" ".join(el.text_content().split()),
                )
            )
            col += colspan

    return cells


def to_structure_only(html: str) -> str:
    """Strip cell text, keeping tags and span attributes.

    This is both the training target and the TEDS-Struct input: it cuts sequence
    length roughly 5-10x and removes OCR accuracy from a structure measurement.
    """
    table = parse_table(html)
    if table is None:
        return ""

    parts = ["<table>"]
    for tr in table.iter("tr"):
        parts.append("<tr>")
        for el in tr:
            if el.tag not in CELL_TAGS:
                continue
            attrs = "".join(
                f' {a}="{_span(el, a)}"' for a in ("rowspan", "colspan") if _span(el, a) > 1
            )
            parts.append(f"<{el.tag}{attrs}></{el.tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def from_token_list(tokens: list[str], cell_texts: list[str] | None = None) -> str:
    """Rebuild HTML from a PubTabNet/FinTabNet ``structure`` token list.

    Both datasets ship structure as tokens like ``['<td', ' rowspan="2"', '>',
    '</td>']`` rather than as an HTML string. Cell texts, when given, fill
    non-empty cells in document order.
    """
    html: list[str] = []
    text_iter = iter(cell_texts or [])
    for token in tokens:
        html.append(token)
        # A cell opener is closed by '>' — that is where content belongs.
        if token == ">" and html and any(t.startswith("<td") or t.startswith("<th") for t in html[-3:]):
            html.append(next(text_iter, ""))
    body = "".join(html)
    return body if body.lstrip().startswith("<table") else f"<table>{body}</table>"
