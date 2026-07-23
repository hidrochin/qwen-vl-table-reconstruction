"""Shared output-parsing for the document specialists (PaddleOCR-VL, MinerU).

Both PaddleOCR-VL and MinerU2.5 are document-parsing *pipelines*, not chat VLMs,
and both return nested, framework-specific result objects (per-page dicts /
objects carrying markdown / structured blocks). The exact shapes are the one open
item flagged for each on the first real run, so the extraction here is written
defensively: walk whatever nested dict/list/attr structure comes back and pull
the first ``<table>...</table>`` out of any text payload. Returns "" when none is
found, which the eval scores as 0 rather than guessing.

Pure-Python and import-safe on the Mac -- no framework import lives here.
"""

from __future__ import annotations

import re

_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)


def extract_table_html(results) -> str:
    """Pull the first ``<table>...</table>`` out of a doc-parser result."""
    for text in _iter_text_fields(results):
        match = _TABLE_RE.search(text)
        if match:
            return match.group(0)
    return ""


def _iter_text_fields(obj):
    """Yield candidate text payloads from nested pipeline output (dicts, objects
    with ``.markdown`` / ``.json`` / ``.html``, lists, plain strings)."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_text_fields(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_text_fields(item)
    else:
        for attr in ("markdown", "md", "html", "res", "json", "content"):
            value = getattr(obj, attr, None)
            if value is not None and value is not obj:
                yield from _iter_text_fields(value)
