"""Draw per-cell bounding boxes from a model's ``data-bbox`` output.

The schema / grounded prompts, run with ``build_schema_instruction(with_bbox=True)``
(or ``append_bbox_rule(...)``), make the model emit ``data-bbox="x1,y1,x2,y2"`` on
every non-empty cell. ``html_utils.extract_cells`` parses those into ``Cell.bbox``;
this module paints them back onto the cropped table image so a reconstruction can
be checked by eye -- header cells and body cells in different colours, so the
inferred header is obvious.

PIL is imported inside the drawing function, so the module stays import-safe on
the Mac (the repo's deferred-import invariant).
"""

from __future__ import annotations

from pathlib import Path

from src.data.html_utils import Cell, extract_cells

# Distinct colours for the two cell roles (R, G, B). Header vs body is the split
# that matters most when eyeballing a schema-inference result.
HEADER_COLOR = (59, 130, 246)   # blue
CELL_COLOR = (220, 38, 38)      # red


def boxed_cells(html: str) -> list[Cell]:
    """The cells that carry a parsed ``data-bbox`` (i.e. are drawable).

    Blank cells and any prediction produced without ``with_bbox`` have no box, so
    this is usually the non-empty cells of a ``with_bbox`` run.
    """
    return [c for c in extract_cells(html) if c.bbox is not None]


def draw_cell_boxes(
    image,
    html: str,
    *,
    header_color: tuple[int, int, int] = HEADER_COLOR,
    cell_color: tuple[int, int, int] = CELL_COLOR,
    width: int = 2,
    show_text: bool = False,
):
    """Overlay one rectangle per logical cell onto a copy of ``image``.

    ``image`` is a PIL image or a path. Returns a new RGB PIL image; the input is
    left untouched. ``<th>`` cells use ``header_color`` and body cells
    ``cell_color``. With ``show_text`` the cell text is drawn at the box corner --
    handy for spotting a value placed in the wrong cell.

    Boxes are clamped to the image so a cell at a clipped crop edge (the box the
    prompt was told to run to the border) still draws inside the canvas.
    """
    from PIL import Image, ImageDraw

    im = Image.open(image) if isinstance(image, (str, Path)) else image
    canvas = im.convert("RGB").copy()
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)

    for c in boxed_cells(html):
        x1, y1, x2, y2 = c.bbox
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        color = header_color if c.tag == "th" else cell_color
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if show_text and c.text:
            draw.text((x1 + 2, y1 + 2), c.text, fill=color)
    return canvas
