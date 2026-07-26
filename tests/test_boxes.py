"""Tests for per-cell bounding boxes: parsing ``data-bbox`` and drawing it.

The ``with_bbox`` prompts emit ``data-bbox="x1,y1,x2,y2"`` on non-empty cells so
the caller can draw cell boxes. Parsing must survive the junk models emit, and it
must stay invisible to the structure metrics (a box is not part of the table
tree), so the same output still scores under TEDS-Struct.
"""

import pytest

from src.data.html_utils import extract_cells, to_structure_only
from src.demo.boxes import boxed_cells, draw_cell_boxes

BOXED = (
    '<table>'
    '<tr><th data-bbox="10,20,110,60">Description</th>'
    '<th data-bbox="120,20,200,60">Value</th></tr>'
    '<tr><td data-bbox="10,70,110,110">Return</td>'
    '<td data-bbox="120,70,200,110">3%</td></tr>'
    '<tr><td data-bbox="10,120,110,160">Claim</td><td></td></tr>'  # blank last cell
    '</table>'
)


class TestParseBbox:
    def test_boxes_reach_the_cells(self):
        by_pos = {(c.row, c.col): c.bbox for c in extract_cells(BOXED)}
        assert by_pos[(0, 0)] == (10, 20, 110, 60)
        assert by_pos[(1, 1)] == (120, 70, 200, 110)

    def test_blank_cell_has_no_box(self):
        blank = next(c for c in extract_cells(BOXED) if c.row == 2 and c.col == 1)
        assert blank.text == ""
        assert blank.bbox is None

    def test_missing_attribute_is_none(self):
        cells = extract_cells("<table><tr><td>x</td></tr></table>")
        assert cells[0].bbox is None

    def test_corner_order_normalised(self):
        """Bottom-right written first still yields (x1,y1,x2,y2) top-left first."""
        cells = extract_cells('<table><tr><td data-bbox="200,110,120,70">x</td></tr></table>')
        assert cells[0].bbox == (120, 70, 200, 110)

    def test_float_and_whitespace_tolerated(self):
        cells = extract_cells('<table><tr><td data-bbox=" 10.4, 20.6 , 110, 60 ">x</td></tr></table>')
        assert cells[0].bbox == (10, 21, 110, 60)

    def test_malformed_bbox_is_none_not_error(self):
        for junk in ('data-bbox="1,2,3"', 'data-bbox="a,b,c,d"', 'data-bbox=""'):
            cells = extract_cells(f"<table><tr><td {junk}>x</td></tr></table>")
            assert cells[0].bbox is None

    def test_bbox_invisible_to_structure(self):
        """A box is not structure -- to_structure_only must drop it, so a boxed
        prediction still scores identically under TEDS-Struct."""
        plain = BOXED.replace('data-bbox="10,20,110,60"', "")
        assert to_structure_only(BOXED) == to_structure_only(plain)
        assert "data-bbox" not in to_structure_only(BOXED)


class TestDrawCellBoxes:
    def test_boxed_cells_excludes_blanks(self):
        assert len(boxed_cells(BOXED)) == 5  # 6 cells, one blank

    def test_draws_without_mutating_input(self):
        Image = pytest.importorskip("PIL.Image")
        src = Image.new("RGB", (240, 180), "white")
        out = draw_cell_boxes(src, BOXED)
        assert out.size == src.size
        assert out is not src
        # White canvas gains coloured pixels where boxes were drawn.
        assert out.getcolors(maxcolors=100000) != src.getcolors(maxcolors=100000)

    def test_accepts_a_path(self, tmp_path):
        Image = pytest.importorskip("PIL.Image")
        p = tmp_path / "table.png"
        Image.new("RGB", (240, 180), "white").save(p)
        out = draw_cell_boxes(str(p), BOXED)
        assert out.size == (240, 180)
