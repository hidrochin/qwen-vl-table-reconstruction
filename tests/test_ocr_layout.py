"""Tests for the deterministic OCR-to-layout algorithm (Option A grounding).

This runs on the Mac with no OCR engine and no GPU: the clustering and
serialization are pure geometry over synthetic ``OcrWord``s. It is the same
discipline as the metric tests -- a silent bug here would feed the VLM a
mis-shaped grid and quietly degrade every grounded prediction, with nothing
crashing to warn you.

Coordinates are ``(x, y, w, h)``, origin top-left. Boxes below are laid out on
a clean grid so the expected rows/columns are unambiguous.
"""

import pytest

from src.data.html_utils import extract_cells
from src.ocr.engine import OcrWord
from src.ocr.layout import (
    build_grid,
    cluster_rows,
    detect_column_separators,
    serialize_layout,
    to_coord_lines,
    to_grid_html,
    to_grid_text,
)


def w(text: str, x: float, y: float, width: float = 20, height: float = 10) -> OcrWord:
    return OcrWord(text=text, bbox=(x, y, width, height), conf=1.0)


# A plain 2x2 grid: columns at x=0 and x=100, rows at y=0 and y=20.
GRID_2X2 = [w("A", 0, 0), w("B", 100, 0), w("C", 0, 20), w("D", 100, 20)]


class TestRowClustering:
    def test_two_rows_split_on_vertical_gap(self):
        rows = cluster_rows(GRID_2X2)
        assert len(rows) == 2
        assert [c.text for c in rows[0]] == ["A", "B"]
        assert [c.text for c in rows[1]] == ["C", "D"]

    def test_row_is_ordered_left_to_right(self):
        # Feed the same row out of order; clustering must sort by x.
        rows = cluster_rows([w("B", 100, 0), w("A", 0, 0)])
        assert [c.text for c in rows[0]] == ["A", "B"]

    def test_slight_vertical_jitter_stays_one_row(self):
        # Photographed pages slant; a 2px wobble on a 10px line is one row.
        rows = cluster_rows([w("A", 0, 0), w("B", 100, 2), w("C", 200, 1)])
        assert len(rows) == 1

    def test_empty_input(self):
        assert cluster_rows([]) == []


class TestColumnDetection:
    def test_finds_single_separator_for_two_columns(self):
        seps = detect_column_separators(GRID_2X2)
        assert len(seps) == 1
        assert 8 < seps[0] < 100  # in the whitespace corridor between the columns

    def test_multiword_cell_stays_one_column(self):
        # "Total Amount" is two words with a small gap; "100" is a real column
        # over. Projection profiling must not split the multi-word cell.
        words = [
            w("Total", 0, 0, width=30),
            w("Amount", 35, 0, width=40),
            w("100", 200, 0, width=20),
            w("Sub", 0, 20, width=30),
            w("5", 200, 20, width=20),
        ]
        seps = detect_column_separators(words)
        assert len(seps) == 1  # one boundary -> two columns, not three
        grid = build_grid(words)
        assert grid[0] == ["Total Amount", "100"]

    def test_empty_input(self):
        assert detect_column_separators([]) == []


class TestBuildGrid:
    def test_rectangular_grid(self):
        assert build_grid(GRID_2X2) == [["A", "B"], ["C", "D"]]

    def test_missing_cell_is_empty_string(self):
        # Second row has no word in the left column.
        words = [w("A", 0, 0), w("B", 100, 0), w("D", 100, 20)]
        grid = build_grid(words)
        assert grid == [["A", "B"], ["", "D"]]


class TestSerialization:
    def test_grid_text_is_pipe_delimited_and_aligned(self):
        assert to_grid_text(GRID_2X2) == "A | B\nC | D"

    def test_grid_html_parses_to_expected_cells(self):
        html = to_grid_html(GRID_2X2)
        assert html.startswith("<table>")
        cells = extract_cells(html)
        assert len(cells) == 4
        assert [c.text for c in cells] == ["A", "B", "C", "D"]

    def test_grid_html_has_no_spans(self):
        # The floor is deliberately span-free; spans are the VLM's job.
        cells = extract_cells(to_grid_html(GRID_2X2))
        assert all(not c.is_spanning for c in cells)

    def test_grid_html_escapes_cell_text(self):
        html = to_grid_html([w("a<b&c", 0, 0)])
        assert "&lt;" in html and "&amp;" in html

    def test_coord_lines_carry_text_and_position(self):
        line = to_coord_lines([w("Hello", 5, 7)]).splitlines()[0]
        assert line.startswith("Hello\t")
        assert "x=5" in line and "y=7" in line

    def test_serialize_dispatch(self):
        assert serialize_layout(GRID_2X2, style="grid") == to_grid_text(GRID_2X2)
        assert serialize_layout(GRID_2X2, style="coords") == to_coord_lines(GRID_2X2)

    def test_serialize_rejects_unknown_style(self):
        with pytest.raises(ValueError):
            serialize_layout(GRID_2X2, style="nonsense")

    def test_empty_input_serializes_to_empty(self):
        assert to_grid_text([]) == ""
        assert to_grid_html([]) == ""
        assert to_coord_lines([]) == ""
