"""Correctness tests for the metric and difficulty scorer.

These run before any model does. A silent bug here would not crash anything --
it would just produce plausible-looking numbers that mean nothing, which is the
worst possible failure mode four days before a demo.
"""

import pytest

from src.data.difficulty import score_table
from src.data.html_utils import extract_cells, to_structure_only
from src.eval.teds import teds_score

SIMPLE = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"

SPANNED = (
    '<table><tr><td rowspan="2">Year</td><td colspan="2">Revenue</td></tr>'
    "<tr><td>USA</td><td>EU</td></tr></table>"
)

NESTED_HEADER = (
    "<table><thead>"
    '<tr><th rowspan="2">Item</th><th colspan="2">2025</th><th colspan="2">2026</th></tr>'
    "<tr><th>Q1</th><th>Q2</th><th>Q1</th><th>Q2</th></tr>"
    "</thead><tbody>"
    "<tr><td>Widgets</td><td>10</td><td>20</td><td>30</td><td>40</td></tr>"
    "</tbody></table>"
)


class TestTEDS:
    def test_identical_scores_one(self):
        assert teds_score(SPANNED, SPANNED) == pytest.approx(1.0)

    def test_unparseable_prediction_scores_zero(self):
        assert teds_score("not html at all", SIMPLE) == 0.0
        assert teds_score("", SIMPLE) == 0.0

    def test_swapped_span_is_penalized(self):
        """rowspan=2 -> colspan=2 must cost something but not everything."""
        corrupted = SPANNED.replace('rowspan="2"', 'colspan="2"', 1)
        score = teds_score(corrupted, SPANNED)
        assert 0.0 < score < 1.0

    def test_dropped_cell_is_penalized(self):
        corrupted = SPANNED.replace("<td>EU</td>", "")
        assert teds_score(corrupted, SPANNED) < 1.0

    def test_more_damage_scores_lower(self):
        """Monotonicity -- the ranking is what drives every decision this week."""
        one_error = SPANNED.replace('rowspan="2"', 'rowspan="3"', 1)
        two_errors = one_error.replace("<td>EU</td>", "")
        assert teds_score(one_error, SPANNED) > teds_score(two_errors, SPANNED)

    def test_structure_only_ignores_text(self):
        """The whole point of TEDS-Struct: OCR errors must not move the score."""
        retexted = SPANNED.replace("USA", "XXX").replace("Revenue", "ZZZZ")
        assert teds_score(retexted, SPANNED, structure_only=True) == pytest.approx(1.0)

    def test_full_teds_does_penalize_text(self):
        retexted = SPANNED.replace("USA", "XXX")
        assert teds_score(retexted, SPANNED, structure_only=False) < 1.0

    def test_missing_ground_truth_raises(self):
        with pytest.raises(ValueError):
            teds_score(SIMPLE, "garbage")


class TestGridLayout:
    def test_rowspan_shifts_following_row(self):
        """A rowspan from row 0 must push row 1's first cell to column 1.

        Get this wrong and every column count in the corpus is wrong.
        """
        cells = extract_cells(SPANNED)
        row1 = sorted([c for c in cells if c.row == 1], key=lambda c: c.col)
        assert [c.text for c in row1] == ["USA", "EU"]
        assert [c.col for c in row1] == [1, 2]

    def test_column_count_accounts_for_spans(self):
        assert score_table(SPANNED).n_cols == 3

    def test_malformed_span_attribute_does_not_crash(self):
        cells = extract_cells('<table><tr><td colspan="two">x</td></tr></table>')
        assert len(cells) == 1 and cells[0].colspan == 1

    def test_bare_rows_without_table_tag(self):
        assert len(extract_cells("<tr><td>a</td><td>b</td></tr>")) == 2


class TestStructureOnly:
    def test_strips_text_keeps_spans(self):
        out = to_structure_only(SPANNED)
        assert "Year" not in out and "USA" not in out
        assert 'rowspan="2"' in out and 'colspan="2"' in out

    def test_is_substantially_shorter(self):
        """Sequence-length reduction is what makes training fit on a small GPU."""
        assert len(to_structure_only(NESTED_HEADER)) < len(NESTED_HEADER)

    def test_roundtrip_is_teds_identical(self):
        """Stripping text must not perturb structure -- else training targets lie."""
        assert teds_score(to_structure_only(SPANNED), SPANNED) == pytest.approx(1.0)


class TestDifficulty:
    def test_spanned_scores_above_simple(self):
        assert score_table(SPANNED).score > score_table(SIMPLE).score

    def test_nested_header_depth_detected(self):
        assert score_table(NESTED_HEADER).header_depth == 2

    def test_header_depth_without_thead(self):
        html = "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
        assert score_table(html).header_depth == 1

    def test_counts_spanning_cells(self):
        assert score_table(SPANNED).n_spanning == 2

    def test_empty_table_is_handled(self):
        assert score_table("").bin == "empty"

    def test_score_stays_in_unit_range(self):
        for html in (SIMPLE, SPANNED, NESTED_HEADER):
            assert 0.0 <= score_table(html).score <= 1.0
