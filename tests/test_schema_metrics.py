"""Correctness tests for the logical-reconstruction metrics.

These score the invoice task in ``layout_description.md`` -- right fragment in the
right logical cell, blanks kept blank, right logical-column count. A silent bug
here would flatter the model exactly where the task is hardest (sparse, variable
schemas), so the failure modes are pinned explicitly.
"""

import pytest

from src.eval.metrics import blank_preservation, content_placement, schema_match

# One invoice row: Description, Value, Number, Type, Optional(blank), Debit(blank), Credit.
TRUE = (
    "<table><tr>"
    "<td>Return</td><td>3%</td><td>123</td><td>D</td><td></td><td></td><td>567</td>"
    "</tr></table>"
)

# The classic failure: 567 shifts left into a blank column and two columns vanish.
SHIFTED = "<table><tr><td>Return</td><td>3%</td><td>123</td><td>D</td><td>567</td></tr></table>"


class TestContentPlacement:
    def test_perfect_is_one(self):
        assert content_placement(TRUE, TRUE).accuracy == pytest.approx(1.0)

    def test_shifted_value_counts_as_misplaced(self):
        # 4 of 5 text fragments still land correctly; 567 is now in the wrong cell.
        cp = content_placement(SHIFTED, TRUE)
        assert cp.matched == 4 and cp.total == 5

    def test_no_text_in_truth_is_one(self):
        """Structure-only ground truth has nothing to place -- must not read as 0."""
        assert content_placement("<table><tr><td></td></tr></table>",
                                 "<table><tr><td></td></tr></table>").accuracy == 1.0

    def test_empty_prediction_scores_zero(self):
        assert content_placement("", TRUE).accuracy == 0.0

    def test_case_and_whitespace_normalized(self):
        noisy = TRUE.replace("<td>Return</td>", "<td>  return\n</td>")
        assert content_placement(noisy, TRUE).accuracy == pytest.approx(1.0)


class TestBlankPreservation:
    def test_all_blanks_kept_is_one(self):
        assert blank_preservation(TRUE, TRUE).rate == pytest.approx(1.0)

    def test_filling_a_blank_lowers_rate(self):
        # TRUE has 2 blanks; SHIFTED fills one of those positions with 567.
        bp = blank_preservation(SHIFTED, TRUE)
        assert bp.total == 2 and bp.preserved < 2

    def test_no_blanks_in_truth_is_one(self):
        dense = "<table><tr><td>a</td><td>b</td></tr></table>"
        assert blank_preservation(dense, dense).rate == 1.0


class TestSchemaMatch:
    def test_same_shape_is_correct(self):
        sm = schema_match(TRUE, TRUE)
        assert sm.cols_correct and sm.exact and sm.true_cols == 7

    def test_missing_optional_column_is_wrong(self):
        sm = schema_match(SHIFTED, TRUE)
        assert sm.pred_cols == 5 and sm.true_cols == 7 and not sm.cols_correct

    def test_empty_prediction_has_zero_cols(self):
        assert schema_match("", TRUE).pred_cols == 0
