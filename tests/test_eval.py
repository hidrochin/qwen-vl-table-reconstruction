"""Tests for span recovery, bootstrap CIs, and run comparison."""

import pytest

from src.eval.metrics import bootstrap_ci, intervals_overlap, span_recovery
from src.eval.runner import Result, compare_runs, summarize_run

TRUTH = (
    '<table><tr><td rowspan="2">Year</td><td colspan="2">Revenue</td></tr>'
    "<tr><td>USA</td><td>EU</td></tr></table>"
)


class TestSpanRecovery:
    def test_perfect_recovery(self):
        r = span_recovery(TRUTH, TRUTH)
        assert (r.matched, r.total, r.spurious) == (2, 2, 0)
        assert r.recall == 1.0 and r.f1 == 1.0

    def test_missed_span_lowers_recall(self):
        flattened = TRUTH.replace(' rowspan="2"', "")
        r = span_recovery(flattened, TRUTH)
        assert r.matched == 1 and r.total == 2
        assert r.recall == pytest.approx(0.5)

    def test_spurious_span_lowers_precision(self):
        extra = TRUTH.replace("<td>USA</td>", '<td colspan="2">USA</td>')
        r = span_recovery(extra, TRUTH)
        assert r.spurious >= 1
        assert r.precision < 1.0

    def test_right_span_wrong_position_does_not_count(self):
        """A correct span shape in the wrong cell is not a recovered merge."""
        moved = (
            '<table><tr><td>Year</td><td colspan="2">Revenue</td></tr>'
            '<tr><td rowspan="2">USA</td><td>EU</td></tr></table>'
        )
        r = span_recovery(moved, TRUTH)
        assert r.matched < 2

    def test_table_without_spans_is_vacuously_perfect(self):
        plain = "<table><tr><td>a</td><td>b</td></tr></table>"
        assert span_recovery(plain, plain).recall == 1.0

    def test_empty_prediction_recovers_nothing(self):
        assert span_recovery("", TRUTH).matched == 0


class TestBootstrap:
    def test_ci_brackets_the_mean(self):
        mean, lo, hi = bootstrap_ci([0.8, 0.85, 0.9, 0.75, 0.95] * 6)
        assert lo <= mean <= hi

    def test_identical_values_give_zero_width(self):
        mean, lo, hi = bootstrap_ci([0.9] * 20)
        assert mean == pytest.approx(0.9) and lo == pytest.approx(hi)

    def test_is_deterministic_given_seed(self):
        vals = [0.1, 0.5, 0.9, 0.3, 0.7] * 4
        assert bootstrap_ci(vals, seed=42) == bootstrap_ci(vals, seed=42)

    def test_handles_empty_and_singleton(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
        assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)

    def test_overlap_detection(self):
        assert intervals_overlap((0.5, 0.4, 0.6), (0.55, 0.45, 0.65))
        assert not intervals_overlap((0.5, 0.45, 0.55), (0.8, 0.75, 0.85))


def _results(scores, bins=None):
    bins = bins or ["hard"] * len(scores)
    return [
        Result(
            uid=f"t{i}",
            difficulty=b,
            teds_struct=s,
            span_recall=s,
            span_f1=s,
            n_spanning=2,
            pred_html="<table></table>",
            true_html="<table></table>",
            image_path="x.png",
        )
        for i, (s, b) in enumerate(zip(scores, bins))
    ]


class TestRunComparison:
    def test_summary_counts_parse_failures(self):
        rs = _results([0.9, 0.8])
        rs[0].parse_failed = True
        assert summarize_run(rs, "run").parse_failures == 1

    def test_bins_are_reported_separately(self):
        s = summarize_run(_results([0.9, 0.5], ["easy", "hard"]), "run")
        assert s.by_bin["easy"]["n"] == 1 and s.by_bin["hard"]["n"] == 1

    def test_overlapping_intervals_flagged_not_significant(self):
        """The guard that stops a noise-level delta being presented as a win."""
        base = summarize_run(_results([0.70, 0.75, 0.80, 0.85] * 5), "zero-shot")
        cand = summarize_run(_results([0.72, 0.77, 0.82, 0.87] * 5), "fine-tuned")
        assert "NOT SIGNIFICANT" in compare_runs(base, cand)

    def test_clear_separation_reported_as_improvement(self):
        base = summarize_run(_results([0.30, 0.32, 0.31, 0.33] * 5), "zero-shot")
        cand = summarize_run(_results([0.90, 0.92, 0.91, 0.93] * 5), "fine-tuned")
        out = compare_runs(base, cand)
        assert "NOT SIGNIFICANT" not in out and "improves over" in out

    def test_regression_is_named(self):
        base = summarize_run(_results([0.90, 0.92, 0.91, 0.93] * 5), "zero-shot")
        cand = summarize_run(_results([0.30, 0.32, 0.31, 0.33] * 5), "fine-tuned")
        assert "regresses against" in compare_runs(base, cand)
