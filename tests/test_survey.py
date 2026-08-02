"""Tests for the geometric survey (src/ocr/layout.py additions).

The survey is the measured evidence schema discovery reasons over, so the pins
are the spec's hard cases: alignment/kind induced per track from the body (not
the header), anchors discovered rather than assumed, and row typing that keeps a
sparse data row distinct from a section row and a header row distinct from data.
"""

from src.ocr.engine import OcrWord
from src.ocr.layout import (
    alignment_tracks,
    contested_fragments,
    discover_anchors,
    estimate_pitch,
    survey,
)


def W(t, x, y, w=None, h=12):
    return OcrWord(t, (x, y, (w if w else max(12, len(t) * 7)), h))


# Description | Number | Type, with a section row, a sparse data row, and a total.
def sample_words():
    return [
        W("Description", 0, 0), W("Number", 120, 0), W("Type", 200, 0),  # header
        W("ABC", 0, 20),                                                 # section
        W("Return", 0, 40), W("123", 120, 40), W("D", 200, 40),         # data
        W("Claim", 0, 60), W("234", 120, 60),                           # sparse data (no Type)
        W("Total", 0, 80), W("357", 120, 80),                           # summary
    ]


class TestTracks:
    def test_kinds_induced_from_body_not_header(self):
        s = survey(sample_words())
        kinds = {t.index: t.kind for t in s.tracks}
        assert kinds[0] == "text"  # Description / Return / Claim / Total
        assert kinds[1] == "numeric"  # 123 / 234 / 357 (not the word "Number")
        assert kinds[2] == "symbol"  # D (not the word "Type")

    def test_alignment_is_measured(self):
        # right-aligned numerics: a shared right edge with a varying left edge is
        # the general typographic signal (a column of body values, no header row).
        words = [
            OcrWord(str(v), (130 - len(str(v)) * 7, i * 20, len(str(v)) * 7, 12))
            for i, v in enumerate([5, 250, 4000])
        ]
        tracks = alignment_tracks(words)
        assert tracks[0].align == "right"


class TestPitchAndAnchors:
    def test_pitch_is_median_row_gap(self):
        assert estimate_pitch(sample_words()) == 20.0

    def test_anchors_are_dense_tracks(self):
        s = survey(sample_words())
        assert 0 in s.anchors  # the label column is filled in every row

    def test_anchor_fallback_when_nothing_dense(self):
        # a fully sparse two-column body: no track hits 0.8, so the densest is used
        words = [W("a", 0, 0), W("b", 200, 20), W("c", 0, 40)]
        tracks = alignment_tracks(words)
        assert len(discover_anchors(tracks)) == 1


class TestRowTyping:
    def test_header_section_data_sparse_summary(self):
        kinds = [r.kind for r in survey(sample_words()).rows]
        assert kinds == ["header", "section", "data", "data", "summary"]

    def test_sparse_data_row_is_not_a_section(self):
        # "Claim | 234" fills only two of three columns but carries a value, so it
        # must be data, never a section row (the value-shifting trap).
        rows = survey(sample_words()).rows
        claim = next(r for r in rows if r.cells[0] == "Claim")
        assert claim.kind == "data"


class TestContested:
    def test_clean_table_has_no_contested_fragments(self):
        assert contested_fragments(sample_words()) == []

    def test_fragment_near_a_boundary_is_flagged(self):
        # two clear columns plus a stray fragment left near the surviving corridor
        words = [W("A", 0, 0), W("B", 200, 0), W("A", 0, 20), W("B", 200, 20), W("?", 40, 40, w=10)]
        out = contested_fragments(words, margin_frac=2.0)
        assert any(w.text == "?" for w, _ in out)
