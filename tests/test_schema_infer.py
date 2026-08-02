"""Tests for schema discovery (src/model/schema_infer.py).

The model call is injected; the deterministic trial-and-select loop is what is
pinned here -- most importantly the spec's signature difficulty: the body, not
the header, decides whether an optional/implicit column exists. A candidate whose
column count matches the body's attested tracks must beat one that drops or adds
a column, and a value in the wrong-kind column must cost.
"""

import json

from src.ocr.engine import OcrWord
from src.ocr.layout import survey
from src.model.schema_infer import (
    CandidateSchema,
    ColumnSpec,
    build_dossier,
    discover_schema,
    parse_candidates,
    schema_candidates_json_schema,
    schema_to_hint,
    score_candidate,
    select_schema,
    survey_candidate,
)


def W(t, x, y, w=None, h=12):
    return OcrWord(t, (x, y, (w if w else max(12, len(t) * 7)), h))


# Figure100 with an implicit Optional column the body attests ("Nil" twice) but
# whose header is never printed. Three body-attested tracks.
def optional_present_survey():
    words = [
        W("Number", 0, 0), W("Type", 80, 0),                 # header (no Optional label)
        W("123", 0, 20), W("D", 80, 20), W("Nil", 150, 20),  # data + optional
        W("234", 0, 40), W("C", 80, 40),                     # optional blank here
        W("345", 0, 60), W("D", 80, 60), W("Nil", 150, 60),  # data + optional
    ]
    return survey(words)


class TestTrialScoring:
    def test_correct_count_beats_dropped_column(self):
        s = optional_present_survey()
        correct = CandidateSchema([ColumnSpec("Number", "numeric"), ColumnSpec("Type", "symbol"), ColumnSpec("Optional", "text")])
        dropped = CandidateSchema([ColumnSpec("Number", "numeric"), ColumnSpec("Type", "symbol")])
        assert score_candidate(correct, s).total < score_candidate(dropped, s).total

    def test_correct_count_beats_extra_column(self):
        s = optional_present_survey()
        correct = CandidateSchema([ColumnSpec("a", "numeric"), ColumnSpec("b", "symbol"), ColumnSpec("c", "text")])
        extra = CandidateSchema([ColumnSpec("a", "numeric"), ColumnSpec("b", "symbol"), ColumnSpec("c", "text"), ColumnSpec("d", "amount")])
        assert score_candidate(correct, s).total < score_candidate(extra, s).total

    def test_wrong_kinds_cost_type_violations(self):
        s = optional_present_survey()
        # declare Number as a symbol column and Type as numeric -> every value misfits
        wrong = CandidateSchema([ColumnSpec("a", "symbol"), ColumnSpec("b", "numeric"), ColumnSpec("c", "text")])
        rep = score_candidate(wrong, s)
        assert rep.column_count_delta == 0
        assert rep.type_violations > 0


class TestSelection:
    def test_selects_the_optional_present_schema(self):
        s = optional_present_survey()
        correct = CandidateSchema([ColumnSpec("Number", "numeric"), ColumnSpec("Type", "symbol"), ColumnSpec("Optional", "text")])
        dropped = CandidateSchema([ColumnSpec("Number", "numeric"), ColumnSpec("Type", "symbol")])
        sel = select_schema([dropped, correct], s)
        assert sel.best.n_cols == 3
        assert not sel.out_of_family

    def test_deferred_commitment_carries_close_candidates(self):
        s = optional_present_survey()
        # two 3-column candidates differing only by one kind: near-tie -> carry both
        a = CandidateSchema([ColumnSpec("n", "numeric"), ColumnSpec("t", "symbol"), ColumnSpec("o", "text")])
        b = CandidateSchema([ColumnSpec("n", "numeric"), ColumnSpec("t", "symbol"), ColumnSpec("o", "other")])
        sel = select_schema([a, b], s)
        assert len(sel.carried) == 2

    def test_out_of_family_when_all_bad(self):
        s = optional_present_survey()
        wild = CandidateSchema([ColumnSpec(f"c{i}", "text") for i in range(9)])  # 9 vs 3 tracks
        sel = select_schema([wild], s)
        assert sel.out_of_family


class TestDossierAndParsing:
    def test_dossier_states_track_count_and_sparsity(self):
        d = build_dossier(optional_present_survey())
        assert "3 vertical alignment track" in d
        assert "sparse" in d

    def test_parse_candidates_reads_columns_and_parents(self):
        raw = json.dumps({"candidates": [
            {"columns": [
                {"name": "Number", "kind": "numeric", "printed": True, "parent": "Figure100"},
                {"name": "Optional", "kind": "text", "printed": False, "parent": "Figure100"},
            ], "justification": "body attests two"}
        ]})
        cands = parse_candidates(raw)
        assert len(cands) == 1
        assert cands[0].columns[1].printed is False
        assert cands[0].columns[0].parent == "Figure100"

    def test_parse_garbage_is_empty(self):
        assert parse_candidates("nope") == []

    def test_guided_schema_enumerates_kinds(self):
        col = schema_candidates_json_schema()["properties"]["candidates"]["items"]["properties"]["columns"]["items"]
        assert "numeric" in col["properties"]["kind"]["enum"]


class TestDiscoverAndHint:
    def test_discover_without_model_uses_survey_baseline(self):
        s = optional_present_survey()
        sel = discover_schema(s, generate_fn=None)
        assert sel.best.n_cols == 3  # matches the three attested tracks

    def test_model_candidate_wins_tie_over_baseline(self):
        s = optional_present_survey()
        fake = lambda ins, sch: json.dumps({"candidates": [{"columns": [
            {"name": "Number", "kind": "numeric", "parent": "Figure100"},
            {"name": "Type", "kind": "symbol", "parent": "Figure100"},
            {"name": "Optional", "kind": "text", "parent": "Figure100"},
        ]}]})
        sel = discover_schema(s, generate_fn=fake)
        assert [c.name for c in sel.best.columns] == ["Number", "Type", "Optional"]

    def test_survey_candidate_matches_attested_tracks(self):
        assert survey_candidate(optional_present_survey()).n_cols == 3

    def test_hint_groups_by_parent_and_marks_implicit(self):
        cand = CandidateSchema([
            ColumnSpec("Description", "text", parent="Currency"),
            ColumnSpec("Optional", "text", printed=False, parent="Figure100"),
        ])
        hint = schema_to_hint(cand)
        assert "Currency ->" in hint and "Figure100 ->" in hint
        assert "(implicit)" in hint
