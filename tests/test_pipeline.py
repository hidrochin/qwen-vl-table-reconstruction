"""Tests for the staged pipeline orchestration (src/pipeline.py).

The model calls are injected, so the whole assembly runs on the Mac with fakes
and no PIL: ``run_from_fragments(..., images_for_ink=None)`` exercises survey ->
schema -> grounded cells -> vote -> verify -> monotone repair -> serialize with a
scripted ``cell_fn``. The pins are the orchestration decisions the design turns
on -- k-vote, the strictly-monotone repair accept/reject rule, schema-hint
wiring, and graceful failure on an unparseable draft.
"""

import json

from src.ocr.read import Fragment
from src.pipeline import StagedPipeline, _reindex, _table_confidence


def frag(i, text, x, y, w=20, h=10, page=1):
    return Fragment(id=i, text=text, bbox=(x, y, w, h), page=page)


# A 2-column, 2-row table: a text label column and a percent-value column.
FRAGS = [
    frag(0, "Return", 0, 20), frag(1, "3%", 120, 20),
    frag(2, "Claim", 0, 40), frag(3, "5%", 120, 40),
]


def payload(*used_ids, ignore=()):
    """A grounded cell payload placing the given fragment ids on a 2-col grid."""
    cells = []
    for fid in used_ids:
        r = 0 if fid in (0, 1) else 1
        c = 0 if fid in (0, 2) else 1
        cells.append({"r": r, "c": c, "rs": 1, "cs": 1, "h": False, "f": [fid]})
    obj = {
        "columns": [{"name": "Desc", "kind": "text"}, {"name": "Val", "kind": "percent"}],
        "cells": cells,
    }
    if ignore:
        obj["ignored"] = list(ignore)
    return json.dumps(obj)


OK = payload(0, 1, 2, 3)  # every fragment placed -> clean partition


class ScriptedCell:
    """A fake ``cell_fn`` that returns queued raw strings and records its calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, image, instruction, ocr_layout=None, guided_json=None, temperature=0.0, seed=None):
        self.calls.append({"instruction": instruction, "temperature": temperature, "seed": seed})
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[i]


def run(cell_fn, **cfg):
    pipe = StagedPipeline(cell_fn=cell_fn, **cfg)
    return pipe.run_from_fragments(FRAGS, image="img", n_pages=1, images_for_ink=None)


class TestHappyPath:
    def test_clean_reconstruction(self):
        res = run(ScriptedCell(OK), use_schema_hint=False)
        assert res.ok and res.report.problems == []
        assert len(res.payload["cells"]) == 4
        assert "<table>" in res.html
        assert res.table_confidence > 0

    def test_text_is_filled_from_fragments_not_retyped(self):
        res = run(ScriptedCell(OK), use_schema_hint=False)
        by_pos = {(c["r"], c["c"]): c for c in res.payload["cells"]}
        assert by_pos[(0, 0)]["t"] == "Return"
        assert by_pos[(1, 1)]["t"] == "5%"

    def test_unparseable_draft_returns_empty_not_crash(self):
        res = run(ScriptedCell("not json at all"), use_schema_hint=False)
        assert res.payload["cells"] == []
        assert res.html == ""
        assert res.table_confidence == 0.0
        assert any("no parseable" in n for n in res.notes)


class TestVoting:
    def test_k_drafts_vote_and_yield_confidence(self):
        cell = ScriptedCell(OK, OK, OK)
        res = run(cell, k=3, use_schema_hint=False)
        assert len(cell.calls) == 3
        assert len(res.payload["cells"]) == 4
        assert res.confidence and all(v == 1.0 for v in res.confidence.values())

    def test_draft_temperature_diversity(self):
        cell = ScriptedCell(OK, OK, OK)
        run(cell, k=3, use_schema_hint=False, draft_temperature=0.5)
        temps = [c["temperature"] for c in cell.calls]
        assert temps[0] == 0.0 and temps[1] == 0.5 and temps[2] == 0.5  # draft 0 greedy


class TestMonotoneRepair:
    def test_repair_accepted_when_strictly_fewer_problems(self):
        # draft 1 drops fragment 3 (1 problem); the repair places it (clean).
        cell = ScriptedCell(payload(0, 1, 2), OK)
        res = run(cell, use_schema_hint=False, max_repair_rounds=1)
        assert res.repair_rounds == 1
        assert res.report.problems == []
        assert any("accepted" in n for n in res.notes)

    def test_repair_rejected_keeps_argmin(self):
        # draft 1 has 1 problem (drops 3); the "repair" is worse (drops 2 and 3).
        cell = ScriptedCell(payload(0, 1, 2), payload(0, 1))
        res = run(cell, use_schema_hint=False, max_repair_rounds=1)
        assert res.repair_rounds == 1
        assert len(res.report.problems) == 1  # kept the better first draft
        assert any("rejected" in n for n in res.notes)

    def test_no_repair_when_disabled(self):
        cell = ScriptedCell(payload(0, 1, 2))
        res = run(cell, use_schema_hint=False, max_repair_rounds=0)
        assert res.repair_rounds == 0
        assert len(res.report.problems) == 1  # the dropped fragment stands


class TestSchemaWiring:
    def _generate(self, instruction, guided_json=None):
        return json.dumps({"candidates": [{
            "columns": [
                {"name": "Description", "kind": "text", "printed": True, "parent": None},
                {"name": "Value", "kind": "percent", "printed": True, "parent": None},
            ],
            "justification": "two attested tracks",
        }]})

    def test_schema_hint_reaches_the_cell_prompt(self):
        cell = ScriptedCell(OK)
        pipe = StagedPipeline(cell_fn=cell, generate_fn=self._generate, use_schema_hint=True)
        res = pipe.run_from_fragments(FRAGS, image="img", images_for_ink=None)
        assert res.schema_hint and "Description" in res.schema_hint
        assert "Discovered logical schema" in cell.calls[0]["instruction"]
        assert res.selection is not None and res.selection.carried

    def test_no_hint_when_disabled(self):
        cell = ScriptedCell(OK)
        pipe = StagedPipeline(cell_fn=cell, generate_fn=self._generate, use_schema_hint=False)
        res = pipe.run_from_fragments(FRAGS, image="img", images_for_ink=None)
        assert res.schema_hint is None
        assert "Discovered logical schema" not in cell.calls[0]["instruction"]

    def test_survey_baseline_hint_without_generate_fn(self):
        cell = ScriptedCell(OK)
        res = run(cell, use_schema_hint=True)  # no generate_fn
        assert res.schema_hint is not None  # survey baseline still gives a hint
        assert any("survey baseline" in n for n in res.notes)


class TestHelpers:
    def test_reindex_makes_ids_unique_across_pages(self):
        # two "pages" each numbered 0,1 locally -> globally 0..3
        collided = [frag(0, "a", 0, 0, page=1), frag(1, "b", 0, 0, page=1),
                    frag(0, "c", 0, 0, page=2), frag(1, "d", 0, 0, page=2)]
        out = _reindex(collided)
        assert [f.id for f in out] == [0, 1, 2, 3]
        assert [f.text for f in out] == ["a", "b", "c", "d"]  # order preserved

    def test_from_client_wires_three_callables(self):
        class FakeClient:
            def as_cell_fn(self):
                return lambda *a, **k: OK
            def as_read_fn(self):
                return lambda *a, **k: "{}"
            def as_generate_fn(self):
                return lambda *a, **k: "{}"

        pipe = StagedPipeline.from_client(FakeClient(), k=2)
        assert pipe.k == 2
        assert callable(pipe.cell_fn) and callable(pipe.read_fn) and callable(pipe.generate_fn)

    def test_confidence_is_a_bounded_scalar(self):
        from src.eval.verify import VerificationReport

        c = _table_confidence(FRAGS, None, {}, VerificationReport())
        assert 0.0 <= c <= 1.0
