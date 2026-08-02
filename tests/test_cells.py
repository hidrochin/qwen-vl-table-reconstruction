"""Tests for the JSON cell-list output path (src/model/cells.py).

The cells path is the structural answer to two observed failures: bboxes being
dropped from later calls, and neighbouring values shifted into blank cells. The
tests pin the guarantees: the guided schema requires b/p on every cell, blanks
are grid holes, the validator catches the failure modes, and the HTML conversion
round-trips through the existing ``extract_cells`` grid semantics.
"""

import json
from dataclasses import dataclass

from src.data.html_utils import extract_cells
from src.model.cells import (
    CELLS_INSTRUCTION,
    build_cells_instruction,
    build_repair_suffix,
    cells_json_schema,
    cells_to_html,
    collect_ocr_missed,
    fill_cells_from_fragments,
    parse_cells,
    validate_cells,
    vote_cells,
)
from src.ocr.read import Fragment

# A two-level header over a 3-column body, one blank at (2,1), one merged header.
PAYLOAD = {
    "columns": [
        {"name": "Description", "kind": "text"},
        {"name": "Number", "kind": "numeric"},
        {"name": "Type", "kind": "symbol"},
    ],
    "cells": [
        {"r": 0, "c": 0, "rs": 2, "cs": 1, "h": True, "t": "Description", "b": [0, 0, 40, 30], "p": 1},
        {"r": 0, "c": 1, "rs": 1, "cs": 2, "h": True, "t": "Figure", "b": [45, 0, 120, 14], "p": 1},
        {"r": 1, "c": 1, "rs": 1, "cs": 1, "h": True, "t": "Number", "b": [45, 16, 80, 30], "p": 1},
        {"r": 1, "c": 2, "rs": 1, "cs": 1, "h": True, "t": "Type", "b": [85, 16, 120, 30], "p": 1},
        {"r": 2, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "Fee", "b": [0, 32, 40, 46], "p": 1},
        # (2,1) intentionally absent -- a real blank
        {"r": 2, "c": 2, "rs": 1, "cs": 1, "h": False, "t": "D", "b": [85, 32, 120, 46], "p": 1},
    ],
}


@dataclass
class Word:
    text: str


class TestInstruction:
    def test_default_is_document_agnostic(self):
        for hardcoded in ("Currency", "Figure100", "Debit", "Credit"):
            assert hardcoded not in CELLS_INSTRUCTION

    def test_shares_the_schema_principles(self):
        low = build_cells_instruction().lower()
        assert "blank" in low
        assert "variable" in low
        assert "crop" in low
        assert "qr" in low

    def test_asks_for_json_not_html(self):
        assert '"cells"' in CELLS_INSTRUCTION
        assert '"columns"' in CELLS_INSTRUCTION
        assert "<th>" not in CELLS_INSTRUCTION
        assert "columns FIRST" in CELLS_INSTRUCTION  # schema-before-cells ordering

    def test_hint_is_framed_as_a_hint(self):
        with_hint = build_cells_instruction("Header: A -> B, C")
        assert "Header: A -> B, C" in with_hint
        assert "hint" in with_hint.lower()

    def test_boxes_required_every_response(self):
        assert "every response" in CELLS_INSTRUCTION.lower()


class TestGuidedSchema:
    def test_bbox_and_page_are_required_on_every_cell(self):
        """The structural fix for the disappearing-bbox bug."""
        cell_schema = cells_json_schema()["properties"]["cells"]["items"]
        assert "b" in cell_schema["required"]
        assert "p" in cell_schema["required"]

    def test_blank_cells_cannot_be_emitted(self):
        cell_schema = cells_json_schema()["properties"]["cells"]["items"]
        assert cell_schema["properties"]["t"]["minLength"] == 1

    def test_example_payload_satisfies_required_fields(self):
        cell_schema = cells_json_schema()["properties"]["cells"]["items"]
        for cell in PAYLOAD["cells"]:
            assert set(cell_schema["required"]) <= set(cell)


class TestParseCells:
    def test_clean_json(self):
        assert parse_cells(json.dumps(PAYLOAD)) == PAYLOAD

    def test_fenced_json(self):
        raw = "```json\n" + json.dumps(PAYLOAD) + "\n```"
        assert parse_cells(raw) == PAYLOAD

    def test_prose_and_thinking_stripped(self):
        raw = "<think>the table has {braces} in it</think>Here you go:\n" + json.dumps(PAYLOAD)
        assert parse_cells(raw) == PAYLOAD

    def test_garbage_returns_none(self):
        assert parse_cells("no json here") is None
        assert parse_cells("") is None
        assert parse_cells("[1, 2, 3]") is None


class TestValidateCells:
    def test_clean_payload_passes(self):
        assert validate_cells(PAYLOAD) == []

    def test_overlap_is_caught(self):
        bad = json.loads(json.dumps(PAYLOAD))
        bad["cells"].append(
            {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "X", "b": [0, 0, 5, 5], "p": 1}
        )
        assert any("overlaps" in p for p in validate_cells(bad))

    def test_cell_past_declared_columns_is_caught(self):
        bad = json.loads(json.dumps(PAYLOAD))
        bad["cells"].append(
            {"r": 3, "c": 2, "rs": 1, "cs": 2, "h": False, "t": "Y", "b": [0, 50, 5, 60], "p": 1}
        )
        assert any("columns are declared" in p for p in validate_cells(bad))

    def test_degenerate_bbox_and_bad_page(self):
        bad = json.loads(json.dumps(PAYLOAD))
        bad["cells"][0]["b"] = [40, 30, 0, 0]
        bad["cells"][1]["p"] = 2
        problems = validate_cells(bad, n_pages=1)
        assert any("degenerate bbox" in p for p in problems)
        assert any("outside 1..1" in p for p in problems)

    def test_dropped_ocr_fragment_is_caught(self):
        words = [Word("Fee"), Word("D"), Word("Ghost")]
        assert any("dropped" in p for p in validate_cells(PAYLOAD, words=words))

    def test_invented_value_is_caught(self):
        """A token in a cell that no OCR fragment supplied -- the invention
        failure the prompt forbids, now machine-detected."""
        words = [
            Word(c["t"]) for c in PAYLOAD["cells"][:-1]
        ]  # everything except "D"
        assert any("invented" in p and "'D'" in p for p in validate_cells(PAYLOAD, words=words))

    def test_empty_text_flagged(self):
        bad = json.loads(json.dumps(PAYLOAD))
        bad["cells"][4]["t"] = "  "
        assert any("blank cells must be omitted" in p for p in validate_cells(bad))

    def test_none_payload(self):
        assert validate_cells(None) != []

    def test_repair_suffix_carries_previous_answer_and_problems(self):
        suffix = build_repair_suffix('{"cells": []}', ["problem one", "problem two"])
        assert '{"cells": []}' in suffix
        assert "- problem one" in suffix
        assert "fix ONLY these problems" in suffix


class TestCellsToHtml:
    def test_round_trips_through_extract_cells(self):
        cells = extract_cells(cells_to_html(PAYLOAD))
        by_pos = {(c.row, c.col): c for c in cells}
        desc = by_pos[(0, 0)]
        assert desc.tag == "th" and desc.rowspan == 2 and desc.text == "Description"
        fig = by_pos[(0, 1)]
        assert fig.colspan == 2 and fig.bbox == (45, 0, 120, 14)

    def test_blank_stays_blank(self):
        cells = extract_cells(cells_to_html(PAYLOAD))
        blank = next(c for c in cells if (c.row, c.col) == (2, 1))
        assert blank.text == "" and blank.bbox is None

    def test_data_page_only_when_multi_page(self):
        single = cells_to_html(PAYLOAD)
        assert "data-page" not in single
        multi = json.loads(json.dumps(PAYLOAD))
        multi["cells"][-1]["p"] = 2
        assert 'data-page="2"' in cells_to_html(multi)

    def test_overlapping_anchor_skipped_not_fatal(self):
        bad = json.loads(json.dumps(PAYLOAD))
        bad["cells"].append(
            {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "X", "b": [0, 0, 5, 5], "p": 1}
        )
        cells = extract_cells(cells_to_html(bad))
        texts = [c.text for c in cells]
        assert "Description" in texts and "X" not in texts

    def test_empty_or_none(self):
        assert cells_to_html(None) == ""
        assert cells_to_html({"columns": [], "cells": []}) == ""

    def test_text_is_escaped(self):
        payload = {
            "columns": [{"name": "A", "kind": "text"}],
            "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "a < b & c", "b": [0, 0, 1, 1], "p": 1}],
        }
        html = cells_to_html(payload)
        assert "a &lt; b &amp; c" in html


# --- fragment-grounded path (the reading-pass contract) -----------------------


def frag(i, text, x, y, w=20, h=10, page=1):
    return Fragment(id=i, text=text, bbox=(x, y, w, h), page=page)


# Three fragments; a grounded payload references them by id and ignores one.
FRAGS = [frag(0, "Return", 0, 20), frag(1, "123", 50, 20), frag(2, "STAMP", 200, 200)]
GROUNDED = {
    "columns": [{"name": "Desc", "kind": "text"}, {"name": "Number", "kind": "numeric"}],
    "cells": [
        {"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "f": [0]},
        {"r": 0, "c": 1, "rs": 1, "cs": 1, "h": False, "f": [1]},
    ],
    "ignored": [2],
}


class TestFragmentGuidedSchema:
    def test_fragments_variant_requires_f_not_t(self):
        item = cells_json_schema(with_fragments=True)["properties"]["cells"]["items"]
        assert "f" in item["required"]
        assert "t" not in item["required"]  # text is looked up, not retyped

    def test_default_variant_unchanged(self):
        item = cells_json_schema()["properties"]["cells"]["items"]
        assert set(item["required"]) == {"r", "c", "rs", "cs", "h", "t", "b", "p"}
        assert "f" not in item["properties"]

    def test_fragments_variant_allows_ignored_and_ocr_missed(self):
        schema = cells_json_schema(with_fragments=True)
        assert "ignored" in schema["properties"]
        assert "ocr_missed" in schema["properties"]

    def test_fragment_instruction_explains_referencing(self):
        ins = build_cells_instruction(with_fragments=True).lower()
        assert "fragment id" in ins
        assert "ignored" in ins and "ocr_missed" in ins


class TestFragmentIdentity:
    def test_clean_partition_passes(self):
        assert validate_cells(GROUNDED, fragments=FRAGS) == []

    def test_unknown_id_is_invented(self):
        bad = json.loads(json.dumps(GROUNDED))
        bad["cells"][0]["f"] = [99]
        assert any("invented" in p for p in validate_cells(bad, fragments=FRAGS))

    def test_dropped_fragment_detected(self):
        bad = json.loads(json.dumps(GROUNDED))
        bad["ignored"] = []  # fragment 2 now neither used nor ignored
        assert any("dropped" in p and "id 2" in p for p in validate_cells(bad, fragments=FRAGS))

    def test_duplicate_reference_detected(self):
        bad = json.loads(json.dumps(GROUNDED))
        bad["cells"][1]["f"] = [0]  # id 0 referenced by two cells
        assert any("duplicated" in p for p in validate_cells(bad, fragments=FRAGS))


class TestFillFromFragments:
    def test_text_box_and_page_filled_from_fragments(self):
        filled = fill_cells_from_fragments(GROUNDED, FRAGS)
        by_pos = {(c["r"], c["c"]): c for c in filled["cells"]}
        assert by_pos[(0, 0)]["t"] == "Return"
        assert by_pos[(0, 1)]["t"] == "123"
        # box is the union (here a single fragment) as [x1,y1,x2,y2]
        assert by_pos[(0, 1)]["b"] == [50, 20, 70, 30]

    def test_multi_fragment_cell_joins_in_reading_order(self):
        frags = [frag(0, "Total", 0, 0), frag(1, "Amount", 25, 0)]
        payload = {"columns": [{"name": "L", "kind": "text"}],
                   "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "f": [1, 0]}]}
        filled = fill_cells_from_fragments(payload, frags)
        assert filled["cells"][0]["t"] == "Total Amount"  # left-to-right, not ref order

    def test_filled_payload_feeds_cells_to_html(self):
        html = cells_to_html(fill_cells_from_fragments(GROUNDED, FRAGS))
        assert "Return" in html and "123" in html

    def test_existing_text_is_preserved(self):
        payload = {"columns": [{"name": "L", "kind": "text"}],
                   "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "kept", "f": [0]}]}
        assert fill_cells_from_fragments(payload, FRAGS)["cells"][0]["t"] == "kept"


class TestOcrMissed:
    def test_collect_ocr_missed(self):
        payload = dict(GROUNDED, ocr_missed=[{"t": "faint", "b": [0, 0, 5, 5], "p": 1}])
        assert collect_ocr_missed(payload)[0]["t"] == "faint"

    def test_none_and_missing(self):
        assert collect_ocr_missed(GROUNDED) == []
        assert collect_ocr_missed(None) == []


class TestVoteCells:
    def test_majority_cell_wins_and_confidence_is_agreement(self):
        def payload(v):
            return {"columns": [{"name": "N", "kind": "numeric"}],
                    "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": v, "b": [0, 0, 9, 9], "p": 1}]}
        merged, conf = vote_cells([payload("123"), payload("123"), payload("128")])
        assert merged["cells"][0]["t"] == "123"
        assert conf[(0, 0)] == 2 / 3

    def test_minority_anchor_dropped(self):
        base = {"columns": [{"name": "N", "kind": "numeric"}],
                "cells": [{"r": 0, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "1", "b": [0, 0, 9, 9], "p": 1}]}
        withextra = json.loads(json.dumps(base))
        withextra["cells"].append({"r": 1, "c": 0, "rs": 1, "cs": 1, "h": False, "t": "x", "b": [0, 10, 9, 19], "p": 1})
        # (1,0) appears in only 1 of 3 drafts -> not a majority -> dropped
        merged, conf = vote_cells([base, base, withextra])
        assert (1, 0) not in conf
        assert len(merged["cells"]) == 1

    def test_single_payload_votes_to_itself(self):
        merged, conf = vote_cells([PAYLOAD])
        assert len(merged["cells"]) == len(PAYLOAD["cells"])
        assert all(v == 1.0 for v in conf.values())

    def test_empty_input(self):
        merged, conf = vote_cells([])
        assert merged["cells"] == [] and conf == {}
