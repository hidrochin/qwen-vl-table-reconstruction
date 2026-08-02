"""Tests for the layout-independent verifier (src/eval/verify.py).

The verifier splits findings into repairable ``problems`` (fed to the cells
repair loop) and soft ``flags`` (recall/human). Each family is checked in
isolation on synthetic payloads shaped like the hard cases: a value in the wrong
column, a horizontally-shifted cell, a near-miss total, a declared-but-empty
column, and -- with a tiny in-memory image -- ink sitting in a "blank" cell.
"""

import pytest

from src.eval.verify import (
    VerificationReport,
    check_arithmetic,
    check_geometry,
    check_schema_echo,
    check_type_coherence,
    verify,
)


def cell(r, c, t, b, rs=1, cs=1, h=False, p=1):
    return {"r": r, "c": c, "rs": rs, "cs": cs, "h": h, "t": t, "b": b, "p": p}


# A clean 3-column table: Description(text) | Number(numeric) | Type(symbol).
def clean_payload():
    return {
        "columns": [
            {"name": "Description", "kind": "text"},
            {"name": "Number", "kind": "numeric"},
            {"name": "Type", "kind": "symbol"},
        ],
        "cells": [
            cell(0, 0, "Description", [0, 0, 40, 10], h=True),
            cell(0, 1, "Number", [50, 0, 90, 10], h=True),
            cell(0, 2, "Type", [100, 0, 140, 10], h=True),
            cell(1, 0, "Return", [0, 20, 40, 30]),
            cell(1, 1, "123", [50, 20, 90, 30]),
            cell(1, 2, "D", [100, 20, 140, 30]),
            cell(2, 0, "Claim", [0, 40, 40, 50]),
            cell(2, 1, "234", [50, 40, 90, 50]),
            cell(2, 2, "C", [100, 40, 140, 50]),
            cell(3, 1, "345", [50, 60, 90, 70]),
            cell(3, 2, "D", [100, 60, 140, 70]),
        ],
    }


class TestClean:
    def test_no_problems_on_a_consistent_table(self):
        rep = verify(clean_payload(), check_ink=False)
        assert rep.problems == []
        assert rep.ok


class TestTypeCoherence:
    def test_number_in_symbol_column_flagged(self):
        p = clean_payload()
        # put a number where the Type (symbol) column lives
        p["cells"][5]["t"] = "999"  # was "D" at (1,2)
        problems = check_type_coherence(p)
        assert any("999" in x and "wrong column" in x for x in problems)

    def test_text_column_never_flags(self):
        p = clean_payload()
        p["cells"][3]["t"] = "D"  # a code in the free-text Description column is fine
        assert check_type_coherence(p) == []


class TestGeometry:
    def test_shifted_cell_detected_by_its_box(self):
        p = clean_payload()
        # a cell logically in column 1 but whose box sits in column 2's x-range
        p["cells"][7]["b"] = [100, 40, 140, 50]  # "234" box moved under Type
        problems = check_geometry(p)
        assert any("shifted" in x for x in problems)

    def test_column_order_disagreement(self):
        p = clean_payload()
        # swap column 1 and 2 x-ranges so declared order disagrees with boxes
        for cc in p["cells"]:
            if cc["c"] == 1 and not cc["h"]:
                cc["b"] = [100, cc["b"][1], 140, cc["b"][3]]
            if cc["c"] == 2 and not cc["h"]:
                cc["b"] = [50, cc["b"][1], 90, cc["b"][3]]
        assert any("order disagrees" in x for x in check_geometry(p))


class TestSchemaEcho:
    def test_declared_but_unused_column_flagged(self):
        p = clean_payload()
        p["columns"].append({"name": "Optional", "kind": "text"})  # nothing placed in col 3
        assert any("declared but no cell" in x for x in check_schema_echo(p))


class TestArithmetic:
    def test_exact_total_is_silent(self):
        p = {
            "columns": [{"name": "n", "kind": "numeric"}],
            "cells": [cell(i, 0, str(v), [0, i * 10, 9, i * 10 + 9]) for i, v in enumerate([10, 20, 30, 60])],
        }
        assert check_arithmetic(p) == []

    def test_near_miss_total_flagged(self):
        p = {
            "columns": [{"name": "n", "kind": "numeric"}],
            "cells": [cell(i, 0, str(v), [0, i * 10, 9, i * 10 + 9]) for i, v in enumerate([10, 20, 30, 58])],
        }
        assert any("looks like a total" in x for x in check_arithmetic(p))

    def test_plain_number_column_not_flagged(self):
        p = {
            "columns": [{"name": "n", "kind": "numeric"}],
            "cells": [cell(i, 0, str(v), [0, i * 10, 9, i * 10 + 9]) for i, v in enumerate([123, 234, 345])],
        }
        assert check_arithmetic(p) == []


class TestOcrMissedAndFragments:
    def test_ocr_missed_becomes_a_flag_not_a_problem(self):
        p = clean_payload()
        p["ocr_missed"] = [{"t": "faint", "b": [0, 80, 20, 90], "p": 1}]
        rep = verify(p, check_ink=False)
        assert any("ocr_missed" in f for f in rep.flags)
        assert not any("ocr_missed" in x for x in rep.problems)


class TestInkInBlank:
    def test_ink_in_a_blank_cell_is_flagged(self):
        Image = pytest.importorskip("PIL.Image")
        # white page; paint ink into the region of the (1,1) cell, then leave that
        # grid position blank in the payload -> the check should notice the ink.
        img = Image.new("L", (160, 80), color=255)
        for x in range(50, 90):
            for y in range(20, 30):
                img.putpixel((x, y), 0)
        p = clean_payload()
        p["cells"] = [c for c in p["cells"] if not (c["r"] == 1 and c["c"] == 1)]  # blank the (1,1)
        flags = verify(p, images=img).flags
        assert any("ink found in blank cell r=1 c=1" in f for f in flags)

    def test_no_image_skips_silently(self):
        rep = verify(clean_payload(), images=None)
        assert not any("ink" in f for f in rep.flags)


class TestVerifyToggles:
    def test_families_can_be_disabled(self):
        p = clean_payload()
        p["cells"][5]["t"] = "999"  # a type violation
        assert not verify(p, check_type=False, check_ink=False).problems
        assert verify(p, check_type=True, check_ink=False).problems

    def test_repair_suffix_only_when_problems(self):
        rep = VerificationReport(problems=["p1"], flags=["f1"])
        assert "p1" in rep.repair_suffix('{"cells": []}')
        assert VerificationReport().repair_suffix("{}") is None
