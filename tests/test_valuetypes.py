"""Tests for the shared value-type classifier (src/data/valuetypes.py).

These kinds are the lexical primitive schema discovery and verification both
rely on, so the discriminative cases from layout_description.md are pinned here:
numbers vs codes vs prose, percentages, accounting amounts, and the parse used
by the arithmetic check.
"""

from src.data.valuetypes import (
    COLUMN_KINDS,
    NUMBER_KINDS,
    classify_kind,
    induce_kind,
    kind_accepts,
    parse_number,
)


class TestClassifyKind:
    def test_numbers_and_percentages(self):
        assert classify_kind("123") == "numeric"
        assert classify_kind("45.6") == "numeric"
        assert classify_kind("3%") == "percent"
        assert classify_kind("-12%") == "percent"

    def test_accounting_amounts(self):
        assert classify_kind("1,000") == "amount"
        assert classify_kind("1,234.56") == "amount"
        assert classify_kind("$500") == "amount"
        assert classify_kind("(2,500.00)") == "amount"  # parenthesised negative

    def test_symbols_are_short_codes(self):
        assert classify_kind("D") == "symbol"
        assert classify_kind("C") == "symbol"
        assert classify_kind("+") == "symbol"

    def test_prose_is_text(self):
        assert classify_kind("Return") == "text"
        assert classify_kind("Nil") == "text"  # 3 letters -> text, not a code
        assert classify_kind("Total Amount") == "text"

    def test_dates(self):
        assert classify_kind("2024-01-31") == "date"
        assert classify_kind("31/01/2024") == "date"

    def test_blank_is_other(self):
        assert classify_kind("") == "other"
        assert classify_kind("   ") == "other"


class TestKindAccepts:
    def test_number_columns_reject_codes(self):
        assert not kind_accepts("numeric", "D")
        assert not kind_accepts("amount", "Return")

    def test_number_columns_accept_number_forms(self):
        assert kind_accepts("numeric", "3%")  # a percentage is number-family
        assert kind_accepts("amount", "1,000")

    def test_symbol_column_rejects_numbers_and_prose(self):
        assert not kind_accepts("symbol", "123")
        assert not kind_accepts("symbol", "Description")
        assert kind_accepts("symbol", "D")

    def test_text_column_is_permissive(self):
        for v in ("D", "123", "Nil", "3%", "anything"):
            assert kind_accepts("text", v)

    def test_blank_accepted_by_every_kind(self):
        for kind in COLUMN_KINDS:
            assert kind_accepts(kind, "")


class TestInduceKind:
    def test_plurality_over_body(self):
        assert induce_kind(["123", "234", "345"]) == "numeric"
        assert induce_kind(["D", "C", "D"]) == "symbol"

    def test_mixed_number_column_stays_number(self):
        # a Value column of "3%" and "1000" resolves to a number kind, and that
        # kind still accepts both members.
        kind = induce_kind(["3%", "1000", "2500"])
        assert kind in NUMBER_KINDS
        assert kind_accepts(kind, "3%") and kind_accepts(kind, "1000")

    def test_blanks_ignored(self):
        assert induce_kind(["", "  ", "123"]) == "numeric"
        assert induce_kind(["", ""]) == "other"


class TestParseNumber:
    def test_forms(self):
        assert parse_number("1,234.56") == 1234.56
        assert parse_number("$500") == 500.0
        assert parse_number("(1,200.50)") == -1200.5
        assert parse_number("3%") == 3.0

    def test_prose_is_none(self):
        assert parse_number("Return") is None
        assert parse_number("") is None
