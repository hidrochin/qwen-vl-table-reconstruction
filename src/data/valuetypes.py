"""Layout-independent value typing -- the one lexical primitive shared by the
survey, the verifier, and schema discovery.

The reconstruction design leans on "value semantics" in three places: the survey
types each alignment track (``ocr/layout.py``), the verifier flags a value that
does not fit its column's induced kind (``eval/verify.py``), and schema discovery
scores a candidate by how well every fragment fits the column it would land in
(``model/schema_infer.py``). All three must agree on *what kind a string is*, so
that logic lives here once.

Every rule is a **general** property of printed values -- a number looks like a
number, a percentage carries ``%``, a one-glyph token is a code -- never a
layout fact. There is deliberately no "Type means D/C" or "Figure100 has three
columns" knowledge here; that would violate the design's generalization
constraint. ``kind_accepts`` is permissive for the ``text``/``other`` kinds and
discriminative for the number/symbol kinds, because the failure the scorer must
catch is a number landing in a code column (or vice-versa), not the difference
between two number sub-kinds.

Pure stdlib; imports on the Mac with nothing installed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# The column-kind vocabulary, single source of truth (``cells.py`` re-exports it
# for the guided-JSON enum). Keep it in sync with ``COLUMN_KINDS`` there -- they
# are the same object by import, so there is nothing to drift.
COLUMN_KINDS = ("text", "numeric", "symbol", "amount", "percent", "date", "other")

# The three "looks like a number" kinds. Any of them satisfies a number-typed
# column; the scorer does not try to separate amount-vs-numeric (that distinction
# is stylistic and not what discriminates a real schema error).
NUMBER_KINDS = frozenset({"numeric", "amount", "percent"})

_CURRENCY = "$€£¥₩₹฿"
_CURRENCY_RE = re.compile(f"[{re.escape(_CURRENCY)}]")
_PERCENT_RE = re.compile(r"^[-+(]?\s*\d[\d,]*\.?\d*\s*%\)?$")
_GROUPED_RE = re.compile(r"^\(?[-+]?\s*\d{1,3}(,\d{3})+(\.\d+)?\)?$")  # 1,000 / (2,500.00)
_PAREN_NUM_RE = re.compile(r"^\(\s*[-+]?\d[\d,]*\.?\d*\s*\)$")  # accounting negative
_NUMERIC_RE = re.compile(r"^[-+]?\d+(\.\d+)?$")
_DATE_NUM_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE_WORD_RE = re.compile(rf"^\d{{1,2}}[-\s]({_MONTHS})[a-z]*([-\s,]*\d{{2,4}})?$", re.I)


def _norm(text: str) -> str:
    return " ".join(text.split())


def classify_kind(text: str) -> str:
    """Best single kind for one printed value, most specific first.

    Order matters: ``%`` wins over plain-number, currency/grouping/parentheses
    mark an amount, a bare run of digits is numeric, a one- or two-character
    non-numeric token is a symbol/code, and anything else is prose ``text``. An
    empty string is ``other`` -- blanks have no kind and callers filter them.
    """
    s = _norm(text)
    if not s:
        return "other"
    if "%" in s and _PERCENT_RE.match(s):
        return "percent"
    if _DATE_NUM_RE.match(s) or _DATE_WORD_RE.match(s):
        return "date"
    if _CURRENCY_RE.search(s) or _GROUPED_RE.match(s) or _PAREN_NUM_RE.match(s):
        return "amount"
    if _NUMERIC_RE.match(s):
        return "numeric"
    # A lone glyph, or a very short all-non-digit token (D, C, Dr, x, +, -), is a
    # code -- the "symbol" columns in these tables. "Nil", "N/A" (len 3+) stay
    # text, which is what a permissive Optional column accepts.
    if len(s) == 1 or (len(s) <= 2 and not any(ch.isdigit() for ch in s)):
        return "symbol"
    return "text"


def induce_kind(values: Iterable[str]) -> str:
    """The dominant kind of a column, from its own body values (blanks ignored).

    This is the *induced signature* the design conditions schema choice on: no
    track is required to be anything, its kind is discovered from what it holds.
    Plurality vote over ``classify_kind``; empty column -> ``other``.
    """
    kinds = [classify_kind(v) for v in values if _norm(v)]
    if not kinds:
        return "other"
    return Counter(kinds).most_common(1)[0][0]


def parse_number(text: str) -> float | None:
    """Numeric value of a printed cell, or None if it is not a number.

    Handles the accounting forms the arithmetic checks need: currency marks,
    thousands separators, a trailing ``%`` (as its face value, not /100 -- the
    check compares like with like), and parentheses as a negative. Returns None
    for prose so callers can skip it. Layout-independent.
    """
    s = _norm(text)
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    s = _CURRENCY_RE.sub("", s).replace(",", "").replace("%", "").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def kind_accepts(kind: str, text: str) -> bool:
    """Could a column of ``kind`` legitimately hold this value?

    Permissive for ``text``/``other`` (a description column holds anything);
    discriminative for the rest so the scorer/verifier catch the placement
    errors that matter -- a number in a code column, a code in a number column.
    A blank is accepted by every kind (blanks are content, not type violations).
    """
    if not _norm(text):
        return True
    if kind in ("text", "other"):
        return True
    observed = classify_kind(text)
    if kind in NUMBER_KINDS:
        return observed in NUMBER_KINDS
    if kind == "symbol":
        return observed == "symbol"
    if kind == "date":
        return observed in ("date", "numeric")
    return True
