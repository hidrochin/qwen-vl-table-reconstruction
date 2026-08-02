"""Schema discovery -- the keystone stage, without a template.

The logical column count is not on the page: an optional/implicit column exists
only when the body attests it (``layout_description.md``). ``pipeline_design.md``
§4.1 resolves this without a hard-coded schema by keeping the *mechanism* --
schemas as a few explicit, testable hypotheses -- while changing the *source*
from a config library to Qwen's own abduction under general principles, followed
by a deterministic trial that **tests** each hypothesis instead of trusting it.

This module is that loop, minus the one model call (injected, so the module
imports on the Mac and the scoring is unit-tested without a GPU):

1. ``build_dossier`` -- turn a geometric :class:`~src.ocr.layout.Survey` into an
   evidence description (header labels + their track spans, per-track type
   signatures, sparsity) with no layout assumptions baked in.
2. ``build_schema_discovery_instruction`` + ``schema_candidates_json_schema`` --
   the general-principles prompt and guided-JSON contract Qwen abduces 2-4
   candidate schemas into. Each candidate carries its own justification, so the
   decision is auditable.
3. ``score_candidate`` -- deterministic trial assignment: how many body values
   fail their column's kind, and how far the column count is from the attested
   tracks. Column count is weighted heavily -- it is the highest-blast-radius bit.
4. ``select_schema`` -- pick the argmin; carry the top two when they are close
   (deferred commitment); flag out-of-family when every candidate scores badly
   (novelty detection routes to the holistic fallback + human review).

The winner renders to the ``build_cells_instruction(schema=...)`` hint slot via
``schema_to_hint`` -- the existing, unchanged plumbing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from src.data.valuetypes import COLUMN_KINDS, kind_accepts
from src.ocr.layout import Survey

# Trial-scoring weights. Column count dominates: getting the number of logical
# columns wrong shifts entire column regions, so a count error must cost far more
# than a single misfit value (design: the schema is the highest-blast-radius bit).
_W_COUNT = 100
_W_TYPE = 1
# Above this best-score the document is out-of-family: no abduced schema fits, so
# route to the holistic fallback and human review rather than force a bad schema.
_OUT_OF_FAMILY = 300
# Two candidates within this score are "close" -> carry both (deferred commitment).
_CARRY_MARGIN = _W_COUNT // 2


@dataclass
class ColumnSpec:
    """One logical column of a candidate schema."""

    name: str
    kind: str
    printed: bool = True  # False for an implicit/never-printed column
    parent: str | None = None  # parent header for a two-level schema


@dataclass
class CandidateSchema:
    """A hypothesized logical schema -- a testable object, not a commitment."""

    columns: list[ColumnSpec]
    justification: str = ""

    @property
    def n_cols(self) -> int:
        return len(self.columns)


@dataclass
class ScoreReport:
    """Deterministic trial-assignment score for one candidate (lower is better)."""

    candidate: CandidateSchema
    column_count_delta: int  # signed: declared logical cols - attested body tracks
    type_violations: int  # body values that fail their column's induced kind

    @property
    def total(self) -> int:
        return abs(self.column_count_delta) * _W_COUNT + self.type_violations * _W_TYPE


@dataclass
class Selection:
    """The outcome of testing all candidates."""

    ranked: list[ScoreReport] = field(default_factory=list)
    carried: list[CandidateSchema] = field(default_factory=list)  # 1, or 2 if close
    out_of_family: bool = False

    @property
    def best(self) -> CandidateSchema | None:
        return self.ranked[0].candidate if self.ranked else None


# --- evidence dossier ---------------------------------------------------------


def _attested_tracks(survey: Survey):
    """Body-attested tracks in reading order -- the columns the body actually has."""
    return [t for t in survey.tracks if t.fill > 0]


def _track_body_values(survey: Survey, track_index: int) -> list[str]:
    return [
        r.cells[track_index]
        for r in survey.rows
        if r.kind in ("data", "summary")
        and track_index < len(r.cells)
        and r.cells[track_index].strip()
    ]


def build_dossier(survey: Survey) -> str:
    """A layout-free evidence description of the table for schema abduction.

    States what the header *prints* (labels and the tracks they span), what each
    track *holds* (induced kind, alignment, how often it is filled), and how
    sparse the body is -- the joint header/body evidence the schema must reconcile.
    Deliberately asserts no column meanings or counts; that is Qwen's job.
    """
    tracks = _attested_tracks(survey)
    header_rows = [r for r in survey.rows if r.kind == "header"]

    lines = [f"The body attests {len(tracks)} vertical alignment track(s), left to right:"]
    for pos, t in enumerate(tracks):
        vals = _track_body_values(survey, t.index)
        sample = ", ".join(v[:16] for v in vals[:4]) or "(none)"
        lines.append(
            f"  track {pos}: x=[{t.x1:.0f},{t.x2:.0f}] {t.align}-aligned, "
            f"holds mostly '{t.kind}' values, filled in {t.fill}/{t.n_rows} rows; "
            f"examples: {sample}"
        )

    if header_rows:
        lines.append(f"Printed header rows ({len(header_rows)}, top to bottom):")
        for r in header_rows:
            labels = [f"'{c}' (over track {i})" for i, c in enumerate(r.cells) if c.strip()]
            lines.append("  " + "; ".join(labels))
    else:
        lines.append("No printed header row was detected; child labels may be implicit.")

    body_rows = [r for r in survey.rows if r.kind in ("data", "summary")]
    if body_rows and tracks:
        total_cells = len(body_rows) * len(tracks)
        blank = sum(
            1
            for r in body_rows
            for t in tracks
            if not (t.index < len(r.cells) and r.cells[t.index].strip())
        )
        lines.append(
            f"Body: {len(body_rows)} data/summary rows; {blank}/{total_cells} cells are blank "
            f"({100 * blank / total_cells:.0f}% sparse). Blank cells are real content."
        )
    return "\n".join(lines)


# --- candidate generation contract (model injected) ---------------------------

_SCHEMA_DISCOVERY_PRINCIPLES = (
    "Reconstruction principles (they hold for the whole business-table family, "
    "not one layout):\n"
    "1. Every body-attested track belongs to some logical column; a persistent "
    "track with no printed label is an implicit (unprinted) column, not noise.\n"
    "2. A printed header label may span several tracks (a parent) or one (a "
    "leaf); parents partition the tracks under their horizontal span.\n"
    "3. An optional column exists ONLY when the body attests it: values on a "
    "distinct track whose kind does not fit the neighbouring columns. One stray "
    "value is not attestation.\n"
    "4. Columns never merge because one is sparse; blank cells are content.\n"
    "5. Under the true schema every value lands in a column whose kind accepts "
    "it, across all rows."
)


def build_schema_discovery_instruction(dossier: str) -> str:
    """Prompt for Qwen to abduce 2-4 candidate schemas from the dossier.

    Asks for explicit, testable hypotheses with justifications -- not one answer
    -- because selection among tested candidates is more reliable and auditable
    than one-shot generation (design §4.1). No layout is described or assumed.
    """
    return (
        "You are inferring the LOGICAL column schema of a business/accounting "
        "table from evidence gathered off the page. Do not assume any particular "
        "layout or column count.\n\n"
        f"{_SCHEMA_DISCOVERY_PRINCIPLES}\n\n"
        "Evidence:\n"
        f"{dossier}\n\n"
        "Propose 2 to 4 candidate logical schemas that could explain this "
        "evidence (for example, one with and one without a debated optional "
        "column). For each candidate list its logical columns in order -- name, "
        "value kind, whether its header is printed, and its parent header if any "
        "-- and justify it against the evidence. Output ONE JSON object:\n"
        '{"candidates": [{"columns": [{"name": "<meaning>", '
        '"kind": "<text|numeric|symbol|amount|percent|date|other>", '
        '"printed": <true|false>, "parent": "<parent header or null>"}, ...], '
        '"justification": "<why this fits>"}, ...]}\n'
        "No markdown fences, no extra text."
    )


def schema_candidates_json_schema() -> dict:
    """Guided-JSON schema for the candidate-generation response."""
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "kind": {"type": "string", "enum": list(COLUMN_KINDS)},
                                    "printed": {"type": "boolean"},
                                    "parent": {"type": ["string", "null"]},
                                },
                                "required": ["name", "kind"],
                                "additionalProperties": False,
                            },
                        },
                        "justification": {"type": "string"},
                    },
                    "required": ["columns"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def parse_candidates(raw: str) -> list[CandidateSchema]:
    """Parse a candidate-generation response into ``CandidateSchema`` objects."""
    if not raw:
        return []
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if "```" in text:
        for block in text.split("```")[1::2]:
            body = block.split("\n", 1)[-1] if block[:12].lower().startswith("json") else block
            if '"candidates"' in body:
                text = body
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    out: list[CandidateSchema] = []
    for cand in payload.get("candidates", []) if isinstance(payload, dict) else []:
        if not isinstance(cand, dict):
            continue
        cols = []
        for col in cand.get("columns", []):
            if isinstance(col, dict) and col.get("name") and col.get("kind"):
                cols.append(
                    ColumnSpec(
                        name=str(col["name"]),
                        kind=str(col["kind"]),
                        printed=bool(col.get("printed", True)),
                        parent=col.get("parent") or None,
                    )
                )
        if cols:
            out.append(CandidateSchema(columns=cols, justification=str(cand.get("justification", ""))))
    return out


# --- deterministic trial and selection ----------------------------------------


def survey_candidate(survey: Survey) -> CandidateSchema:
    """A zero-model baseline candidate straight from the survey: one logical
    column per attested track, named generically, kind = the track's induced
    kind. Always included in the pool so selection has a grounded fallback."""
    cols = [
        ColumnSpec(name=f"col{pos}", kind=t.kind, printed=True)
        for pos, t in enumerate(_attested_tracks(survey))
    ]
    return CandidateSchema(columns=cols or [ColumnSpec("col0", "text")], justification="survey baseline")


def score_candidate(candidate: CandidateSchema, survey: Survey) -> ScoreReport:
    """Trial-assign the body under a candidate and count violations.

    Attested tracks are matched to the candidate's logical columns left-to-right;
    each body value that its column's kind rejects is a type violation, and the
    difference between the declared column count and the attested-track count is
    the (heavily weighted) count delta. This is the test that lets the body --
    not the header -- decide the schema, and that catches an over- or
    under-declared optional column.
    """
    tracks = _attested_tracks(survey)
    delta = candidate.n_cols - len(tracks)

    type_violations = 0
    for track, col in zip(tracks, candidate.columns):
        for value in _track_body_values(survey, track.index):
            if not kind_accepts(col.kind, value):
                type_violations += 1

    return ScoreReport(candidate=candidate, column_count_delta=delta, type_violations=type_violations)


def select_schema(candidates: list[CandidateSchema], survey: Survey) -> Selection:
    """Score every candidate and commit at the point of maximal evidence.

    Returns the ranked reports, the carried candidate(s) -- the winner alone, or
    the top two when their scores are within ``_CARRY_MARGIN`` (deferred
    commitment: let downstream verification break the near-tie) -- and an
    ``out_of_family`` flag when even the best candidate scores badly.
    """
    reports = sorted((score_candidate(c, survey) for c in candidates), key=lambda r: r.total)
    if not reports:
        return Selection()
    carried = [reports[0].candidate]
    if len(reports) > 1 and reports[1].total - reports[0].total <= _CARRY_MARGIN:
        carried.append(reports[1].candidate)
    return Selection(
        ranked=reports,
        carried=carried,
        out_of_family=reports[0].total > _OUT_OF_FAMILY,
    )


def discover_schema(
    survey: Survey,
    generate_fn: Callable[..., str] | None = None,
) -> Selection:
    """End-to-end discovery: abduce (if a model is given) + test + select.

    ``generate_fn(instruction, guided_json) -> raw`` is the injected Qwen call;
    when omitted, only the deterministic survey baseline is tested (a valid,
    model-free mode). The survey baseline is always in the pool so selection
    degrades gracefully if the model returns nothing usable.
    """
    candidates: list[CandidateSchema] = []
    if generate_fn is not None:
        raw = generate_fn(build_schema_discovery_instruction(build_dossier(survey)),
                          schema_candidates_json_schema())
        candidates.extend(parse_candidates(raw))
    # Survey baseline appended LAST so that, with a stable sort, an equally-scoring
    # model candidate (richer names + parent grouping) wins the tie; the baseline
    # still guarantees a grounded fallback if the model returns nothing usable.
    candidates.append(survey_candidate(survey))
    return select_schema(candidates, survey)


def schema_to_hint(candidate: CandidateSchema) -> str:
    """Render a chosen candidate into the ``build_cells_instruction(schema=...)``
    hint slot -- the existing plumbing for feeding a known header to the cell
    prompt. Groups leaves under their parent and notes implicit columns."""
    by_parent: dict[str, list[ColumnSpec]] = {}
    order: list[str] = []
    for col in candidate.columns:
        key = col.parent or ""
        if key not in by_parent:
            by_parent[key] = []
            order.append(key)
        by_parent[key].append(col)

    lines = ["Discovered logical schema (verify against what is printed):"]
    for key in order:
        leaves = by_parent[key]
        rendered = ", ".join(
            f"{c.name} [{c.kind}]" + ("" if c.printed else " (implicit)") for c in leaves
        )
        lines.append(f"  {key} -> {rendered}" if key else f"  {rendered}")
    return "\n".join(lines)
