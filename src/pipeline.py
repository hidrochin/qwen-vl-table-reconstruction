"""End-to-end staged reconstruction -- the runnable assembly of the cores.

``pipeline_design.md`` (Architecture A) factors "reconstruct the table" into
stages, each a purpose-built call whose failures are impossible or machine-found.
The individual cores live in ``src/ocr`` (reading pass, geometric survey),
``src/model`` (schema discovery, grounded cell list) and ``src/eval`` (the
layout-independent verifier). This module is the missing wiring: it runs them in
order on a served model and returns one :class:`PipelineResult`.

The order (design §2): **read** (structure-blind fragments) -> **survey**
(deterministic geometry) -> **discover schema** (Qwen abduces, the trial scorer
tests) -> **grounded cells** (fragment-id references, k diverse drafts, vote) ->
**verify** (repairable ``problems`` + soft ``flags``) -> bounded, monotone
**repair** -> deterministic **serialize**. Blanks are grid holes throughout;
values are looked up from fragments, never retyped, so invention is impossible.

Every model call is *injected*, matching the cores' convention, so the whole
pipeline imports and unit-tests on the Mac with fakes (heavy PIL/model work is
deferred into ``ReadingPass`` and ``verify``). On the company GPU box, build the
callables from a served endpoint in one line::

    from src.model.vllm_client import VLLMTableReconstructor
    from src.pipeline import StagedPipeline

    client = VLLMTableReconstructor(base_url="http://localhost:8000/v1", thinking=True)
    pipe = StagedPipeline.from_client(client, k=3, max_repair_rounds=1)
    result = pipe.run("data/invoices/0001.png")   # one image, or a page list
    print(result.html)

``notebook/run_pipeline.py`` is the CLI wrapper around exactly this; see
``RUNBOOK.md`` for the serve-then-run order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from statistics import mean
from typing import Callable

from src.eval.verify import VerificationReport, verify
from src.model.cells import (
    build_cells_instruction,
    cells_json_schema,
    cells_to_html,
    fill_cells_from_fragments,
    parse_cells,
    vote_cells,
)
from src.model.schema_infer import Selection, discover_schema, schema_to_hint
from src.ocr.layout import Survey, survey
from src.ocr.read import Fragment, ReadingPass, render_fragment_block

# Injected model callables. Kept as bare callables (not a client object) so the
# pipeline never imports a backend and a fake is a two-line function in a test.
ReadFn = Callable[..., str]  # (image, instruction, guided_json) -> raw
GenerateFn = Callable[..., str]  # (instruction, guided_json) -> raw
CellFn = Callable[..., str]  # (image, instruction, ocr_layout, guided_json, temperature, seed) -> raw


@dataclass
class PipelineResult:
    """Everything one table's reconstruction produced -- output plus its evidence.

    ``payload``/``html`` are the answer; the rest is why-to-trust-it: the read
    ``fragments``, the geometric ``survey``, the schema ``selection`` (ranked
    candidates + what was carried), the final ``report`` (repairable problems and
    soft flags), per-cell ``confidence`` from voting, and a heuristic scalar
    ``table_confidence`` for triage. ``notes`` is a human-readable log of what
    each stage decided.
    """

    payload: dict
    html: str
    fragments: list[Fragment]
    survey: Survey | None
    selection: Selection | None
    report: VerificationReport
    confidence: dict[tuple[int, int], float]
    schema_hint: str | None
    repair_rounds: int
    table_confidence: float
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the final payload has no repairable problems (flags may remain)."""
        return bool(self.payload.get("cells")) and self.report.ok


def _reindex(fragments: list[Fragment]) -> list[Fragment]:
    """Give fragments globally-unique ids after concatenating several pages.

    Each page's reading pass numbers its fragments 0..n locally; concatenating
    pages would collide those ids, and the grounded cell list references them, so
    the whole set is renumbered in (already reading-order) sequence."""
    return [replace(f, id=i) for i, f in enumerate(fragments)]


@dataclass
class StagedPipeline:
    """Run the staged reconstruction with injected model calls.

    Build it from a served client with :meth:`from_client`, or pass the three
    callables directly (tests, or an alternate backend). ``run`` does the reading
    pass then delegates to :meth:`run_from_fragments`; call the latter directly
    when OCR/fragments come from elsewhere.

    Knobs map to the design's ablations: ``k`` (vote drafts), ``max_repair_rounds``
    (bounded monotone repair), ``use_schema_hint`` (schema discovery on/off),
    ``consensus_offsets``/``tile``/``sweep`` (reading pass). ``draft_temperature``
    is the diversity source for k>1 -- temperature here, the cheap axis; adding
    conditioning diversity (design §4.5) is a stronger, later axis.
    """

    cell_fn: CellFn
    read_fn: ReadFn | None = None
    generate_fn: GenerateFn | None = None
    k: int = 1
    use_schema_hint: bool = True
    max_repair_rounds: int = 1
    tile: int = 1024
    overlap: float = 0.2
    consensus_offsets: tuple[int, ...] = (0,)
    sweep: bool = True
    draft_temperature: float = 0.4
    seed: int | None = 0

    @classmethod
    def from_client(cls, client, **cfg) -> "StagedPipeline":
        """Wire the three model callables from a served endpoint client.

        ``client`` is a ``VLLMTableReconstructor`` (or anything exposing
        ``as_cell_fn``/``as_read_fn``/``as_generate_fn``). ``**cfg`` sets the
        knobs above.
        """
        return cls(
            cell_fn=client.as_cell_fn(),
            read_fn=client.as_read_fn(),
            generate_fn=client.as_generate_fn(),
            **cfg,
        )

    # --- reading -> fragments -------------------------------------------------

    def read_fragments(self, image, *, page_start: int = 1) -> list[Fragment]:
        """Reading pass over one table (a path or an ordered page list) -> fragments.

        Tiles each page, reads structure-blind, merges cross-tile/cross-read, runs
        the unread-ink sweep, then renumbers ids globally across pages. Needs
        ``read_fn`` and PIL (both live only on the GPU box)."""
        if self.read_fn is None:
            raise ValueError("read_fragments/run need read_fn; use run_from_fragments otherwise")
        paths = _as_list(image)
        rp = ReadingPass(
            predict_fn=self.read_fn,
            tile=self.tile,
            overlap=self.overlap,
            consensus_offsets=self.consensus_offsets,
            sweep=self.sweep,
        )
        frags: list[Fragment] = []
        for i, p in enumerate(paths, page_start):
            frags.extend(rp.read(p, page=i))
        return _reindex(frags)

    def run(self, image, *, page_start: int = 1) -> PipelineResult:
        """Full pipeline from image path(s): read, then reconstruct."""
        paths = _as_list(image)
        fragments = self.read_fragments(paths, page_start=page_start)
        return self.run_from_fragments(
            fragments, image=paths, n_pages=len(paths), images_for_ink=paths
        )

    # --- reconstruction from fragments (pure orchestration; Mac-testable) -----

    def run_from_fragments(
        self,
        fragments: list[Fragment],
        *,
        image,
        n_pages: int = 1,
        images_for_ink=None,
    ) -> PipelineResult:
        """Survey -> schema -> grounded cells (k-vote) -> verify -> repair.

        ``image`` is passed to ``cell_fn`` (the model still sees the pixels while
        it references fragments); ``images_for_ink`` (paths/PIL, default =
        ``image``) is what the ink-in-blank check opens. With fake callables and
        ``images_for_ink=None`` this runs with no model and no PIL.
        """
        notes: list[str] = []

        sv = survey(fragments) if fragments else None
        selection, hint = self._discover_schema(sv, notes)

        instruction = build_cells_instruction(schema=hint, with_fragments=True)
        schema = cells_json_schema(with_fragments=True)
        frag_block = render_fragment_block(fragments)

        drafts, last_raw = self._draw_drafts(image, instruction, frag_block, schema, fragments, notes)
        if not drafts:
            empty = {"columns": [], "cells": []}
            report = verify(empty, fragments=fragments, n_pages=n_pages)
            return PipelineResult(empty, "", fragments, sv, selection, report, {}, hint, 0, 0.0, notes)

        if len(drafts) > 1:
            payload, confidence = vote_cells(drafts)
            notes.append(f"voted {len(drafts)} drafts -> {len(payload.get('cells', []))} cells")
        else:
            payload, confidence = drafts[0], {}

        ink = image if images_for_ink is None else images_for_ink
        report = verify(payload, fragments=fragments, images=ink, n_pages=n_pages)
        payload, report, rounds = self._repair(
            payload, report, image, instruction, frag_block, schema,
            fragments, ink, n_pages, notes,
        )

        conf = _table_confidence(fragments, selection, confidence, report)
        return PipelineResult(
            payload=payload,
            html=cells_to_html(payload),
            fragments=fragments,
            survey=sv,
            selection=selection,
            report=report,
            confidence=confidence,
            schema_hint=hint,
            repair_rounds=rounds,
            table_confidence=conf,
            notes=notes,
        )

    # --- stages ---------------------------------------------------------------

    def _discover_schema(self, sv, notes) -> tuple[Selection | None, str | None]:
        if sv is None or not self.use_schema_hint:
            return None, None
        selection = discover_schema(sv, generate_fn=self.generate_fn)
        if selection.out_of_family:
            # No abduced schema fits -> do NOT force a bad one; reconstruct
            # hint-free (design's Architecture-B fallback) and route to review.
            notes.append(
                "schema out-of-family: no candidate fit; hint-free reconstruction "
                "(Architecture B) + human review"
            )
            return selection, None
        if selection.best is not None:
            notes.append(
                f"schema: {len(selection.carried)} candidate(s) carried, "
                f"best={selection.best.n_cols} logical columns"
                + ("" if self.generate_fn else " (survey baseline; no generate_fn)")
            )
            return selection, schema_to_hint(selection.best)
        return selection, None

    def _draw_drafts(self, image, instruction, frag_block, schema, fragments, notes):
        """k grounded cell drafts (draft 0 greedy, the rest temperature-diverse)."""
        drafts: list[dict] = []
        last_raw = ""
        for j in range(max(1, self.k)):
            temp = 0.0 if j == 0 else self.draft_temperature
            seed = None if self.seed is None else self.seed + j
            raw = self.cell_fn(image, instruction, frag_block, schema, temp, seed)
            last_raw = raw or last_raw
            payload = fill_cells_from_fragments(parse_cells(raw), fragments)
            if payload and payload.get("cells"):
                drafts.append(payload)
        if not drafts:
            notes.append("no parseable cell draft")
        return drafts, last_raw

    def _repair(
        self, payload, report, image, instruction, frag_block, schema,
        fragments, ink, n_pages, notes,
    ):
        """Bounded, monotone repair: a revision is kept only if it strictly
        reduces repairable problems, else it is rejected and the argmin stands
        (design §4.5 -- unanchored 'look again' loops oscillate)."""
        best_payload, best_report, rounds = payload, report, 0
        while best_report.problems and rounds < self.max_repair_rounds:
            prev = json.dumps({
                "columns": best_payload.get("columns", []),
                "cells": best_payload.get("cells", []),
            })
            suffix = best_report.repair_suffix(prev)
            if not suffix:
                break
            raw = self.cell_fn(image, instruction + "\n\n" + suffix, frag_block, schema, 0.0, self.seed)
            cand = fill_cells_from_fragments(parse_cells(raw), fragments)
            rounds += 1
            if not cand or not cand.get("cells"):
                notes.append(f"repair {rounds}: unparseable -- kept previous")
                break
            cand_report = verify(cand, fragments=fragments, images=ink, n_pages=n_pages)
            if len(cand_report.problems) < len(best_report.problems):
                best_payload, best_report = cand, cand_report
                notes.append(f"repair {rounds}: {len(cand_report.problems)} problems (accepted)")
            else:
                notes.append(
                    f"repair {rounds}: {len(cand_report.problems)} problems "
                    f"(>= {len(best_report.problems)}; rejected, kept argmin)"
                )
                break
        return best_payload, best_report, rounds


def _as_list(image) -> list:
    return list(image) if isinstance(image, (list, tuple)) else [image]


def _table_confidence(fragments, selection, confidence, report: VerificationReport) -> float:
    """A rough, UNCALIBRATED per-table confidence for triage ordering.

    Combines reading agreement, vote agreement, the schema selection margin, and
    a penalty for outstanding findings. It orders tables for human review; it is
    not a probability. Calibrate against the labeled public proxy before trusting
    a threshold (design §4, stage 10)."""
    read_conf = mean([f.conf for f in fragments]) if fragments else 0.0
    vote_conf = mean(confidence.values()) if confidence else 1.0
    margin = 1.0
    if selection and len(selection.ranked) > 1:
        gap = selection.ranked[1].total - selection.ranked[0].total
        margin = min(1.0, gap / 100.0)  # one logical column apart -> full margin
    penalty = 1.0 / (1.0 + len(report.problems) + 0.25 * len(report.flags))
    return round(read_conf * vote_conf * (0.5 + 0.5 * margin) * penalty, 3)
