# Qwen-Centric System Design for Logical Reconstruction of Business Tables

## Context

The baseline — one call, photo → Qwen3.6-35B-A3B-FP8 → HTML — performs poorly on the
accounting-style tables described in `layout_description.md`. This document is a
first-principles, accuracy-only system design for reconstructing the *logical* table
(right columns, right cells, blanks kept blank), produced under the following binding
constraints:

- **Qwen3.6-35B-A3B-FP8 is the core and only reasoning model.** The design's job is to
  maximize *its* accuracy through system architecture, not to route around it.
- **Specialist models (PaddleOCR-VL, MinerU2.5-Pro) are out of scope** — future work only.
  The objective now is to maximize Qwen's performance through system design.
- **The layout in `layout_description.md` is one hard instance, not a template.** No
  hard-coded schemas, no layout-specific rules ("Figure100 has three columns" is
  forbidden knowledge). Every deterministic rule must derive from layout-independent
  document principles. The system must generalize to unseen column counts, header
  hierarchies, merged-cell configurations, and accounting formats within the
  business-table family.
- Inference speed does not matter; reconstruction accuracy is the only objective.
- Everything self-hosted; the confidential pages never leave the network.

The document proceeds in the order the reasoning demands: the latent reasoning tasks
(§1), the order of inference and why (§2), why the Image → HTML formulation is wrong
(§3), the proposed architecture (§4), its self-critique (§5), two fundamentally
different architectures and an objective comparison (§6), the final recommendation
(§7), the evaluation plan (§8), and the mapping onto this repository (§9).

---

## 1. The latent reasoning tasks Qwen must solve

Before any pipeline is drawn, name the inferences hiding inside "reconstruct the table."
Each has different locality, different verifiability, and different failure blast radius —
that is what dictates the architecture.

1. **Reading (transcription + localization).** Enumerate every printed fragment with its
   position. Purely local; verifiable against pixels; errors are contained (one wrong
   value in the right cell).
2. **Geometric surveying.** Vertical alignment tracks, line spacing, whitespace
   corridors, ruling lines where present. A measurement, not a judgment.
3. **Foreground/background discrimination.** Table content vs stamps, signatures,
   handwriting, QR text, footnotes. Local, semantic-lite.
4. **Header interpretation.** Which region is header; which printed labels span which
   alignment tracks; the parent–child candidate structure the *print* supports.
5. **Column-semantics induction.** For each track, what kind of values does the body
   put there (amounts, percentages, short codes/symbols, dates, prose)? Aggregation over
   local evidence.
6. **Logical schema inference.** The keystone: reconcile what the header *claims* with
   what the body *attests* into a set of logical columns with parent grouping — including
   columns with no printed label (body-attested tracks) and optional columns whose
   existence only body evidence can establish. Global; one wrong bit here shifts entire
   column regions; the hardest task and the least locally verifiable.
7. **Row segmentation.** Visual lines → logical rows; wrapped text merged; implicit
   boundaries recovered from spacing and alignment, not rules.
8. **Row typing / grouping.** Section-header rows vs data rows vs summary rows — the
   row-axis analogue of task 6, defining the record-group structure.
9. **Cell assignment.** Every fragment to exactly one (logical row, logical column);
   every truly blank cell left blank. Mostly geometric; the contested residue is
   semantic.
10. **Merged-cell reasoning.** Spans in the header hierarchy, section rows, summary
    rows; occasionally in the body. Parasitic on 6–9 — spans ride on correct coordinates.
11. **Consistency verification.** Global coherence: nothing invented, nothing dropped,
    types coherent per column, arithmetic identities plausible, blanks genuinely ink-free.

The dependency structure: 1–3 depend on nothing; 4–5 depend on 1–2; 6 depends on 4–5;
7 depends on 1–2 and weakly on 5; 8 depends on 7 and partially 6; 9 depends on 6+7+8;
10 on 6–9; 11 on everything. Crucially there is one **cycle**: schema (6) constrains
assignment (9), but assignment evidence is what validates a schema. That cycle cannot be
solved by any forward pass — it demands hypothesize–test–revise.

---

## 2. The order of inference — what first, what later, and why

Five ordering principles, each derived from general document reasoning rather than any
layout:

1. **Perceive before theorizing.** Transcription must be *unconditioned* on structural
   hypotheses. If reading happens under a schema expectation, perception bends toward
   the theory (confirmation bias) and can no longer falsify it — verification would then
   check the model against itself. An early, structure-blind reading pass is what makes
   every later check independent evidence.
2. **Measure before reasoning.** Alignment tracks and line pitch are measurements a
   deterministic algorithm gets exactly right from fragment boxes; a VLM only
   approximates them. Handing Qwen measured geometry converts "estimate where things
   are" into "reason about what they mean" — moving each subtask to its best solver.
3. **Induce local regularities before global structure.** Column-type signatures are
   near-certain aggregates of near-certain observations. They should exist *before*
   schema inference so the schema is chosen against typed evidence, not raw pixels.
4. **Defer the most consequential commitments to the point of maximal evidence.** The
   schema is the highest-blast-radius decision, so it must be the *most provisional*
   one: propose few candidates early, commit only after trial assignment shows which
   candidate the whole body supports. The baseline does the exact opposite — the
   schema is implicitly frozen by the first emitted header tokens, at the moment of
   least information.
5. **Verify with signals not used during construction.** Ink statistics, arithmetic
   identities, and re-rendering comparisons are held out of the construction path so
   they remain genuinely diagnostic at the end.

Resulting order: read → survey → induce types → hypothesize schemas (few, explicit) →
segment/type rows → trial-assign → select schema by test outcome → full assignment →
verify → targeted repair → serialize. Rows and schema are deliberately parallel-ish:
row segmentation needs geometry and types but not the schema; schema testing needs rows
only coarsely. The cycle of §1 is resolved as a bounded loop, not a forward pass.

---

## 3. Why Image → HTML is the wrong formulation

The single call fails structurally, not because the prompt is bad:

- **Blanks must be *emitted*.** In an autoregressive HTML stream, a blank cell is a
  token the model must produce on evidence of absence while tracking an implicit
  which-column counter across thousands of tokens. LM priors favor contentful
  continuation; one skipped blank shifts every later value in the row and still parses.
  Sparse tables — this dataset's defining trait — maximize exposure.
- **The schema is decided before it can be known.** The header is emitted first, but its
  logical width depends on the body. Emission order forces the global decision at the
  moment of least evidence, implicitly, as a byproduct of token order.
- **A near-invariant printed header rewards template emission.** Fluent copying of the
  header plus a plausible hallucinated body looks right and scores deceptively.
- **Perception is stretched past its limit.** Photographed pages, small print, and an
  FP8 MoE with ~3B active parameters per token: digits and one-glyph codes are exactly
  the content most easily misread, and a full-page single look gives them the fewest
  pixels.
- **No precise geometry.** Column membership of a lone value in a sparse region is an
  x-interval measurement; attention approximates, code measures.
- **Ungrounded output is unverifiable.** A bare HTML string offers nothing to check a
  value against — hallucinations are undetectable, so they are also unrepairable.
- **Long outputs degrade.** Hundreds of cells → thousands of tokens → drift, repetition,
  truncation, worst at the bottom where summary rows live.

**Reformulation.** Replace "generate HTML" with **inference over explicit intermediate
structures**: a fragment set F (perception), a geometric skeleton G (measurement),
schema and row hypotheses S, R (theory), an assignment A: F → S×R (grounded mapping),
and HTML produced only by deterministic serialization of (S, R, A). Qwen appears in
several *roles* — reader, schema theorist, row judge, assigner, auditor — each a
separate call with a purpose-built representation whose failure modes are either
impossible (guided JSON; required fields; blanks as holes, not tokens) or
machine-detectable (fragment-ID references; explicit coordinates per cell so nothing
depends on implicit counters). This is the formulation change; everything below is its
elaboration.

---

## 4. Architecture A (proposed): staged grounded decomposition, one model in many roles

All model calls are Qwen3.6-35B-A3B-FP8 via vLLM with `guided_json` where structure is
required; thinking ON for theory-building roles (schema, contested assignment, audit),
OFF for the literal-perception role (reading should transcribe, not narrate).

**Stage 0 — Normalization (deterministic, validated).** Deskew/dewarp and illumination
flattening estimated from the document's own text baselines and alignment structure —
the general principle that print is organized in straight parallel lines, not any layout
rule. Every rectification is *validated* (alignment variance must improve, else fall
back to the raw image). Output includes an invertible coordinate map. Failure mode:
curved-page overcorrection — bounded by the validation gate.

**Stage 1 — Table localization.** The existing detector crop, generous margins (the
clipped-edge failure is already documented in `prompts.py`); continuation pages kept as
an ordered page list. Multi-page identity is carried through every later stage
(`p` fields exist end-to-end).

**Stage 2 — Reading pass (Qwen as its own OCR).** With specialists out of scope, the
design makes self-grounding reliable by construction rather than by hope:

- **Tiling:** overlapping high-resolution crops so small print reaches the encoder at
  legible scale; per-tile reading; deterministic merge of overlaps (dedup by box + text
  agreement). Layout-independent.
- **Structure-blind contract:** the reading prompt asks only for fragments — text + box
  per fragment, guided JSON, required fields — never for rows, columns, or meaning
  (principle 1 of §2).
- **Consensus:** two to three reads with different tiling offsets; agreements become
  high-confidence fragments, disagreements carry alternates and a flag.
- **Unread-ink sweep (recall net):** binarize the rectified crop; any ink mass not
  covered by a fragment box triggers a targeted re-read of exactly that region at
  maximum zoom. Deterministic, layout-independent, and the answer to "the OCR missed
  faint print."

Output: fragment list — id, text (+ alternates), box, page, confidence.
Assumption (named, tested in §5): Qwen's localization is *ordinally* reliable — boxes
need to land in the right track/line, not be pixel-perfect.

**Stage 3 — Geometric survey (deterministic).** Extends `src/ocr/layout.py`: vertical
alignment tracks from fragment x-intervals with per-track alignment type chosen by fit
(left/right/center — numeric columns right-align, prose left-aligns: a general
typographic fact, not a layout rule); whitespace corridors keep multi-word cells whole;
line pitch estimated from the baseline-gap distribution; candidate row bands; ruling
lines as extra evidence where present. Ambiguities (straddling fragments, mergeable
lines) are carried as explicit alternatives, not silently decided. Output: tracks, row
candidates, a provisional fragment→(track, band) map, and a contested list.

**Stage 4 — Column-type induction.** Each track gets a value-kind signature induced
from its own body fragments (amount / percent / numeric / short-code / date / prose —
the `COLUMN_KINDS` vocabulary already in `cells.py`), by general lexical form plus a
Qwen text call for residuals. No track is *required* to be anything; the signature is
evidence, discovered per document.

**Stage 5 — Schema discovery (Qwen as theorist; the keystone).** Detailed in §4.1.
Qwen abduces a small set of candidate logical schemas from an evidence dossier (header
fragments with their spans over tracks; per-track type signatures; sparsity profile),
under general principles only. Candidates are then *tested*, not trusted: a cheap
deterministic trial assignment under each candidate counts violations; the winner —
or the top two, when close — proceeds. Commitment is deferred (principle 4).

**Stage 6 — Row segmentation and typing.** Detailed in §4.2. Anchor tracks are
*discovered* (a track filled in nearly every candidate row — not assumed to be "the
first column"); wrapped lines merge by fill-signature; rows are typed
(header / section / data / summary) by general signatures with a soft row grammar as a
consistency check; residual ambiguity goes to Qwen as small named questions.

**Stage 7 — Grounded cell assignment (Qwen as assigner).** Detailed in §4.3. The
uncontested majority of fragments is pre-locked deterministically (one track, one band,
type-compatible). Qwen resolves the contested residue and all span decisions, emitting
the `cells.py` JSON cell list extended with **fragment-id references** — cell text is
looked up from the reading pass, never re-typed, so inventing a value is impossible and
dropping one is set arithmetic. Blanks remain grid holes. k samples with different
conditioning (grid-style vs coordinate-style serialization of the survey) are drawn;
per-cell voting; disagreement entropy becomes per-cell confidence.

**Stage 8 — Verification and targeted repair.** Detailed in §4.4. Deterministic,
layout-independent checks — coverage, occupancy, band consistency, ink-in-blank,
type coherence against *induced* signatures, discovered arithmetic identities, schema
echo — feed a bounded, monotone repair loop (violations quoted back verbatim; a
revision that increases violations is rejected; the argmin is kept). If two schemas
were carried, the violation count decides here — the schema commits at the point of
maximal evidence.

**Stage 9 — Independent audit (Qwen as auditor).** The reconstruction is rendered back
into a clean table and shown beside the original; Qwen lists cell-level discrepancies.
Visual diff is a far easier task than generation, and the conditioning differs from
every construction step, which is what gives the audit some independence despite being
the same weights. Deterministic cycle checks accompany it (claimed cell boxes must
cover their fragments; blanks must be ink-free).

**Stage 10 — Serialization, confidence, triage.** `cells_to_html` (deterministic,
unchanged semantics: spans, holes → empty cells, page tags; multi-page stitch — header
dedup, break-row merge — on the logical grid first). Per-cell and per-table confidence
aggregated from reading consensus, schema margin, vote entropy, violation history, and
audit agreement, calibrated on the labeled public proxy set; low-confidence tables go
to human side-by-side review, and verified outputs become distillation labels for the
student (deployment story, unchanged).

### 4.1 Schema discovery without a template (generalized)

A schema *library* of known layouts is disallowed — rightly, since the described layout
is one instance. What survives is the **mechanism**: schemas as explicit, few,
*testable hypotheses*, because hypothesis selection is more reliable, more calibrated,
and more auditable than one-shot generation. Only the hypothesis *source* changes: from
a config file to Qwen's own abduction, constrained by general principles that hold for
the whole business-table family (they are, verbatim, the "Reconstruction Principles" of
`layout_description.md`, which are layout-free):

- Every body-attested alignment track belongs to some logical column; a persistent
  track with no printed label is an **unprinted column**, not noise.
- A printed header label may span several tracks (a parent) or one (a leaf); parent
  labels partition the tracks under their x-span.
- An **optional column exists iff the body attests it** — fragments on a distinct track
  whose values do not fit the neighboring columns' induced types. One stray fragment is
  not attestation; track fit *and* type misfit with neighbors, or multiple independent
  fragments, are.
- Columns never merge because one is sparse; blanks are content.
- Type coherence is table-global: under the true schema, every fragment lands in a
  column whose induced signature accepts it, across all rows. A single row that forces
  a violation falsifies the candidate — sparsity makes each fragment a *strong*
  constraint rather than a weak one.

Procedure: Qwen (thinking on) receives the evidence dossier and produces two to four
candidate schemas — logical columns, parent grouping, printed/unprinted status, and for
each candidate the specific fragments that support or strain it (the justification is
part of the output, making the decision reviewable). Deterministic trial assignment
under each candidate scores violations; clear winner commits, near-tie carries two
candidates into stage 7 and lets stage 8's full-evidence violation count decide.
**Novelty detection falls out free:** if every candidate scores badly, the document is
flagged as out-of-family — routed to the holistic fallback (§7, Architecture B) plus
human review rather than silently mis-fit. A data-derived schema memory (past verified
schemas biasing future hypothesis generation for recurring document families) is future
work; nothing in the core loop depends on it.

### 4.2 Row inference (generalized)

- **Pitch first:** the mode of baseline gaps gives the document's line rhythm; row
  candidates are baseline clusters at that rhythm. Robust to sparse regions and section
  whitespace; adaptive, not a fixed tolerance.
- **Anchors are discovered:** any track filled in nearly all candidate rows is an
  anchor; anchor fragments seed logical rows; value-only lines attach to the nearest
  seed above. (Assuming "the description column is always filled" would be a layout
  fact. Discovery replaces assumption; if no anchor exists, seeding falls back to
  pitch alone and the ambiguity surfaces as lower row confidence.)
- **Wrapped lines merge by fill signature:** a line holding only prose-typed content,
  within one pitch of the line above, indented under a prose cell, is a continuation.
  A legitimately sparse data row differs precisely in holding value-typed content. This
  distinction uses induced types (stage 4), not layout knowledge.
- **Row typing by general signatures:** section rows carry content only in leading
  track(s) with the rest empty and often extra surrounding space; summary rows sit at
  block ends, participate in arithmetic identities (§4.4), and may carry summary
  lexicon (a *soft*, family-level signal, never a rule); header rows sit above the
  first data row and label tracks rather than fill them. A soft grammar — header block,
  then groups each opened by a section row, summaries near block ends — is a
  consistency check that catches isolated mistypes, enforced as a flag, not a
  constraint solver.
- Merged-cell decisions on the row axis (a section label spanning its block, summary
  labels spanning label columns) are made by Qwen in stage 7 *on top of* fixed row
  identities — spans never substitute for segmentation.

### 4.3 Cell assignment (grounded)

Fragments group into content units by intra-line proximity (whitespace-corridor logic,
existing). Uncontested units — one track, one band, type-compatible — are locked by
code; on well-aligned business tables this is most of the ink, and locking removes the
opportunity for scale mistakes. The contested residue goes to Qwen with per-unit
dossiers: local image crop, candidate slots with geometric and type scores, the row's
partial contents, neighboring cells. Semantics decides what geometry cannot — a
one-glyph code between tracks belongs to the code-typed column its row still lacks; a
number straddling two amount columns is resolved by the arithmetic identities it must
join (§4.4). Global constraints (every unit placed exactly once or explicitly marked
non-table; one unit-set per grid position; spans tile rectangularly) are enforced by
the validator; where a violation has a unique feasible fix, code applies it without
another model call. Blank preservation is structural: blanks are the slots left
unassigned, so shifting a value into a blank now requires an affirmative wrong
assignment that must then survive band, type, and arithmetic checks — instead of the
baseline, where preserving a blank required the model to emit nothing, correctly,
repeatedly.

### 4.4 Verification (all layout-independent)

1. **Syntactic** — free under guided decoding; malformed output is impossible.
2. **Referential** — every fragment assigned exactly once or explicitly excluded (the
   ignore rule); no cell references a nonexistent fragment; `ocr_missed` claims (text
   the model asserts despite the reading pass lacking it) are enumerated and audited.
   Tightens the current token-bag check in `validate_cells` to per-fragment identity.
3. **Geometric** — a cell's box (union of its fragments) must sit inside its claimed
   column track and row band; tracks disjoint and ordered. This is the shifted-cell
   detector: displaced ink betrays itself geometrically.
4. **Ink-in-blank** — every grid hole's implied region must be ink-free; every claimed
   cell must contain roughly the ink its fragments occupy. Ink in a "blank" means
   either missed reading (recall failure → targeted re-read) or a value shifted away.
   Closes the loop from logical claim to raw pixels with no model in the path, and
   targets this family's signature requirement — meaningful blanks — directly.
5. **Type coherence** — each cell's value against its column's *induced* signature;
   outliers flag (they are either errors or the evidence that the schema hypothesis is
   wrong — both worth surfacing).
6. **Discovered arithmetic** — search numeric columns and candidate summary rows for
   identities (column sums, group subtotals) that hold on the document; identities that
   validate widely are then enforced on the residual. A shifted value breaks two sums
   at once — missing in one column, excess in another — a signature that localizes and
   even *suggests* the correction; an identity that fails under one OCR alternate and
   holds under the other simultaneously fixes the reading and confirms the placement.
   Discovered, not assumed, hence layout-independent.
7. **Schema echo** — the emitted `columns` must equal the committed candidate; spans
   must tile within bounds (occupancy check exists).
8. **Audit diff** — stage 9's rendered comparison, with disagreements adjudicated once
   or flagged to a human.

### 4.5 Refinement discipline

Every refinement mechanism is anchored to an external signal; unanchored "look again"
loops oscillate (models revise correct answers as readily as wrong ones). Repair
rounds are bounded; each revision must strictly reduce machine-found violations or it
is rejected; the argmin-violations state is kept. Voting happens across conditioning
diversity, not just temperature, because differently-shaped inputs (grid vs coordinate
serialization, tiling offsets) decorrelate a single model's failures better than
sampling noise does.

### 4.6 Post-processing (deterministic)

Serialization via `cells_to_html`; rectangularity completion and span clipping; number
and symbol normalization by *induced* column kind (currency marks, minus vs hyphen,
thousand separators, NFC, whitespace); multi-page continuation-header dedup and
break-row merge on the logical grid; scoring-dialect normalization (`<th>` policy,
attribute order) so measured deltas are logical, not stylistic; final projection
against the committed schema with anything outside it flagged, never silently kept or
dropped.

---

## 5. Self-critique of Architecture A

**Weakest assumptions, in order of how much rests on them:**

1. **Qwen's self-localization is ordinally reliable.** The whole grounding story —
   tracks, bands, band-consistency checks, ink accounting — consumes boxes produced by
   the same model being checked. If its boxes are systematically sloppy (merged
   fragments, drifted coordinates on dense regions), the survey inherits bias and the
   "deterministic" checks check a distorted map. Tiling, consensus reads, and the
   ink-sweep mitigate recall and precision, but a *systematic* localization bias is the
   design's single point of failure. This assumption is unverified for this checkpoint
   and must be the first thing measured.
2. **The stage decomposition matches the problem's real factorization.** The stages are
   hand-designed. If the model resolves schema and rows *better jointly than
   sequentially* (plausible — humans flick between header and body), a staged system
   locks in a worse factorization. Mitigated by deferred commitment and by carrying
   alternatives, but the risk that decomposition *subtracts* from a strong reasoner is
   real — it is the exact question ablation #1 must answer before anything else is
   built.
3. **One model, many roles ≠ independence.** Reader, theorist, and auditor share
   weights, so they share blind spots: a glyph Qwen misreads at reading time it may
   re-misread at audit time, in every conditioning. Deterministic checks (ink,
   geometry, arithmetic) are the only truly independent evidence in the system; where
   they are silent, correlated error passes. Honest posture: accept the residual risk,
   surface it as confidence, and triage to humans — a second, independently-trained
   witness is exactly the de-scoped future work.
4. **Multi-call compounding.** Ten-ish model interactions per table multiply per-call
   adherence and parse failure rates; every guided schema, repair loop, and merge step
   is surface area. Guided decoding removes the syntactic component but not semantic
   drift between stages.
5. **Geometry on degraded photos.** Curvature and shadow can defeat both rectification
   and track clustering; stage 0's validation gate falls back to raw-image geometry,
   which on a curved page may be poor — degrading exactly the grounding the later
   stages assume.

**Where it likely fails in practice:** consistently misread one-glyph codes (shared
blind spot, invisible to arithmetic); an over-fragmented reading pass that makes unit
grouping noisy and floods stage 7 with contested items; a spurious alignment track from
a stray annotation surviving into schema candidates (foreground filter and
multi-fragment attestation guard it, imperfectly); repair-loop thrash on genuinely
ambiguous layouts, bounded but then flagged rather than solved; and tables whose
body is *so* sparse that even type induction is underdetermined — where every system,
human included, needs the audit loop and low-confidence routing.

---

## 6. Two fundamentally different architectures

**Architecture B — holistic draft–audit–revise.** No stages, no reading pass, no
explicit schema object. Qwen (thinking on) drafts the *entire* logical table directly
from the image in one call — but into the cells-JSON representation, not HTML, keeping
blanks-as-holes and required boxes. k diverse drafts are sampled; per-cell voting
merges them. Then a loop: the deterministic layout-independent checks that need no
fragments (occupancy, span tiling, type self-coherence induced from the draft itself,
discovered arithmetic, ink-in-blank against the draft's own claimed boxes) plus a
rendered side-by-side audit produce a violation list; Qwen revises the whole table;
monotone acceptance; bounded rounds. The schema is whatever the accepted draft
implies. This is "the current `cells.py` path, taken seriously": representation fix +
voting + external-check refinement, with the model's joint reasoning left intact.

**Architecture C — agentic active examination.** Reformulate reconstruction as an
investigation. Qwen runs as an agent with a small, layout-independent toolset: zoom
into a region, measure (projection profiles, track overlays rendered onto the image),
re-read a crop, and write to an evolving workspace (a growing cell map + notes). It
chooses its own inspection order — read the header, zoom the body, test "is there a
third track under this parent?" by looking, revisit. The workspace is the intermediate
representation; the tools are perception amplifiers; control flow is model-driven
rather than pipeline-driven, so the decomposition is *discovered per document* instead
of fixed by the designer.

(A specialist-grounded variant — independent OCR as witness — is the constraint-mandated
future work and is not compared here.)

**Objective comparison.**

| Axis | A: staged grounded | B: holistic refine | C: agentic |
|---|---|---|---|
| Ceiling on sparse, variable-schema tables | Highest: schema is explicit, tested, deferred; every value grounded | Bounded by draft basin: a schema error shared by all k drafts is unrecoverable — revision explores locally | High in principle; adaptivity targets scrutiny where needed |
| Robustness to designer error (wrong factorization) | Weakest — stages are hand-imposed | Strongest — no imposed factorization | Strong — model picks its own |
| Grounding / hallucination control | Strongest: fragment-ID references make invention impossible, drops computable | Weak–medium: values are generated, checked only statistically (voting) and by ink/box checks against self-claimed boxes | Medium: re-reads ground locally; workspace entries are still generated |
| Error visibility & debuggability | Highest: every stage inspectable, every failure attributable | Medium: violations visible, causes opaque | Lowest: trajectories vary per document; failures are process-shaped |
| Dependence on Qwen localization fidelity | High (named assumption #1) | Low — boxes are audit hints, not load-bearing | Medium |
| Reliability burden on the model | Many small, well-posed calls | Few large, hard calls | Long-horizon agency — the hardest ask for a 35B-A3B; compounding tool-use errors, wandering, unbounded cost |
| Implementation risk / complexity | High | Low (mostly exists) | Highest |
| Generalization to unseen layouts | Via general principles + per-document discovery | Native (nothing layout-shaped anywhere) | Native |

The honest reading: **B is the strongest baseline and the correct fallback** — cheapest,
robust to design errors, and it already fixes the representation-level failures
(blanks, spans, boxes). **A dominates exactly on this dataset's defining difficulties**
— meaningful blanks (structural in A, statistical in B), body-decided schemas (explicit
and tested in A, implicit and vote-locked in B), and hallucination (impossible in A,
merely discouraged in B) — at the price of resting on localization fidelity and design
judgment. **C's distinctive value is adaptivity and zooming**, but free-running agency
is the least reliable regime for this model class, and in practice C converges to A
with model-chosen ordering and extra variance: its tools are A's stages.

---

## 7. Final recommendation

**Architecture A as the backbone, with B embedded as its proposal-and-refinement engine
and C's examination tools bounded inside stages — plus B as the graceful-degradation
path.** Concretely: stage 7 *is* B's k-draft-and-vote mechanism operating on A's
grounded representation; stages 8–9 *are* B's external-check refinement loop; targeted
zoom-and-re-read (C's best idea) appears only as bounded tool calls inside reading
(unread-ink sweep) and contested assignment (per-unit dossiers), never as free agency.
When stage 2's grounding proves poor on a given document (measurable: consensus
collapse, ink-coverage residuals), the system degrades to B for that document — the
shared cells-JSON representation makes the two arms interchangeable downstream, and the
same checks, confidence, and triage apply.

Decision-by-decision justification: the reformulation (§3) because the baseline's
failures are representational, not promptable; reading-first because unconditioned
perception is what makes verification evidence rather than echo; deterministic
measurement because geometry is the one subtask with an exact solver; explicit
tested schema hypotheses because the highest-blast-radius decision must be the most
provisional, and selection-with-justification is more reliable and auditable than
implicit commitment; fragment-ID grounding because it converts the worst failure class
from probable to impossible; deferred commitment because pipelines die by locking early
errors; monotone bounded refinement because unanchored self-correction oscillates;
deterministic serialization because syntax must never be a source of error; calibrated
confidence and human triage because at n=20 confidential tables, selective prediction
plus a review flywheel (verified outputs → distillation labels and future hypothesis
priors) is the honest operating mode.

Ordered by expected accuracy-per-effort, the build sequence is: representation +
verification first (B, essentially: cells-JSON + checks + refinement + voting), then
the reading pass and grounding (A's stage 2 + fragment-ID referencing), then explicit
schema discovery, then the survey upgrades — with ablation #1 (B vs A) as the gate
before each increment of machinery. If decomposition does not beat the holistic
refiner on the public proxy, the extra stages do not get built — the design is
explicitly falsifiable.

---

## 8. Evaluation plan and ablations

Metrics exist: TEDS-Struct, span recovery, content placement, blank preservation,
schema-column accuracy, bootstrap CIs (`compare_runs`, honoring NOT SIGNIFICANT).
Quantitative work runs on the public proxy (hard-FinTabNet slice + hand-drawn samples);
the 20 confidential tables are a human-audited case set, never a statistics set.

1. **B vs A** (the gate): holistic refine vs staged grounding, same representation.
2. **Reading-pass grounding on/off** — fragment-ID referencing vs model-typed text:
   invented/dropped counts, content placement.
3. **Explicit schema discovery on/off** — and enumerate-and-test vs free generation:
   schema-column accuracy, and downstream placement conditioned on schema correctness.
4. **Verification families off one at a time** — coverage, band, ink, type,
   arithmetic: marginal repair yield and false-flag rate each.
5. **Deferred schema commitment off** (hard top-1): how often reranking flips.
6. **Voting k ∈ {1,3,5}; temperature-only vs conditioning diversity.**
7. **Tiling on/off in the reading pass; thinking on/off per role.**
8. **Oracle substitutions** (error attribution): gold fragments, gold schema, gold rows
   injected one at a time — bounds each stage's contribution to residual error and
   directs the next unit of effort.

---

## 9. Mapping to this repo — implementation roadmap (priority order)

1. **Representation + checks (the B core):** extend `src/model/cells.py` with optional
   fragment-id references (`f` per cell) and `ocr_missed`; tighten `validate_cells`
   coverage to per-fragment identity; new layout-independent verifier module
   (`src/eval/verify.py`: band consistency, ink-in-blank via PIL with deferred imports,
   induced-type coherence, discovered-arithmetic checks) emitting
   `build_repair_suffix`-compatible strings so the existing repair loop consumes them
   unchanged; k-sample voting in the served-endpoint driver.
2. **Reading pass:** new guided fragment schema + structure-blind prompt (pattern of
   `cells.py`/`prompts.py`), tiling + merge + consensus + unread-ink sweep
   (`src/ocr/read.py`), producing `OcrWord`-compatible fragments so `layout.py` and
   the grounding block work unchanged.
3. **Schema discovery:** new `src/model/schema_infer.py` — evidence dossier from the
   survey, Qwen candidate generation (general-principles prompt), deterministic trial
   scoring, deferred commitment; winner feeds `build_cells_instruction(schema=...)` as
   the existing hint slot.
4. **Survey upgrades** in `src/ocr/layout.py`: alignment-aware tracks, pitch
   estimation, anchor discovery, fill-signature row typing, contested lists.
5. **Drivers:** wire into `two-stage-reconstruct.ipynb` (Track 2) and a Track-1 CLI for
   the ablation ladder; confidence aggregation into `RunSummary`; triage coloring in
   the demo renderer.
6. `paddle_client.py` / `mineru_client.py` remain in-repo but are re-labeled future
   work (independent witness) in docs; no new effort.

Repo invariants preserved: deferred heavy imports (every module Mac-importable),
prompts passed not edited, train/inference prompt match for any later distillation,
registry `data` gate, one model in VRAM.

## Verification plan

- **Mac unit tests** (pure Python): fragment-id validation and round-trip in
  `tests/test_cells.py`; verifier checks and schema trial-scoring on synthetic fragment
  sets, including the spec's own hard cases (2- vs 3-child parents, unprinted columns,
  section rows, wrapped lines, meaningful blanks).
- **Public-track ablation ladder** with CIs — B first, each increment gated on a
  significant win.
- **Track 2:** full run on the 20 confidential tables, per-table human side-by-side
  audit; violation/flag rates and coverage–accuracy curve as the operational report.
