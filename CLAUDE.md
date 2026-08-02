# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Semantic-Aware Table Structure Reconstruction using Qwen-VL and Unsloth

## 0. Repository State

**This is a feasibility spike that became a phase-two distillation pipeline, not a
research project.** The goal is to decide whether a **self-hosted** Qwen-VL is accurate
enough on hard financial-invoice tables to commit company resources, and to stand up the
pipeline that turns a strong on-prem teacher into a servable student. Sections 1–12 below
are the original proposal, amended where the spike's constraints made them wrong.

**The real task is in `layout_description.md` — logical-schema *inference*, not
conventional TSR.** The invoice tables have a near-invariant two-level header but a
*variable body schema*: whether an "Optional" column exists is implied by the **body**,
not the header; some columns are never printed; many cells are legitimately blank and must
stay blank (no value-shifting). Recovering the latent logical table — right columns, right
cells, blanks kept — is the problem; reproducing the appearance is not.

**`RUNBOOK.md` is the execution order** — what to run, on which machine, in what sequence.
This file explains *why*; that one says *what to type*.

### Two tracks, one shared library — everything self-hosted, no hosted API

There is **no hosted-API path anywhere in this repo** (Kimi-K2.5 and all frontier/API
models were removed — not self-hostable, not private). `src/` is the shared library; the
two tracks differ only in drivers, data, hardware, and model size:

- **Track 1 — Python, Lightning AI free GPU, PUBLIC data** (FinTabNet + the hand-drawn
  sample). Reproducible CLI drivers: `notebook/bakeoff.py`, `train.py`, `eval.py`.
- **Track 2 — notebooks, company server, PRIVATE data** (the 20 confidential invoices):
  `teacher-label-tables.ipynb`, `finetune-and-serve.ipynb`, `two-stage-reconstruct.ipynb`.
  Nothing leaves the network.

```
src/data/    html_utils, difficulty scoring, FinTabNet loader, image prep (images.py),
             valuetypes.py (layout-independent value-kind classifier — shared by the
             survey, verifier, and schema discovery)
src/ocr/     engine.py (OcrWord + PaddleOCR adapter) + layout.py (geometric grounding
             AND the survey: alignment-typed tracks, pitch, anchors, row typing,
             contested list) + read.py (the reading pass — Qwen as its own OCR:
             structure-blind guided fragments, tiling, cross-read consensus, unread-ink sweep)
src/eval/    TEDS-Struct, span recovery, schema-inference metrics, bootstrap CIs, compare,
             verify.py (layout-independent checks — geometry/type/schema-echo → repairable
             problems; arithmetic/ink-in-blank/ocr_missed → soft flags)
src/model/   registry.py (single source of truth for model roles), prompts
             (structure/full/schema), inference.py (transformers/unsloth/vLLM backends,
             thinking toggle), vllm_client.py (local served-endpoint client + guided
             decoding), cells.py (JSON cell-list output: guided schema, validator,
             repair suffix, cells_to_html — plus the fragment-id grounded path:
             fill_cells_from_fragments, per-fragment identity, ocr_missed, vote_cells),
             schema_infer.py (per-document schema discovery: dossier → Qwen candidates →
             deterministic trial scoring → deferred commitment), paddle_client.py +
             mineru_client.py (specialists), docparse_utils.py (shared parsing)
src/pipeline.py  StagedPipeline — the end-to-end assembly (read → survey → schema →
             grounded cells → k-vote → verify → bounded monotone repair → serialize);
             model calls injected (from_client wires a served endpoint), Mac-testable
src/train/   Unsloth LoRA — structure / schema / OCR-grounded / teacher-distillation
src/demo/    side-by-side comparison renderer
notebook/    Track-1 drivers (bakeoff/train/eval .py); Track-2 notebooks + run_pipeline.py
             (the staged-pipeline CLI) live at repo root
tests/       run these before trusting any number
```

The staged-reconstruction modules above (`valuetypes`, `read`, `verify`, `schema_infer`,
the `layout` survey, the `cells` fragment path) implement `pipeline_design.md` — the
Qwen-only logical-reconstruction architecture that supersedes the two-stage
PaddleOCR/MinerU framing further down. **`src/pipeline.py` is the runnable assembly of
them**, driven on the company GPU box by the `run_pipeline.py` CLI or
`two-stage-reconstruct.ipynb` §6 (`vllm serve` → `StagedPipeline.from_client`). Heavy work
(the model calls in `read`/`schema_infer`, PIL in `verify`/`read`) is injected or deferred;
the deterministic cores and the orchestration are unit-tested on the Mac. Specialists
(`paddle_client`/`mineru_client`) are retained but de-scoped to future work
(independent-witness OCR), not on the main path.

### Status: what has actually been run

Verified end-to-end against live FinTabNet: corpus build (scan → rank → dedup → download →
manifest → leakage check), TEDS-Struct, span recovery, the new schema-inference metrics,
bootstrap CIs, and the comparison renderer. **All `src/` modules import on the Mac with
no torch/vllm/paddle** — the deferred-import invariant holds.

The `pipeline_design.md` modules land as pure-Python cores with the model/PIL calls
injected or deferred, and are covered by Mac unit tests (`tests/test_valuetypes.py`,
`test_read.py`, `test_verify.py`, `test_survey.py`, `test_schema_infer.py`,
`test_pipeline.py`, and the fragment-path additions in `test_cells.py`): value typing,
tiling + cross-read consensus, the verifier families, the survey's track/row typing, schema
trial-scoring + deferred commitment (including the spec's 2- vs 3-child optional-column
case), and the full-pipeline orchestration (k-vote, monotone repair accept/reject,
schema-hint wiring) exercised with fake model callables. The end-to-end loop is now **wired
and runnable** — `src/pipeline.py` + `run_pipeline.py` + `two-stage-reconstruct.ipynb` §6,
with `vllm_client` extended to encode PIL crops and to make a text-only schema call. **Not
yet run on a GPU:** the reading-pass and schema-discovery *model* calls and the end-to-end
staged loop against a live endpoint — treat the first served run as a debugging session
(start with one image; check the fragment count, column count, and problem list), and gate
further staging on the B-vs-A ablation (`--no-schema`, `pipeline_design.md` §7-8).

**Not verified — treat the first run as a debugging session:**
- `inference.py` (transformers/unsloth/**vLLM** backends), `train/lora.py` — never executed
  (no GPU on the Mac).
- `vllm_client.py` — stand up `vllm serve` first; the client is exercised only up to the
  OpenAI-protocol boundary.
- `paddle_client.py` / `mineru_client.py` — the exact result-object shapes are unconfirmed;
  `docparse_utils.extract_table_html` is defensive. Budget a short debugging pass.
  `mineru`/`paddleocr` live in `requirements-onprem.txt`, not `requirements-base`.

### All-local model selection (supersedes the Qwen3-VL-4B/8B framing below)

Every model is open-weights and self-hostable. `data` is the confidential-boundary gate:
`MODELS` in `src/model/registry.py` enforces it, so a Track-1 cloud driver can only
enumerate `public`/`both` models — never a `private` teacher.

| Model (repo id) | Role | Track · hardware |
|---|---|---|
| **Qwen3.6-35B-A3B-FP8** `Qwen/Qwen3.6-35B-A3B-FP8` (MoE 35B/3B) | **on-prem teacher — the plan is pinned to this checkpoint** (user decision 2026-08-02); vision + thinking | private · L40 48 GB / vLLM |
| `Qwen/Qwen3.6-27B` dense | alternate teacher, config swap-in | private · L40 FP8 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | public anchor / ceiling | public · Lightning 24 GB 4-bit |
| **MinerU2.5-Pro** `opendatalab/MinerU2.5-Pro-2605-1.2B` | stage-1 OCR+geometry **+ best-open-Table-TEDS ceiling** (non-AGPL) | both · local 1.2 B |
| **PaddleOCR-VL-1.6** `PaddlePaddle/PaddleOCR-VL-1.6` (0.9 B) | stage-1 OCR / overall-doc leader (96.33 OmniDocBench) | both · local |
| `Qwen/Qwen3-VL-8B`/`4B-Instruct` | trainable **student** (distillation target) | both · Lightning / RTX 5000 |
| `rednote-hilab/dots.ocr`, `zai-org/GLM-OCR` | specialist alternates | both · local |

**Confidential-data boundary (non-negotiable):** the 20 invoices never leave the network.
Their teacher, the specialists that draft their labels, and the student all run **on-prem
(Track 2)**. Track 1 (Lightning cloud) is **public data only**. The registry's `data`
field is the gate — keep it honest when adding models.

### The two-stage approach (SOTA for schema inference)

Matches `layout_description.md`'s own "infer schema → assign fragments → HTML" pipeline and
reuses the OCR grounding already built:
- **Stage 1 — OCR + geometry (local):** PaddleOCR / MinerU words → `serialize_layout()`
  grid or coords block (`src/ocr/`).
- **Stage 2 — schema-conditioned reasoning VLM (thinking on):** hand it the near-invariant
  header schema as a prior (`prompts.build_schema_instruction`), make it infer the variable
  body schema (Optional column?), assign every OCR fragment to a logical cell, and keep
  blanks blank. Force valid HTML with vLLM guided decoding (`vllm_client` `guided_*`).

Two capabilities added on top (2026-08-02):
- **JSON cell-list output** (`src/model/cells.py`) — the recommended high-accuracy
  path since output format is free: `guided_json` makes bbox+page *required* on
  every cell (the "boxes vanish after the first call" failure becomes impossible),
  blanks are grid holes the model cannot shift values into, `columns` must be
  committed before any cell, and `validate_cells` + one repair round-trip machine-
  check overlaps and invented/dropped values. `cells_to_html` feeds the unchanged
  eval/demo stack. A/B this against the HTML+`data-bbox` path before picking the
  distillation target.
- **Multi-page tables** — `predict` (both backends) takes a list of page crops for
  one long table; the prompt gains a stitching note (continuation header once, the
  row cut at the page break merged, `data-page`/`p` on every position) and
  `ocr_layout` becomes per-page. Serve with `--limit-mm-per-prompt image=4`.

### Three constraints that are easy to break by accident

- **Prompts are passed, not edited.** `predict`/`predict_many`/`TrainConfig` all take
  `instruction=`. Editing `src/model/prompts.py` mid-session does nothing — the module is
  already imported, so the old prompt is scored again and it reads as "prompt engineering
  had no effect." (Use `build_schema_instruction(schema)` to vary the header prior.)
- **The training prompt and the inference prompt must match.** For the two-stage path this
  means the *same* `mode="schema"` instruction **and** the same stage-1 `ocr_layouts=` in
  `train()`. A mismatch discards most of what the adapter learned.
- **Both comparison arms must load the same way.** Unsloth's 4-bit path is not bit-identical
  to plain bitsandbytes, so a baseline and a fine-tune loaded differently measure
  quantization as well as fine-tuning. Load both arms identically (`backend="unsloth"` on
  both, or the shared `load_in_4bit` path in `eval.py`).

One model in VRAM at a time. `TableReconstructor.close()` / `free_memory()` exist because
the failure mode is an OOM *after* a training run, not before it.

Environment: Python 3.11 via `uv` (`.venv/`). `requirements-base.txt` is CPU-only (Mac);
`requirements-gpu.txt` adds torch/Unsloth/**vLLM**/openai for Lightning (Track 1);
`requirements-onprem.txt` adds the specialists (`mineru`, `paddleocr`) for the company box
(Track 2). **Every heavy import in `src/` is deferred into function bodies** — torch, vllm,
openai, huggingface_hub, PIL/pillow-heif, paddleocr, mineru — so every module imports on
the Mac. Preserve that when adding code.

Compute: **Lightning AI Studio** (Track 1, public) + the **company server** (Track 2,
private: L40 serving, RTX 5000 training). Persistent filesystem means a disconnect does not
cost the corpus/checkpoints.

Two calibration facts, both measured, both easy to get wrong again:

- FinTabNet difficulty scores run p50=0.12, p90=0.23. An absolute "hard" threshold of
  0.55 matched **1 table in 3,000**. Corpus selection therefore ranks by score and takes
  the top N; bins are assigned by percentile within the selection.
- FinTabNet HTML uses no `<thead>`/`<th>`, so `header_depth` is 0 for every row there.
  That compresses scores uniformly and leaves the ranking intact. PubTabNet activates it.

## 1. Motivation

Table Structure Recognition (TSR) remains a challenging problem, especially for **complex tables** where the structure cannot be recovered solely from visual cues.

Typical OCR-based or rule-based approaches often fail when tables contain:

- Missing or partial borders
- Mixed bordered and borderless regions
- Row spans and column spans
- Nested headers
- Multi-level headers
- Irregular layouts
- Semantic relationships that determine the true structure

For example, two visually separated cells may actually belong to the same logical header, while two adjacent cells may belong to different hierarchical groups. Recovering the correct HTML representation therefore requires understanding both **layout** and **semantic context**.

Recent Vision-Language Models (VLMs), such as **Qwen3-VL**, possess stronger multimodal reasoning abilities than traditional TSR methods. This project investigates whether a modern VLM can directly reconstruct complex table structures without relying on handcrafted rules.

---

# 2. Objective

The objective of this project is to evaluate whether a Vision-Language Model can generate accurate HTML representations for highly complex tables by jointly reasoning over:

- visual appearance,
- document layout,
- semantic relationships between cells.

Instead of predicting bounding boxes or graph structures explicitly, the model is expected to directly generate the final HTML table.

The output should preserve:

- row hierarchy
- column hierarchy
- rowspan
- colspan
- reading order
- logical grouping

---

# 3. Question (amended)

The original framing — "can a VLM reconstruct complex table structures?" — is already
answered in the literature and should not be presented as an open question. PaliGemma 2,
a general VLM fine-tuned on PubTabNet, reports **97.6% S-TEDS**; specialized models
(MuTabNet, GridFormer) sit at **97–98% TEDS-Struct**. Reproducing that is not a result.

The question this spike actually answers is an engineering one:

> Is a **self-hostable** Qwen-VL accurate enough on hard financial-invoice tables —
> merged cells, nested headers, **and (the real target) an implicit, variable logical
> schema** — to justify committing company resources, and does teacher-distillation move
> a small student enough to be worth the pipeline?

**The reframed task (`layout_description.md`) raises the bar past TSR.** Standard TSR
assumes the schema is given and recovers cells. These invoices require *inferring the
logical schema first*: the number of logical columns is not on the page (the "Optional"
column exists only when the body implies it), some child headers are never printed, and
blank cells are real content that must not be back-filled by shifting a neighbour. This is
a **reasoning** problem, which is why the teacher is a dense thinking model and the
approach is two-stage (ground with OCR geometry, then reason over the schema).

Useful calibration for what to expect: Qwen2.5-VL-**32B** scores **81.7 TEDS zero-shot,
83.7 fine-tuned**. Do not benchmark a 4B LoRA run against 97% and call it a failure.

Note for the demo: a large share of any fine-tuning gain will come from **output-format
alignment** — the base model emits structurally sound HTML in a different dialect
(attribute ordering, `<th>` vs `<td>`, whitespace) that TEDS penalizes. That is a real
gain and exactly what domain fine-tuning buys, but it is not "the model learned to reason
about tables." Overselling that distinction costs credibility in month two.

---

# 4. Proposed Method

> **Amended:** the single-stage "image → HTML" method below is the public-data baseline.
> The recommended method for the invoice tables is the **two-stage decoupled** pipeline in
> §0 (stage-1 OCR+geometry grounding → stage-2 schema-conditioned reasoning VLM), because
> the logical schema cannot be read off the pixels alone. The training path mirrors it:
> `TrainConfig(mode="schema")` + `ocr_layouts=` + teacher `targets=` (distillation).

## Input

Single table image.

```
Table Image
```

---

## Model

The project uses

```
Qwen3 Vision-Language Model
```

Examples include

- Qwen3-VL-4B
- Qwen3-VL-8B

depending on available GPU memory.

The model receives both

- image tokens
- text prompt

and generates the HTML sequence autoregressively.

---

## Fine-tuning

Fine-tuning is performed using

```
Unsloth
```

with LoRA.

Only lightweight adapters are trained while the base model remains frozen.

Advantages include:

- lower GPU memory
- faster experimentation
- suitable for Google Colab GPUs

---

## Output

The model predicts HTML directly.

Example

```html
<table>
<tr>
<td rowspan="2">Year</td>
<td colspan="2">Revenue</td>
</tr>
<tr>
<td>USA</td>
<td>EU</td>
</tr>
</table>
```

No intermediate graph prediction or rule-based postprocessing is required.

---

# 5. Dataset (amended)

Primary is **FinTabNet** (`apoidea/fintabnet-html`, config `en`), not PubTabNet.
PubTabNet is biomedical tables from PubMed Central; the target domain is finance
invoices. FinTabNet is financial tables from S&P 500 annual reports — borderless-heavy,
hierarchical headers, far closer to invoice layouts and far more persuasive to a finance
customer. `apoidea/pubtabnet-html` is the fallback and has the same
`{image, html_table}` schema.

**Do not use `load_dataset(..., streaming=True)` to build the corpus.** The train split
is 10.28 GB of parquet and stalls on a home connection. Difficulty is scorable from the
HTML text alone, so `src/data/loader.py` pages the datasets-server `/rows` API (~42
rows/sec, images returned as URLs) and downloads image bytes only for tables that pass
the filter — roughly 25 MB instead of 10 GB.

Train draws from the `train` split, eval from `validation`, so they cannot overlap by
construction. `assert_no_leakage()` verifies it anyway. FinTabNet also repeats tables
(~25% of one sampled eval selection), so the loader deduplicates by normalized HTML.

---

## Dataset Sampling

This project is only a proof-of-concept rather than full-scale training.

Therefore, only a small subset of PubTabNet will be used.

Instead of random sampling, the experiment intentionally selects the **most challenging tables**, including:

- multi-row headers
- multi-column headers
- rowspan
- colspan
- missing borders
- partially bordered tables
- irregular layouts
- dense scientific tables
- complex financial tables

The goal is to stress-test the reasoning capability of the Vision-Language Model.

Expected dataset size:

- approximately 500–2000 complex tables

---

# 6. Prompt Design

The model is prompted to behave as a table reconstruction system.

Example instruction:

> Reconstruct the complete HTML representation of the table. Preserve all row spans, column spans, merged cells, logical hierarchy, and reading order. Return only valid HTML.

No OCR information is provided.

The model must infer the structure directly from the image.

Future experiments may incorporate OCR tokens as additional context.

---

# 7. Experimental Pipeline

```
PubTabNet

        │
        ▼

Select difficult tables

        │
        ▼

Image preprocessing

        │
        ▼

Prompt construction

        │
        ▼

Qwen3-VL

        │
        ▼

LoRA Fine-tuning
(Unsloth)

        │
        ▼

Generate HTML

        │
        ▼

Evaluation
```

---

# 8. Evaluation (amended)

**Primary: TEDS-Struct**, not plain TEDS. Plain TEDS blends OCR errors into the score,
so a misread number is indistinguishable from a broken rowspan — useless when the problem
under study is structure. `src/eval/teds.py` implements both; `structure_only=True` is
the default.

**Secondary: span recovery rate** ("recovered 47 of 52 merged cells"). Position-aware —
a correct span shape in the wrong cell does not count. Customers have intuition for this
number and none for 0.91 TEDS-Struct.

**Schema-inference metrics (for the logical-reconstruction task).** TEDS-Struct and span
recovery measure *structure*; `layout_description.md` also demands the right text in the
right cell and blanks kept blank. `src/eval/metrics.py` adds three position-aware metrics,
threaded through `RunSummary` and `compare_runs`:

- **content placement** — fraction of true text fragments placed in the correct logical
  cell (a right value in the wrong cell does not count — that is the value-shifting
  failure).
- **blank preservation** — fraction of truly-blank cells left blank (catches a neighbour
  shifted into an empty position).
- **schema-column accuracy** — fraction of tables with the right logical-column count (did
  it resolve the Optional/implicit columns).

These are meaningful only for text-emitting runs (`mode="schema"`/`"full"`); on a
structure-only run the prediction has no cell text, so content placement reads ~0 **by
design** — do not read that as a regression.

**Always report bootstrap confidence intervals.** At ~100 eval tables a 2–3 point TEDS
difference is noise. `compare_runs()` prints `NOT SIGNIFICANT` when intervals overlap;
honor that rather than presenting the delta.

Dropped from the original list:

- **Exact Match** — ~0 on complex tables, carries no gradient of information.
- **BLEU** — HTML token sequences share enormous n-gram overlap (`</td><td>` everywhere),
  so it is both inflated and insensitive to the errors that matter.
- **HTML validity** — ~100% after any fine-tuning. Kept only as a zero-shot sanity check
  (`parse_failures` in the run summary).

Training targets are **structure-only** token sequences (tags and span attributes, no
cell text) via `to_structure_only()`. This cuts sequence length 5–10x, matches the
metric exactly, and is what makes the model trainable on a small GPU. Demo *visuals* use
`full` mode so rendered tables carry text — label clearly that the metric is
structure-only.

---

# 9. Development Environment

The experiment is designed to be lightweight and reproducible.

## Hardware

- Apple MacBook M2 Pro (development environment)

---

## IDE

- Visual Studio Code

Extensions:

- Python
- Jupyter
- Pylance

---

## Remote Execution (amended)

Training runs on **Lightning AI Studio**, not Colab. The deciding factor is the
persistent filesystem: on Colab every disconnect costs the corpus and checkpoints, which
over a 4-day sprint is worse than any GPU speed difference. Free tier gives ~22 GPU-h/mo
with L4/A10G.

Colab free-tier T4 remains the fallback — 16 GB, fp16 only (no bf16), aggressive idle
disconnects. If used, `src/train/lora.py` checkpoints every 50 steps and resumes
automatically; do not disable that. Modal is the better long-term platform but its
remote-function and ephemeral-volume model costs setup time this sprint does not have.

**Model sizing: Qwen3-VL-4B for training, 8B for zero-shot only.** Under a deadline,
iteration count beats parameter count — a 4B trains in ~2h leaving room for a second
attempt, while an 8B that takes 8h and OOMs twice leaves nothing. Inference needs far
less VRAM than training, so the 8B zero-shot number is cheap to get and gives a ceiling
estimate. *4B fine-tuned beating 8B zero-shot* is a realistic and compelling outcome.

---

## Framework

- Python 3.11
- PyTorch
- Transformers
- Unsloth
- Hugging Face Datasets
- Jupyter Notebook

---

# 10. Expected Contributions

This project aims to demonstrate that modern Vision-Language Models can reconstruct highly complex table structures without specialized TSR architectures.

Compared with traditional pipelines based on:

- OCR
- cell detection
- graph construction
- heuristic postprocessing

the proposed approach simplifies the workflow into a single end-to-end generative model.

Although the experiment is limited to a subset of PubTabNet, it serves as an initial exploration of semantic-aware table reconstruction using multimodal large language models.

---

# 11. Future Work

Potential future improvements include:

- Incorporating OCR tokens as auxiliary input.
- Comparing Qwen3-VL with other VLMs such as InternVL, GPT-4.1 Vision, and Gemini.
- Evaluating different prompting strategies (zero-shot, few-shot, chain-of-thought).
- Extending experiments to additional datasets such as FinTabNet, SciTSR, and TableBank.
- Exploring larger-scale fine-tuning with higher-capacity GPUs.
- Investigating hybrid approaches that combine geometric priors with semantic reasoning to improve reconstruction of extremely irregular tables.

---

# 12. Limitations

This project is intended as a proof-of-concept and therefore has several limitations:

- Only a small subset of PubTabNet is used.
- Training is constrained by the GPU resources available in Google Colab.
- Results may not generalize to all document domains.
- The study focuses on HTML structure reconstruction and does not explicitly evaluate OCR quality or downstream information extraction tasks.