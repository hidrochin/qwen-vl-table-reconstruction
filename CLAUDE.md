# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Semantic-Aware Table Structure Reconstruction using Qwen-VL and Unsloth

## 0. Repository State

**This is a 4-day feasibility spike, not a research project.** Customer demo is
Wednesday 22 July 2026. The goal is to decide whether Qwen3-VL is worth committing
company resources to for finance invoice table extraction — not to produce a novel
finding. Sections 1–12 below are the original proposal, amended where the spike's
constraints made them wrong.

**`RUNBOOK.md` is the execution order** — what to run, on which machine, in what
sequence, and what to do when a step fails. This file explains *why*; that one says
*what to type*.

Implementation lives in `src/`; `qwen-vl-table-reconstruction.ipynb` is a thin driver
that imports from it. Notebooks-as-codebase is how this becomes unreproducible.

```
src/data/    html_utils, difficulty scoring, FinTabNet loader
src/eval/    TEDS-Struct, span recovery, bootstrap CIs, run comparison
src/model/   prompts, output cleaning, Qwen3-VL inference
src/train/   Unsloth LoRA
src/demo/    side-by-side comparison renderer
tests/       56 tests — run these before trusting any number
```

### Status: what has actually been run

Verified end-to-end against live FinTabNet: corpus build (scan → rank → dedup →
download → manifest → leakage check), TEDS-Struct, span recovery, bootstrap CIs, and
the comparison renderer. All nine `src/` modules import without torch.

**Not verified: `src/model/inference.py` and `src/train/lora.py` have never been
executed** — there is no GPU on the dev Mac. They are written against documented
Unsloth/Transformers APIs, so treat the first GPU run as a debugging session and
budget 30–60 minutes for it.

### Three constraints that are easy to break by accident

- **Prompts are passed, not edited.** `predict`/`predict_many`/`TrainConfig` all take
  `instruction=`. Editing `src/model/prompts.py` mid-session does nothing — the module
  is already imported, so the old prompt is scored again and it reads as "prompt
  engineering had no effect."
- **The training prompt and the inference prompt must match.** `TrainConfig.instruction`
  exists for this. A mismatch discards most of what the adapter learned.
- **Both comparison arms must load the same way.** Unsloth's 4-bit path is not
  bit-identical to plain bitsandbytes, so a baseline loaded one way against a fine-tune
  loaded the other measures quantization as well as fine-tuning. `use_unsloth=True`
  everywhere.

One model in VRAM at a time. `TableReconstructor.close()` and `free_memory()` exist
because the failure mode is an OOM *after* a training run, not before it.

Environment: Python 3.11 via `uv` (`.venv/`). `requirements-base.txt` is CPU-only and
installs on the Mac; `requirements-gpu.txt` adds torch/Unsloth for the GPU box. **Every
heavy import in `src/` is deferred into function bodies**, so all nine modules import
without CUDA — data prep, evaluation, and rendering stay runnable locally while only
generation and training need a GPU. Preserve that when adding code.

Compute is **Lightning AI Studio**, not Colab: its persistent filesystem means a
disconnect does not cost the corpus and checkpoints. Colab free-tier T4 is the fallback
and requires checkpointing to Drive.

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

> Is Qwen3-VL accurate enough on hard financial tables — merged cells, nested headers,
> mixed bordered/borderless — to justify committing company resources, and does LoRA
> fine-tuning move it enough to be worth the pipeline?

Useful calibration for what to expect: Qwen2.5-VL-**32B** scores **81.7 TEDS zero-shot,
83.7 fine-tuned**. Do not benchmark a 4B LoRA run against 97% and call it a failure.

Note for the demo: a large share of any fine-tuning gain will come from **output-format
alignment** — the base model emits structurally sound HTML in a different dialect
(attribute ordering, `<th>` vs `<td>`, whitespace) that TEDS penalizes. That is a real
gain and exactly what domain fine-tuning buys, but it is not "the model learned to reason
about tables." Overselling that distinction costs credibility in month two.

---

# 4. Proposed Method

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