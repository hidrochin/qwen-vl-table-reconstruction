# RUNBOOK — Qwen3-VL Table Reconstruction Spike

Demo: **Wednesday 22 July 2026.** Build days: Sat 19 → Tue 21.

This is the execution order. `CLAUDE.md` explains *why* the design is what it is;
this file is *what to type, in what order, and what should come back*.

---

## 0. Before anything — read this part

**Two machines.** Not everything runs in both places, and confusing them wastes an
hour:

| Runs on the Mac | Needs the GPU box |
|---|---|
| `pytest` | zero-shot generation |
| corpus build (`build_corpus`) | prompt-variant generation |
| all scoring (`evaluate_predictions`, `compare_runs`) | LoRA training |
| demo rendering (`render_comparison`) | fine-tuned generation |
| rebuilding pages from saved runs | |

Every heavy import in `src/` is deferred into a function body specifically so the
left column works with no CUDA. Keep it that way when you add code.

**Two modules have never been executed.** `src/model/inference.py` and
`src/train/lora.py` are written against documented Unsloth/Transformers APIs but
there is no GPU on the dev Mac, so their first run is on your GPU box. Expect to
spend **30–60 minutes on signature mismatches the first time**. Budget that on
Saturday, not Monday night. Everything else in the repo has been run against real
FinTabNet data.

**GPU-hour budget.** The whole plan including one retry is roughly 8–9 GPU-hours
against Lightning's ~22/month free tier. You are not short on compute — do not
skip the 8B datapoint or a second training run to ration hours you have.

---

## 1. Saturday — Day 1

### 1.1 On the Mac: confirm the repo is sound (2 min)

```bash
cd "/Users/hydrochii/Documents/Coding Folder"
.venv/bin/python -m pytest tests/ -q
```

Expect `56 passed`. If TEDS tests fail, **stop** — every number downstream is
meaningless and you would not find out until the demo.

```bash
.venv/bin/python -c "
import src.data.loader, src.eval.runner, src.model.inference, src.train.lora, src.demo.render
print('imports clean without torch')"
```

### 1.2 Set up the GPU box (20 min, first time only)

On **Lightning AI Studio** — pick an L4 or A10G. Then, in a terminal:

```bash
git clone <your-repo-url> table-spike && cd table-spike   # or upload the folder
pip install -r requirements-base.txt
pip install -r requirements-gpu.txt
```

Verify before going further:

```bash
python -c "
import torch; from src.model.inference import gpu_report
print(gpu_report())
import unsloth; print('unsloth ok')"
```

You want a device name, VRAM, and `bf16=True`. If `bf16=False` you are on a T4 —
workable, but see §6.

> **Colab instead?** Same two installs, then **restart the runtime** before any
> other import. Unsloth pins its own torch and conflicts with Colab's
> preinstalled build. Also mount Drive and point `outputs/` at it, or a
> disconnect costs you the adapter.

### 1.3 Build the corpus (15–25 min)

Runs anywhere with a network — do it on the GPU box so the files land next to the
training run.

```python
from src.data.loader import build_corpus
counts = build_corpus(CORPUS, n_train=500, n_eval=100, max_scanned=20000)
```

Watch for these lines:

```
  scanned 20,000 -> 4,281 unique candidates (912 duplicates dropped) -> kept 500
  selected score range: 0.31 - 0.68
[train] 500 records, 42.3 MB images -> .../train/manifest.jsonl
leakage check passed: 500 train / 100 eval, 0 overlap
```

**Three things to actually check, not skim:**

1. `kept 500` — if it says fewer, raise `max_scanned` to 40000. A short corpus
   silently weakens the training run.
2. `leakage check passed` — if this raises, do not continue. Inflated numbers
   that a customer later disproves are worse than no numbers.
3. The **score range**. If the bottom of the range is near 0.12 (the corpus
   median) you are training on average tables, not hard ones, and demo goal #3
   evaporates. Raise `max_scanned`.

Then eyeball ten of them:

```python
from src.data.loader import load_manifest
eval_records = load_manifest(CORPUS / "eval")
for r in sorted(eval_records, key=lambda r: -r.complexity.score)[:10]:
    print(r.uid, f"{r.complexity.score:.3f}", f"spans={r.complexity.n_spanning}", r.image_path)
```

Open a few images. If tables labelled "hard" are not visibly hard, the filter is
wrong and everything built on it is wrong. **This check costs five minutes and is
the one most worth doing.**

### 1.4 Zero-shot baseline (25–45 min on L4)

```python
model = TableReconstructor(model_id=MODEL_4B, load_in_4bit=True, use_unsloth=True)
preds = model.predict_many([r.image_path for r in eval_records], mode="structure")
```

**Run it on 5 tables first.** The first execution of untested code is not the
moment to discover a signature error 40 minutes in:

```python
smoke = model.predict_many([r.image_path for r in eval_records[:5]], mode="structure")
print(smoke[0].raw[:400])
```

Read that raw output before trusting any score. You are checking that it is HTML
at all, that `clean_prediction` found a `<table>`, and that it respected
"no cell text" in structure mode. If it emits full text in structure mode, the
prompt is being ignored — fix that before spending 40 minutes.

Then the full run, score it, and save:

```python
results, summary = evaluate_predictions(eval_records, predictions, "zeroshot-4b")
save_run(results, summary, RUNS)
```

**Calibration for what you see.** Qwen2.5-VL-**32B** scores 81.7 TEDS zero-shot.
A 4B in 4-bit on deliberately hard tables landing anywhere in **0.55–0.75
TEDS-Struct is a normal result**, not a failure. Published 97% numbers are
specialized models trained on the target dataset. Do not read a 0.6 as "this
doesn't work."

`parse_failures` is the number to worry about. A handful is fine. Twenty out of
100 means the prompt or `max_new_tokens` is wrong, not the model.

### 1.5 The Day 1 gate (5 min)

```python
render_comparison(to_cases(results, limit=20, hardest_first=True),
                  DEMO / "day1_zeroshot.html", title="...")
```

Download that file and **open it with the network off**. It inlines images as
base64, so it should render fully offline.

> **Gate: you now have a demo-able artifact.** If everything from here fails, you
> can still stand up on Wednesday. Do not start Day 2 until this file exists and
> opens.

### 1.6 Optional: 8B ceiling (~1 GPU-hour)

Only if Day 1 finished early. `model.close()` first — two models on one card is
an OOM. *4B fine-tuned beating 8B zero-shot* is a genuinely good Wednesday
result, which is why this datapoint is worth having.

---

## 2. Sunday — Day 2: prompt iteration

Zero training cost, minutes per experiment. Many "we need fine-tuning" problems
are output-format problems, and finding that out today saves Monday.

**Define prompt variants as strings in the notebook.** Do not edit
`src/model/prompts.py` and re-run a cell — the module is already imported, Python
serves the cached version, and you will score the *same* prompt twice and
conclude prompt engineering does nothing.

```python
PROMPT_V2 = "Reconstruct this table's HTML structure. Work row by row ..."
probe = eval_records[:30]
probe_preds = model.predict_many([r.image_path for r in probe],
                                 mode="structure", instruction=PROMPT_V2)
```

**Iterate on 30 tables, confirm the winner on 100.** The ranking almost never
changes and each full pass costs 30 minutes.

Things worth trying, roughly in order of expected payoff:

1. Naming the failure you actually observed on Day 1 ("every row must account for
   the same number of columns" if column counts are drifting)
2. Tightening the no-cell-text instruction if structure mode leaks text
3. Row-by-row framing vs. holistic
4. A one-shot example in the prompt — expensive in tokens, sometimes decisive

Then **fix the winner in a variable**:

```python
BEST_PROMPT = PROMPT_V2   # or None to keep the default
```

Day 3 trains with this and generates with this. A mismatch between the training
prompt and the inference prompt throws away most of what the adapter learned —
`TrainConfig(instruction=...)` exists to prevent exactly that.

**Stop when `compare_runs` says NOT SIGNIFICANT twice in a row.** You have
reached the noise floor at n=100 and further prompt tweaking is measuring nothing.

If a prompt change moves TEDS-Struct by 10+ points, that is worth saying out loud
on Wednesday — it means the base model's structural understanding was largely
there and the gap was format. Useful, honest, and it de-risks the production
phase.

---

## 3. Monday — Day 3: LoRA fine-tune

### 3.1 Free VRAM first (30 seconds, prevents a 40-minute loss)

```python
model.close()
print(gpu_report())
```

Used VRAM should drop back near zero. **If it does not, restart the kernel.**
Starting training with an inference model still resident OOMs partway through the
first epoch — long after it looked healthy.

### 3.2 Start training (~1.5–2h for 500 samples × 2 epochs on L4)

```python
cfg = TrainConfig(model_id=MODEL_4B,
                  output_dir=str(ROOT / "outputs" / "lora-4b-structure"),
                  mode="structure", instruction=BEST_PROMPT,
                  epochs=2, max_seq_length=4096)
ft_model, ft_processor, trainer = train(train_records, cfg, resume=True)
```

**Verify resume works before you walk away.** This is the single highest-value
five minutes of Monday:

1. Let it reach step 50 and write `checkpoint-50`
2. Interrupt the kernel
3. Re-run the same cell
4. Confirm it prints `resuming from .../checkpoint-50`

A run that cannot resume is a run that starts over, and on a four-day clock that
can cost the experiment.

**What healthy looks like:** loss falling fast for the first ~30 steps (that is
mostly format alignment), then a slower decline. Flat loss from step 1 means the
learning rate or the data collator is wrong — kill it, don't wait it out.

Watch the `length filter dropped N/500` line. Dropping 10–20 is fine. Dropping
150 means `max_seq_length` is too tight for this corpus and you are training on
the easy half of it.

### 3.3 Evaluate the adapter (25–45 min)

```python
del ft_model, ft_processor, trainer
free_memory()

tuned = TableReconstructor(model_id=MODEL_4B, load_in_4bit=True, use_unsloth=True,
                           adapter_path=str(ROOT / "outputs" / "lora-4b-structure" / "adapter"))
preds_ft = tuned.predict_many([r.image_path for r in eval_records],
                              mode="structure", instruction=BEST_PROMPT)
print(compare_runs(summary, summary_ft))
```

Reloading from disk rather than reusing the in-memory model is deliberate: it
proves the adapter actually persisted, which is what Wednesday depends on.

**`use_unsloth=True` in both arms.** Unsloth's 4-bit path is not bit-identical to
plain bitsandbytes; load the baseline one way and the fine-tune the other and
part of your "gain" is a quantization artifact. Same loader both sides, always.

### 3.4 If you have time for run 2

Vary **one** thing. Highest expected value first:

1. `lora_rank=32` — most likely to help if run 1 underfit
2. `epochs=3` — if loss was still falling at the end
3. `max_pixels=1280*28*28` — if failures look like missed fine rules rather than
   confused hierarchy

---

## 4. Tuesday — Day 4: assemble and dry-run

### 4.1 The comparison table

```python
results, summary = load_run(RUNS / "zeroshot-4b.json")
results_ft, summary_ft = load_run(RUNS / "lora-4b.json")
print(compare_runs(summary, summary_ft))
```

This loads from JSON and **needs no GPU** — it runs on the Mac. That is
deliberate: nothing should require a GPU on demo day.

**If it says NOT SIGNIFICANT, believe it.** At n=100 a 2–3 point TEDS gap is
noise. You have two honest moves: report it as "directionally positive, not yet
separable at this sample size", or lead with the per-bin breakdown if the hard
bin moved clearly even though the average did not. What you cannot do is present
the delta as a result — that claim will not survive the production evaluation,
and it will be re-run in month two.

### 4.2 The visual page

```python
demo_preds = tuned.predict_many([r.image_path for r in demo_records],
                                mode="full", instruction=None)
```

`mode="full"` so the rendered tables carry text and look real. **Label clearly
that the metric is structure-only** — a customer who thinks 0.72 includes OCR
will draw the wrong conclusion in both directions.

Include **2–3 honest failure cases**. A flawless reel invites "what did you leave
out?", and the failures set up the production-phase ask.

### 4.3 Dry run — do not skip this

1. Restart the kernel. Run the notebook top to bottom.
2. Download every artifact you plan to show.
3. **Turn the network off and open each one.**
4. Time the walkthrough.

Colab/Lightning state drift is the classic Wednesday-morning failure. So is
demoing live inference on a free-tier GPU — pre-render everything.

---

## 5. What to say on Wednesday

Three claims the artifacts support, and one caveat that protects you:

- **Feasibility.** "Qwen3-VL reconstructs table structure on hard financial
  tables at X TEDS-Struct; here it is side by side." Goal #1.
- **Fine-tuning moves it.** Only if the intervals separate. Goal #2.
- **Hard cases specifically.** The per-bin table, plus span recovery — "recovered
  47 of 52 merged cells" is a number customers have intuition for, unlike 0.91
  TEDS-Struct. Goal #3.

**The caveat, said by you before it is asked:** a large share of the fine-tuning
gain is **output-format alignment** — the base model emits structurally sound
HTML in a different dialect (attribute order, `<th>` vs `<td>`, whitespace) that
TEDS penalizes. That is a real gain and exactly what domain fine-tuning buys. It
is *not* "the model learned to reason about tables." Saying so costs nothing on
Wednesday and buys you enormous credibility in month two, when the production
numbers come in lower than the demo.

Also worth stating plainly: this is public financial data, ~600 tables, one 4B
model, four days. It is a feasibility signal, not a production benchmark.

---

## 6. When things break

| Symptom | Cause | Fix |
|---|---|---|
| OOM during training | inference model still resident | `model.close()`, or restart kernel |
| OOM at ~step 40 | one pathological table | lower `max_seq_length` or `max_pixels`; check the length-filter line |
| Corpus build stalls | rate limiting | it retries with backoff; if it persists lower `workers` to 4 |
| `kept N` < target | filter too strict | raise `max_scanned`, or `require_spanning=False` |
| Prompt edits change nothing | module already imported | pass `instruction=` instead of editing `prompts.py` |
| TEDS ≈ 0 everywhere | `clean_prediction` found no `<table>` | print `preds[0].raw` and look at it |
| Every score suspiciously high | leakage | re-run `assert_no_leakage` |
| Adapter won't load | trained under Unsloth, loaded without | `use_unsloth=True` |
| `bf16=False` | T4 | works, but fp16; keep `max_seq_length` at 4096 and expect slower steps |
| Runtime disconnected | free tier | re-run the train cell; `resume=True` picks up the checkpoint |

**Fallback ladder.** If Monday's fine-tune fails outright, demo Day 1–2 output:
zero-shot 4B vs 8B, side-by-sides, accuracy by difficulty bin. That still answers
the feasibility question and still shows hard-case competence. You lose only goal
#2. **Never let goal #2 put goals #1 and #3 at risk.**

---

## 7. After the demo

The production phase is a different problem and should not start by fine-tuning.
With ~20 unlabeled invoices and no eval set there is no way to tell whether a
fine-tune helped — the bottleneck is data, not the model. See the memory note
`invoice-extraction-production-context` for the order that follows.
