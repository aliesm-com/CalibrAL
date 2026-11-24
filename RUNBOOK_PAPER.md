# CalibrAL — Paper experiment runbook

Commands only: [`RUNBOOK_COMMANDS.md`](RUNBOOK_COMMANDS.md).

Run the steps **in order**. Each step ends with a ✅ check: do not start the next step
until the check passes. Commands are for Windows PowerShell from the project root.

> Status legend: `[ ]` to do · `[x]` done. Tick boxes as you go.

## ⚡ Unattended mode (run everything overnight)

After Steps 0–1, one command runs Steps 2–5b for you:

```powershell
python scripts/run_paper_all.py --dry-run   # print what will run
python scripts/run_paper_all.py
```

- Probe once (skipped if `best_llm.json` exists) → final config → IMDB → SST-2 →
  Rotten Tomatoes → SnappFood → SnappFood/XLM-R → Amazon → Yelp, all 5 seeds, all methods,
  then report + figures per dataset and the encoder comparison.
- **Restart-safe:** stop it any time (Ctrl+C, power cut) and run it again; every
  (dataset, seed, method) already completed is skipped. A failed run is retried once.
- Watch progress in `outputs\PAPER_RUN_STATUS.md`; full log in `outputs\paper_run_logs\`.
- It does **not** stop at the Step 3 decision point — review the IMDB pilot afterwards.
- `--only imdb,sst2` limits datasets; `--skip-probe` never calls the probe.

---

## Step 0 — Environment check (5 min)

- [ ] Activate the venv and run the tests:
  ```powershell
  .venv\Scripts\Activate.ps1
  python -m pytest -q
  ```
- [ ] `.env` contains `OPENROUTER_API_KEY=...`
- [ ] Commit the code before experimenting (so results map to a code version):
  ```powershell
  git add -A; git commit -m "paper: ablations, cost tracking, probe report"
  ```

✅ **Check:** all tests pass (63 at the time of writing).

---

## Step 1 — Fix the protocol decisions (before spending GPU time)

Edit `configs/paper_al.yaml` **and** every file in `configs/paper/`:

- [ ] **Validation size.** `data.validation_size` is 2000 gold labels — more than the whole
      human budget (1500). Decide: keep 2000 (and report it), or reduce to e.g. 500.
      Changing it later means re-running everything.
- [ ] **Human cost rate.** Replace the placeholder `cost.human_per_label_usd: 0.10` with a
      rate you can cite. (Raw counts are stored, so this can still be recomputed later.)
- [ ] **Seeds:** keep 5 (`[42, 43, 44, 45, 46]`).

✅ **Check:** the same values appear in all paper configs.

---

## Step 2 — Choose the LLM once (probe on IMDB)

The LLM is chosen **once** on IMDB and then **frozen for every dataset**. Do not re-run the
probe per dataset to pick a model: that is dataset-specific tuning and favours models that
memorised that dataset. (Running the probe on other datasets *after* the choice, for analysis
only, is fine.)

- [ ] Run the probe (API calls only, no GPU):
  ```powershell
  python run.py bakeoff --stage llm-probe --config configs/paper_al.yaml
  ```
  The terminal prints each model's accuracy, tokens and cost, then the winner in a box.
  If it stops midway, run the same command again: finished models are skipped.
- [ ] Open the report of this run: `outputs\paper_bakeoff\PROBE_REPORT_<date_time>.md`
      (VS Code → `Ctrl+Shift+V`). The terminal prints its exact name.
      `PROBE_REPORT.md` is always the latest run; `best_llm.json` records which run
      (`probe_run`) the chosen model came from.
- [ ] Decide which model to use:
  - **Winner** (🏆) = most accurate.
  - **Best value** (💰) = cheapest model statistically tied with the winner — usually the
    better choice for a cost-focused paper.
- [ ] Write the final config:
  ```powershell
  python run.py bakeoff --stage write-final --config configs/paper_al.yaml
  ```
  This puts the 🏆 winner into `configs/paper_al_final.yaml`.
  **If you choose the 💰 model instead**, set `llm.model` to it explicitly in
  `configs/paper_al_final.yaml` **and** in every `configs/paper/*.yaml`
  (they say `auto`, which would otherwise read the 🏆 winner).

✅ **Check:** `configs/paper_al_final.yaml` → `llm.model` is the model you chose;
`PROBE_REPORT.md` shows no surprising failures; note the total probe cost.

> Regenerate the report any time without API calls:
> `python run.py bakeoff --stage probe-report --config configs/paper_al.yaml`
> Every probe run is archived in `outputs\paper_bakeoff\llm_probe\history\<date_time>\`.

---

## Step 3 — Pilot: IMDB, 1 seed, all 6 methods ⚠️ decision point

```powershell
python run.py experiment --config configs/paper_al_final.yaml --seeds 42
```

- [ ] Note the **wall-clock time** printed at the end (per seed). Estimate of the full plan:
      `time per seed × 5 seeds × number of datasets`.
- [ ] Open `outputs\paper_al_silver\significance.md` and compare:

| If… | Meaning | Paper story |
|---|---|---|
| `proposed` clearly beats `llm_distill` and `random_silver` | humans and uncertainty sampling both add value | original claim holds |
| `llm_distill` ≈ `proposed` | human labels add little | reframe: "a cheap LLM + calibration is enough" |
| `llm_zeroshot` beats everything | the classifier is only worth it for inference cost | make the inference-cost comparison explicit |

- [ ] Also look at `silver_label_accuracy` vs `rejected_llm_accuracy` in `results.csv`:
      if rejected answers are much less accurate, the confidence gate works.

✅ **Check:** all 6 methods `completed` and you know which story row applies.
**Stop here and discuss the result before continuing.**

---

## Step 4 — IMDB, remaining seeds

Seed 42 is already done; run only the other four (same output folder, results are merged):

```powershell
python run.py experiment --config configs/paper_al_final.yaml --seeds 43,44,45,46
python run.py report --config configs/paper_al_final.yaml
python run.py figures --config configs/paper_al_final.yaml
```

- [ ] `outputs\paper_al_silver\significance.md` — mean ± std, paired t-test, bootstrap CIs
- [ ] `outputs\paper_al_silver\figures\all_methods_macro_f1.png`, `cost_vs_f1.png`

✅ **Check:** 30 completed rows (6 methods × 5 seeds); `significance.md` lists 5 seeds per method.

> A run failed or was interrupted? Re-run just that part, e.g.
> `--seeds 44 --methods proposed`. Only the latest row per (method, seed) is used.

---

## Step 5 — Other datasets (same frozen LLM)

Fastest first. For each dataset: one seed to measure time, then the rest.

| Order | Config | Notes |
|---|---|---|
| 1 | `configs/paper/sst2.yaml` | short sentences, test = 872 |
| 2 | `configs/paper/rotten_tomatoes.yaml` | small dataset, smaller pool |
| 3 | `configs/paper/amazon_polarity.yaml` | product reviews, medium length |
| 4 | `configs/paper/snappfood.yaml` | **Persian** (SnappFood), short comments, ParsBERT encoder, test = 9,033 |
| 5 | `configs/paper/yelp_polarity.yaml` | long reviews, slowest |

```powershell
python run.py experiment --config configs/paper/sst2.yaml --seeds 42
python run.py experiment --config configs/paper/sst2.yaml --seeds 43,44,45,46
python run.py report --config configs/paper/sst2.yaml
```

- [ ] sst2 · [ ] rotten_tomatoes · [ ] amazon_polarity · [ ] snappfood · [ ] yelp_polarity

### Step 5b — Encoder ablation (Persian)

SnappFood uses ParsBERT; every other dataset uses roberta-base. To show the conclusions do
not depend on that choice, re-run SnappFood with the multilingual XLM-RoBERTa encoder
(same seeds → same splits; same frozen LLM → labels come from the cache, ~no API cost):

```powershell
python run.py experiment --config configs/paper/snappfood_xlmr.yaml --seeds 42
python run.py experiment --config configs/paper/snappfood_xlmr.yaml --seeds 43,44,45,46
python scripts/compare_runs.py outputs/paper_snappfood outputs/paper_snappfood_xlmr
```

- [ ] snappfood_xlmr done · [ ] comparison table written

✅ **Check:** in the comparison, the sign of Δ vs baseline (and the method ranking) is the
same for both encoders → "conclusions are robust to the encoder". If it flips, report it.

✅ **Check:** each `outputs\paper_<dataset>\significance.md` has 5 seeds per method.
The first run of a new dataset downloads it from Hugging Face (Amazon is large).

---

## Step 6 — Where the paper material is

**Paper 1 (method):**
- Tables: `outputs\<profile>\significance.md` (+ `.json`), dated copies in `history\`
- Figures: `docs\paper-figures\` and `outputs\<profile>\figures\`
- Raw: `results.csv` (costs, tokens, silver accuracy), `predictions_<method>_seed<seed>.csv`

**Paper 2 (cheap LLMs as annotators):**
- Model comparison: `outputs\paper_bakeoff\PROBE_REPORT.md` + `llm_probe\predictions\*.jsonl`
- Hard vs random samples: `llm_labels_<method>_seed<seed>.csv` — column `phase`
  (`al_round` = high-entropy picks, `silver_expansion` / `distill` = random picks),
  with confidence, correctness and acceptance per answer.

---

## Not implemented yet (decide after Step 3)

- [ ] Multi-class datasets (AG News) and a hard one (Yelp-5 / Banking77)
- [x] A Persian dataset — `snappfood` (Step 5). Note for the paper: the encoder differs
      (ParsBERT instead of roberta-base), so compare methods *within* the dataset, not raw
      F1 across datasets. Cite ParsBERT (Farahani et al., 2021).
- [ ] Paper 2 ablations: reasoning on vs off; confidence via logprobs / self-consistency;
      re-calibrating LLM confidence on a small gold set
- [ ] Optional: probe the frozen model on the other datasets (analysis only, not selection)

---

## Log

| Date | Step | Result / note |
|---|---|---|
| | | |
