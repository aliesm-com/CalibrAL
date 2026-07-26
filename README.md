# CalibrAL

Minimal research prototype for comparing a **baseline** with a lightweight **hybrid active-learning** method on binary IMDB sentiment.

- Project name: **CalibrAL**
- Internal Python package: `calibral`

This is a research codebase for local machines and Google Colab — not a production platform.

## Status

Milestone **M8** (final checks + delivery) is complete.

## Delivery checklist

- [x] `python -m pytest -q`
- [x] `python run.py smoke`
- [x] `python run.py all`
- [x] README Local + Colab paths
- [x] No committed API secrets
- [x] Test split unused for training / early stopping
- [x] Plots/report generated from `outputs/results.csv`
- [x] Mock/smoke results tagged non-scientific

## Local setup

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

### GPU (optional, recommended on NVIDIA)

If `pip` installed a CPU-only torch wheel, install a CUDA build:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

CalibrAL selects `cuda` when available and falls back to `cpu` otherwise.

Optional API key for scientific LLM mode:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Set `OPENAI_API_KEY` in `.env` / environment. Never commit secrets.

## Local commands

```bash
python run.py --help
python run.py smoke
python run.py experiment
python run.py report
python run.py all
```

Config profiles:

```bash
python run.py smoke --config configs/smoke.yaml
python run.py experiment --config configs/default.yaml
python run.py all --config configs/colab.yaml
```

## Thesis protocol (IMDB ~50k)

Important distinction for the thesis claim:

- IMDB has ~25k train + ~25k test.
- **You evaluate on the full test set** (`test_size: 25000`).
- Active Learning does **not** label all 50k; it labels a limited budget from a large pool.
- To first prove **>90% is achievable** with this stack, run the supervised upper bound:

```bash
# needs GPU recommended; no LLM calls when budget=0
python run.py experiment --config configs/thesis_upperbound.yaml
python run.py report --config configs/thesis_upperbound.yaml
```

Then run the AL comparison (LLM key required):

```bash
# set OPENROUTER_API_KEY in .env
python run.py experiment --config configs/thesis_al.yaml
python run.py report --config configs/thesis_al.yaml
```

Suggested thesis narrative:

1. Upper bound on full IMDB test shows DistilBERT can exceed ~90%.
2. Proposed AL aims to approach that performance with fewer human labels than random baseline.
3. Smoke/mock rows are software tests only and must not be cited as thesis results.

## Model / LLM sweeps

Classifier models:

```bash
python run.py experiment --config configs/models/distilbert.yaml
python run.py experiment --config configs/models/bert_base.yaml
python run.py experiment --config configs/models/roberta_base.yaml
python run.py experiment --config configs/models/electra_small.yaml
```

Or all classifier configs in one go:

```bash
# set OPENROUTER_API_KEY first
python run.py sweep --config-dir configs/models
python run.py report
```

LLM backends (same DistilBERT classifier):

```bash
python run.py sweep --config-dir configs/llm
```

Results include `model_name` and `llm_model` columns in each experiment's `results.csv`.

## Tests

```bash
python -m pytest -q
```

## Expected outputs

Each config writes into its own folder under `outputs/<profile>/` (shared cache: `outputs/cache/`):

```text
outputs/smoke/                  # smoke profile
outputs/thesis_al/
outputs/thesis_upperbound/
outputs/llm/test_llm/
...
  results.csv
  predictions.csv
  run_summary.json
  comparison_summary.json
  experiment_summary.md         # English summary: epochs, runtime, metrics
  learning_curve.png
  method_comparison.png
  report.md
  llm_smoke.json                # after smoke --stage llm
outputs/cache/                  # shared dataset + split cache
```

Smoke / mock LLM results are **not** scientific thesis results.

## Google Colab

Repository: https://github.com/aliesm-com/CalibrAL

Use `notebooks/colab.ipynb`, or run:

```python
!git clone https://github.com/aliesm-com/CalibrAL.git
%cd CalibrAL
!pip install -r requirements.txt -q
!python run.py smoke
!python run.py all --config configs/colab.yaml
```

Notes for Colab:

- Colab usually provides a GPU; torch from `requirements.txt` may still be CPU unless you install a CUDA wheel.
- `configs/colab.yaml` writes under `/content/outputs`.
- `llm.mode: api` needs `OPENAI_API_KEY`. Without a key, scientific runs refuse silent mock fallback.
- For a software-only path test, prefer `python run.py smoke` (mock LLM + tiny model).

## Layout

```text
run.py
requirements.txt
configs/               # smoke / default / colab
calibral/             # package logic
tests/
notebooks/colab.ipynb  # Colab runner only (no training logic)
outputs/               # generated artifacts
```

## Important limitations

- Smoke uses a tiny model and mock LLM → `NOT_SCIENTIFIC_SMOKE_RESULT`.
- Scientific claims need real model (`default`/`colab`), real or replayed LLM answers, and multiple seeds.
- Do not mix mock rows with API rows when interpreting thesis results (`llm_mode` column distinguishes them).
