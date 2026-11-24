# CalibrAL — commands only

Run in order from the project root (PowerShell). Details and checks: `RUNBOOK_PAPER.md`.

**⚡ Everything in one command (unattended)** — runs steps 2–5b in order, skips whatever is
already done, safe to stop and restart. Progress: `outputs/PAPER_RUN_STATUS.md`.
```powershell
python scripts/run_paper_all.py --dry-run   # show the plan first
python scripts/run_paper_all.py
```

**0 · Setup**
```powershell
.venv\Scripts\Activate.ps1
python -m pytest -q
git add -A; git commit -m "paper: experiment setup"
```

**1 · Decisions** — edit `configs/paper_al.yaml` and `configs/paper/*.yaml` (validation size, human cost rate). No command.

**2 · Choose the LLM (once, IMDB)**
```powershell
python run.py bakeoff --stage llm-probe --config configs/paper_al.yaml
python run.py bakeoff --stage write-final --config configs/paper_al.yaml
```

**3 · Pilot: IMDB, 1 seed** ⚠️ stop and review `significance.md` after this
```powershell
python run.py experiment --config configs/paper_al_final.yaml --seeds 42
```

**4 · IMDB, remaining seeds**
```powershell
python run.py experiment --config configs/paper_al_final.yaml --seeds 43,44,45,46
python run.py report  --config configs/paper_al_final.yaml
python run.py figures --config configs/paper_al_final.yaml
```

**5 · Other datasets**
```powershell
python run.py experiment --config configs/paper/sst2.yaml --seeds 42
python run.py experiment --config configs/paper/sst2.yaml --seeds 43,44,45,46
python run.py report     --config configs/paper/sst2.yaml

python run.py experiment --config configs/paper/rotten_tomatoes.yaml --seeds 42
python run.py experiment --config configs/paper/rotten_tomatoes.yaml --seeds 43,44,45,46
python run.py report     --config configs/paper/rotten_tomatoes.yaml

python run.py experiment --config configs/paper/amazon_polarity.yaml --seeds 42
python run.py experiment --config configs/paper/amazon_polarity.yaml --seeds 43,44,45,46
python run.py report     --config configs/paper/amazon_polarity.yaml

python run.py experiment --config configs/paper/snappfood.yaml --seeds 42
python run.py experiment --config configs/paper/snappfood.yaml --seeds 43,44,45,46
python run.py report     --config configs/paper/snappfood.yaml

# 5b · Encoder ablation (Persian: XLM-R instead of ParsBERT)
python run.py experiment --config configs/paper/snappfood_xlmr.yaml --seeds 42
python run.py experiment --config configs/paper/snappfood_xlmr.yaml --seeds 43,44,45,46
python scripts/compare_runs.py outputs/paper_snappfood outputs/paper_snappfood_xlmr

python run.py experiment --config configs/paper/yelp_polarity.yaml --seeds 42
python run.py experiment --config configs/paper/yelp_polarity.yaml --seeds 43,44,45,46
python run.py report     --config configs/paper/yelp_polarity.yaml
```

**Utilities**
```powershell
# Rebuild the probe report (no API calls)
python run.py bakeoff --stage probe-report --config configs/paper_al.yaml

# Re-run only one failed piece (example)
python run.py experiment --config configs/paper_al_final.yaml --seeds 44 --methods proposed
```
