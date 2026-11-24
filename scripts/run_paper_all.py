#!/usr/bin/env python3
"""Run the whole paper plan unattended: probe -> final config -> every dataset x seed.

    python scripts/run_paper_all.py            # run everything still missing
    python scripts/run_paper_all.py --dry-run  # only print the plan
    python scripts/run_paper_all.py --only imdb,sst2

Safe to stop and restart at any time (power cut, Ctrl+C, crash): every
(dataset, seed, method) already marked ``completed`` in its results.csv is
skipped, and the probe is skipped once ``best_llm.json`` exists.

Each (dataset, seed) runs in its own ``python run.py experiment`` process, so a
crash or GPU out-of-memory in one run cannot take down the others. A failed run
is retried once, then the script moves on and reports it at the end.

Progress:  outputs/PAPER_RUN_STATUS.md   (refreshed after every step)
Full log:  outputs/paper_run_logs/<date_time>.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # relative output paths in the configs are relative to the project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from calibral.config import load_config, resolve_output_dir  # noqa: E402

PROBE_CONFIG = "configs/paper_al.yaml"
FINAL_CONFIG = "configs/paper_al_final.yaml"

# (key, config) in run order: fastest first so partial results arrive early.
PLAN: list[tuple[str, str]] = [
    ("imdb", FINAL_CONFIG),
    ("sst2", "configs/paper/sst2.yaml"),
    ("rotten_tomatoes", "configs/paper/rotten_tomatoes.yaml"),
    ("snappfood", "configs/paper/snappfood.yaml"),
    ("snappfood_xlmr", "configs/paper/snappfood_xlmr.yaml"),
    ("amazon_polarity", "configs/paper/amazon_polarity.yaml"),
    ("yelp_polarity", "configs/paper/yelp_polarity.yaml"),
]
DEFAULT_METHODS = ["baseline", "proposed"]

STAMP = datetime.now().strftime("%Y-%m-%d_%H%M")
LOG_DIR = ROOT / "outputs" / "paper_run_logs"
LOG_PATH = LOG_DIR / f"{STAMP}.log"
STATUS_PATH = ROOT / "outputs" / "PAPER_RUN_STATUS.md"

status_rows: list[dict[str, str]] = []
started_all = time.time()
FATAL = {"hit": False, "line": ""}  # set when a child prints LLM_FATAL (no credit / bad key)
BUDGET = {"max": None}  # USD cap on real API spend, from --max-llm-usd


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")


def write_status(current: str = "") -> None:
    ok = sum(1 for r in status_rows if r["result"] == "✅ done")
    bad = sum(1 for r in status_rows if r["result"].startswith("❌"))
    lines = [
        "# Paper run status",
        "",
        f"- Started: {datetime.fromtimestamp(started_all).strftime('%Y-%m-%d %H:%M')}",
        f"- Updated: {now()}  ·  elapsed {fmt_dur(time.time() - started_all)}",
        f"- Now: {current or '—'}",
        f"- Steps done: {ok}  ·  failed: {bad}",
        f"- Log: `{LOG_PATH.relative_to(ROOT).as_posix()}`",
        "",
        "| Step | Result | Duration | Finished |",
        "|---|---|---:|---|",
    ]
    for r in status_rows:
        lines.append(f"| {r['step']} | {r['result']} | {r['duration']} | {r['finished']} |")
    lines.append("")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_cmd(step: str, args: list[str], *, retries: int = 1) -> bool:
    """Run ``python <args>`` as a child process, streaming output to console + log."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for attempt in range(1, retries + 2):
        write_status(step + (f" (retry {attempt - 1})" if attempt > 1 else ""))
        log(f"START {step} :: python {' '.join(args)}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        t0 = time.time()
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [sys.executable, *args],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
                if "LLM_FATAL" in line and not FATAL["hit"]:
                    FATAL["hit"], FATAL["line"] = True, line.strip()
            rc = proc.wait()
        dur = fmt_dur(time.time() - t0)
        if rc == 0:
            log(f"DONE  {step} in {dur}")
            status_rows.append({"step": step, "result": "✅ done", "duration": dur, "finished": now()})
            write_status()
            return True
        log(f"FAIL  {step} (exit {rc}) after {dur}")
        if FATAL["hit"]:
            break  # retrying cannot help without credit / a valid key
    status_rows.append({"step": step, "result": f"❌ exit {rc}", "duration": dur, "finished": now()})
    write_status()
    return False


def completed_methods(config: dict, seed: int) -> set[str]:
    """Methods already completed for this seed in the config's results.csv."""
    path = resolve_output_dir(dict(config), create=False) / "results.csv"
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return set()
    if df.empty or "status" not in df.columns:
        return set()
    done = df[(df["status"].astype(str) == "completed") & (df["seed"].astype(int) == int(seed))]
    return set(done["method"].astype(str))


def plan_for(config_path: str) -> list[tuple[int, list[str]]]:
    """[(seed, missing_methods)] for one config."""
    cfg = load_config(config_path)
    methods = list((cfg.get("experiment") or {}).get("methods") or DEFAULT_METHODS)
    seeds = [int(s) for s in (cfg.get("seeds") or [cfg.get("seed", 42)])]
    out = []
    for seed in seeds:
        missing = [m for m in methods if m not in completed_methods(cfg, seed)]
        if missing:
            out.append((seed, missing))
    return out


def spent_usd() -> float:
    """Real API money spent so far: probe + live calls of every paper run."""
    import json

    total = 0.0
    probe = ROOT / "outputs" / "paper_bakeoff" / "llm_probe" / "llm_probe_results.json"
    if probe.exists():
        try:
            total += float(json.loads(probe.read_text(encoding="utf-8")).get("total_cost_usd") or 0)
        except Exception:  # noqa: BLE001
            pass
    for _, cfg_path in PLAN:
        if not (ROOT / cfg_path).exists():
            continue
        path = resolve_output_dir(load_config(cfg_path), create=False) / "results.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "llm_live_cost_usd" in df.columns:
                    total += float(pd.to_numeric(df["llm_live_cost_usd"], errors="coerce").fillna(0).sum())
            except Exception:  # noqa: BLE001
                pass
    return total


def over_budget() -> bool:
    if BUDGET["max"] is None:
        return False
    spent = spent_usd()
    if spent >= BUDGET["max"]:
        log(f"BUDGET: ${spent:.2f} spent >= --max-llm-usd {BUDGET['max']:.2f}. Stopping before the next step.")
        return True
    log(f"budget: ${spent:.2f} of ${BUDGET['max']:.2f} spent")
    return False


def best_llm_exists() -> bool:
    from calibral.bakeoff import best_llm_path, configure_bakeoff

    configure_bakeoff(load_config(PROBE_CONFIG))
    return best_llm_path().exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--only", default=None, help="comma-separated dataset keys, e.g. imdb,sst2")
    ap.add_argument("--skip-probe", action="store_true", help="never run the LLM probe")
    ap.add_argument("--max-llm-usd", type=float, default=None,
                    help="stop before the next step once real API spend reaches this (USD)")
    args = ap.parse_args()
    BUDGET["max"] = args.max_llm_usd

    only = {k.strip() for k in args.only.split(",")} if args.only else None
    plan = [(k, c) for k, c in PLAN if only is None or k in only]

    log(f"=== Paper run {STAMP} | project {ROOT} ===")
    if args.max_llm_usd is not None:
        log(f"API budget cap: ${args.max_llm_usd:.2f} (spent so far: ${spent_usd():.2f})")

    # 1) LLM probe (once) and final config
    have_best = best_llm_exists()
    need_final = not (ROOT / FINAL_CONFIG).exists()
    if args.dry_run:
        print("\nPLAN")
        print(f"  probe:        {'skip (best_llm.json exists)' if have_best else 'RUN'}")
        print(f"  final config: {'write' if (need_final or not have_best) else 'exists'}")
        for key, cfg_path in plan:
            if cfg_path == FINAL_CONFIG and not (ROOT / FINAL_CONFIG).exists():
                print(f"  {key:16s} all seeds × all methods (final config not written yet)")
                continue
            items = plan_for(cfg_path)
            if not items:
                print(f"  {key:16s} complete")
            for seed, missing in items:
                print(f"  {key:16s} seed {seed}: {', '.join(missing)}")
        return 0

    if not have_best:
        if args.skip_probe:
            log("No best_llm.json and --skip-probe given: cannot continue.")
            return 1
        if not run_cmd("LLM probe", ["run.py", "bakeoff", "--stage", "llm-probe", "--config", PROBE_CONFIG]):
            log("Probe failed: every later step needs its result. Stopping.")
            return 1
    if need_final or not have_best:
        if not run_cmd("Write final config", ["run.py", "bakeoff", "--stage", "write-final", "--config", PROBE_CONFIG]):
            return 1
    else:
        log(f"Using existing {FINAL_CONFIG} (delete it to rewrite from best_llm.json).")

    # 2) every dataset x seed, only what is missing
    failed: list[str] = []
    for key, cfg_path in plan:
        items = plan_for(cfg_path)
        if not items:
            log(f"{key}: already complete, skipping")
            status_rows.append({"step": f"{key}", "result": "⏭️ already complete", "duration": "", "finished": now()})
            write_status()
        cfg = load_config(cfg_path)
        for seed, missing in items:
            if FATAL["hit"] or over_budget():
                break
            ok = False
            for attempt in range(2):  # one retry, re-checking what is still missing
                step = f"{key} · seed {seed} · {len(missing)} method(s)" + (" · retry" if attempt else "")
                ok = run_cmd(step, ["run.py", "experiment", "--config", cfg_path,
                                    "--seeds", str(seed), "--methods", ",".join(missing)], retries=0)
                done = completed_methods(cfg, seed)
                missing = [m for m in missing if m not in done]
                if ok or not missing:
                    ok = not missing
                    break
            if not ok:
                failed.append(f"{key} · seed {seed} · still missing: {', '.join(missing)}")
        if FATAL["hit"]:
            log(f"STOPPED: {FATAL['line']}")
            log("Fix the OpenRouter credit / key, then run this script again (finished work is kept).")
            write_status("STOPPED — out of API credit or invalid key (see log)")
            return 2
        if BUDGET["max"] is not None and spent_usd() >= BUDGET["max"]:
            write_status(f"STOPPED — budget ${BUDGET['max']:.2f} reached")
            return 3
        # report + figures for this dataset (cheap; uses whatever finished)
        run_cmd(f"{key} · report", ["run.py", "report", "--config", cfg_path], retries=0)
        run_cmd(f"{key} · figures", ["run.py", "figures", "--config", cfg_path], retries=0)

    # 3) encoder ablation comparison
    if only is None or {"snappfood", "snappfood_xlmr"} <= only:
        run_cmd("snappfood encoder comparison",
                ["scripts/compare_runs.py", "outputs/paper_snappfood", "outputs/paper_snappfood_xlmr"],
                retries=0)

    log(f"=== Finished in {fmt_dur(time.time() - started_all)} | failed steps: {len(failed)} ===")
    for f in failed:
        log(f"  FAILED: {f}  (re-run this script to retry)")
    write_status("finished" + (f" with {len(failed)} failed step(s)" if failed else " ✅"))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
