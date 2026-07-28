"""Experiment orchestration for baseline, proposed, and comparative runs."""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from calibral.data import DataBundle, prepare_data, print_data_summary
from calibral.evaluation import compute_metrics, predict
from calibral.model import build_tokenizer
from calibral.config import load_config, resolve_output_dir
from calibral.utils import describe_device, detect_device, get_logger, set_seed


def _output_dir(config: dict[str, Any]) -> Path:
    return resolve_output_dir(config, create=True)


def _n_al_rounds(al_result: dict[str, Any]) -> int:
    rounds = al_result.get("rounds") or []
    return int(len(rounds)) if isinstance(rounds, list) else 0


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _is_smoke_profile(config: dict[str, Any]) -> bool:
    if config.get("profile") == "smoke":
        return True
    notes = str(config.get("notes", ""))
    return "NOT_SCIENTIFIC_SMOKE_RESULT" in notes


def _append_results_row(output_dir: Path, row: dict[str, Any]) -> Path:
    results_path = output_dir / "results.csv"
    result_df = pd.DataFrame([row])
    if results_path.exists():
        previous = pd.read_csv(results_path)
        result_df = pd.concat([previous, result_df], ignore_index=True)
    result_df.to_csv(results_path, index=False)
    return results_path


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    run_id = str(summary.get("run_id") or "unknown")
    archive = output_dir / "run_summaries"
    archive.mkdir(parents=True, exist_ok=True)
    archived = archive / f"{run_id}.json"
    with archived.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary_path


def _failed_row(
    *,
    run_id: str,
    method: str,
    seed: int,
    profile: Any,
    llm_mode: str,
    runtime_seconds: float,
    model_name: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "profile": profile,
        "model_name": model_name,
        "llm_model": llm_model,
        "macro_f1": None,
        "accuracy": None,
        "human_labels_used": None,
        "llm_labels_accepted": None,
        "llm_calls": None,
        "runtime_seconds": round(runtime_seconds, 3),
        "llm_mode": llm_mode,
        "status": "failed",
    }


def _evaluate_and_record(
    *,
    config: dict[str, Any],
    method: str,
    run_id: str,
    seed: int,
    device_desc: str,
    bundle: DataBundle,
    al_result: dict[str, Any],
    started: float,
    output_dir: Path,
) -> dict[str, Any]:
    logger = get_logger("experiment")
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    llm_mode = al_result["llm_mode"]

    predictions = predict(
        al_result["model"],
        al_result["tokenizer"],
        bundle.test,
        device=detect_device(),
        max_length=int(model_cfg.get("max_length", 128)),
        batch_size=int(train_cfg.get("batch_size", 8)),
    )
    metrics = compute_metrics(
        predictions["true_label"].tolist(),
        predictions["pred_label"].tolist(),
    )
    runtime_seconds = time.perf_counter() - started
    status = "completed"
    epochs = int(train_cfg.get("epochs", 1))
    n_al_rounds = _n_al_rounds(al_result)

    result_row = {
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "profile": config.get("profile"),
        "model_name": model_cfg.get("name"),
        "llm_model": config.get("llm", {}).get("model"),
        "llm_shot_mode": config.get("llm", {}).get("shot_mode", "zero"),
        "epochs": epochs,
        "n_al_rounds": n_al_rounds,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "human_labels_used": al_result["human_labels_used"],
        "llm_labels_accepted": al_result["llm_labels_accepted"],
        "llm_calls": al_result["llm_calls"],
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_human": _format_duration(runtime_seconds),
        "llm_mode": llm_mode,
        "status": status,
    }

    pred_name = f"predictions_{method}_seed{seed}.csv"
    pred_path = output_dir / pred_name
    predictions.to_csv(pred_path, index=False)
    # Also refresh the canonical predictions.csv with the latest run.
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    results_path = _append_results_row(output_dir, result_row)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "project": "CalibrAL",
        "method": method,
        "seed": seed,
        "profile": config.get("profile"),
        "device": device_desc,
        "model_name": model_cfg.get("name"),
        "llm_mode": llm_mode,
        "status": status,
        "scientific": not _is_smoke_profile(config),
        "notes": config.get("notes"),
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "human_labels_used": al_result["human_labels_used"],
        "human_oracle_used": al_result.get("human_oracle_used"),
        "gold_labels_total": al_result.get("gold_labels_total"),
        "llm_labels_accepted": al_result["llm_labels_accepted"],
        "silver_labels_total": al_result.get("silver_labels_total"),
        "llm_calls": al_result["llm_calls"],
        "epochs": epochs,
        "n_al_rounds": n_al_rounds,
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "learning_rate": train_cfg.get("learning_rate"),
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_human": _format_duration(runtime_seconds),
        "n_train": al_result["n_train_final"],
        "n_test": len(bundle.test),
        "rounds": al_result.get("rounds"),
        "initial_train": al_result.get("initial_train"),
        "predictions_path": str(pred_path),
        "results_path": str(results_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if _is_smoke_profile(config):
        summary["result_tag"] = "NOT_SCIENTIFIC_SMOKE_RESULT"
        logger.warning(
            "NOT_SCIENTIFIC_SMOKE_RESULT — smoke metrics are for pipeline "
            "testing only and must not be treated as thesis results."
        )

    summary_path = _write_summary(output_dir, summary)
    summary["summary_path"] = str(summary_path)
    logger.info(
        "%s done | macro_f1=%.4f | human_labels_used=%d | llm_calls=%d | "
        "llm_mode=%s | runtime=%.2fs",
        method,
        metrics["macro_f1"],
        al_result["human_labels_used"],
        al_result["llm_calls"],
        llm_mode,
        runtime_seconds,
    )
    return summary


def run_baseline(
    config: dict[str, Any],
    bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Random active-learning baseline with the same human budget as proposed."""
    from calibral.active_learning import run_baseline_active_learning

    logger = get_logger("experiment")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = detect_device()
    device_desc = describe_device()
    logger.info("Baseline start | device=%s | seed=%d", device_desc, seed)

    started = time.perf_counter()
    run_id = f"baseline_{config.get('profile', 'run')}_seed{seed}_{uuid.uuid4().hex[:8]}"
    output_dir = _output_dir(config)
    llm_mode = "none"

    try:
        if bundle is None:
            bundle = prepare_data(config)
            print_data_summary(bundle)
        tokenizer = build_tokenizer(str(config.get("model", {}).get("name")))
        al_result = run_baseline_active_learning(
            config,
            bundle,
            tokenizer=tokenizer,
            device=device,
        )
        return _evaluate_and_record(
            config=config,
            method="baseline",
            run_id=run_id,
            seed=seed,
            device_desc=device_desc,
            bundle=bundle,
            al_result=al_result,
            started=started,
            output_dir=output_dir,
        )
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        logger.exception("Baseline failed: %s", exc)
        results_path = _append_results_row(
            output_dir,
            _failed_row(
                run_id=run_id,
                method="baseline",
                seed=seed,
                profile=config.get("profile"),
                llm_mode=llm_mode,
                runtime_seconds=runtime_seconds,
                model_name=str(config.get("model", {}).get("name")),
                llm_model=str(config.get("llm", {}).get("model")),
            ),
        )
        summary = {
            "run_id": run_id,
            "method": "baseline",
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "llm_mode": llm_mode,
            "runtime_seconds": round(runtime_seconds, 3),
            "results_path": str(results_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if _is_smoke_profile(config):
            summary["result_tag"] = "NOT_SCIENTIFIC_SMOKE_RESULT"
        _write_summary(output_dir, summary)
        summary["summary_path"] = str(output_dir / "run_summary.json")
        return summary


def run_proposed(
    config: dict[str, Any],
    bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Proposed hybrid active-learning method."""
    from calibral.active_learning import run_proposed_active_learning

    logger = get_logger("experiment")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = detect_device()
    device_desc = describe_device()
    logger.info("Proposed start | device=%s | seed=%d", device_desc, seed)

    started = time.perf_counter()
    run_id = f"proposed_{config.get('profile', 'run')}_seed{seed}_{uuid.uuid4().hex[:8]}"
    output_dir = _output_dir(config)
    llm_mode = str(config.get("llm", {}).get("mode", "unknown"))

    try:
        if bundle is None:
            bundle = prepare_data(config)
            print_data_summary(bundle)
        tokenizer = build_tokenizer(str(config.get("model", {}).get("name")))
        al_result = run_proposed_active_learning(
            config,
            bundle,
            tokenizer=tokenizer,
            device=device,
        )
        return _evaluate_and_record(
            config=config,
            method="proposed",
            run_id=run_id,
            seed=seed,
            device_desc=device_desc,
            bundle=bundle,
            al_result=al_result,
            started=started,
            output_dir=output_dir,
        )
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        logger.exception("Proposed failed: %s", exc)
        results_path = _append_results_row(
            output_dir,
            _failed_row(
                run_id=run_id,
                method="proposed",
                seed=seed,
                profile=config.get("profile"),
                llm_mode=llm_mode,
                runtime_seconds=runtime_seconds,
                model_name=str(config.get("model", {}).get("name")),
                llm_model=str(config.get("llm", {}).get("model")),
            ),
        )
        summary = {
            "run_id": run_id,
            "method": "proposed",
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "llm_mode": llm_mode,
            "runtime_seconds": round(runtime_seconds, 3),
            "results_path": str(results_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if _is_smoke_profile(config):
            summary["result_tag"] = "NOT_SCIENTIFIC_SMOKE_RESULT"
        _write_summary(output_dir, summary)
        summary["summary_path"] = str(output_dir / "run_summary.json")
        return summary


def run_comparison(
    config: dict[str, Any],
    *,
    methods: tuple[str, ...] = ("baseline", "proposed"),
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run baseline and proposed on identical splits for one or more seeds."""
    logger = get_logger("experiment")
    if seeds is None:
        seeds = list(config.get("seeds") or [config.get("seed", 42)])
    seeds = [int(s) for s in seeds]

    output_dir = _output_dir(config)
    started = time.perf_counter()
    run_summaries: list[dict[str, Any]] = []

    logger.info(
        "Comparison start | profile=%s | seeds=%s | methods=%s | device=%s",
        config.get("profile"),
        seeds,
        list(methods),
        describe_device(),
    )

    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config["seed"] = seed
        set_seed(seed)
        bundle = prepare_data(seed_config)
        print_data_summary(bundle)

        for method in methods:
            if method == "baseline":
                summary = run_baseline(seed_config, bundle=bundle)
            elif method == "proposed":
                summary = run_proposed(seed_config, bundle=bundle)
            else:
                raise ValueError(f"Unknown method: {method}")
            run_summaries.append(summary)

    comparison = {
        "project": "CalibrAL",
        "profile": config.get("profile"),
        "seeds": seeds,
        "methods": list(methods),
        "device": describe_device(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "results_path": str(output_dir / "results.csv"),
        "runs": [
            {
                "run_id": s.get("run_id"),
                "method": s.get("method"),
                "seed": s.get("seed"),
                "status": s.get("status"),
                "model_name": s.get("model_name"),
                "macro_f1": s.get("macro_f1"),
                "accuracy": s.get("accuracy"),
                "human_labels_used": s.get("human_labels_used"),
                "llm_labels_accepted": s.get("llm_labels_accepted"),
                "llm_calls": s.get("llm_calls"),
                "llm_mode": s.get("llm_mode"),
                "result_tag": s.get("result_tag"),
            }
            for s in run_summaries
        ],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if _is_smoke_profile(config):
        comparison["result_tag"] = "NOT_SCIENTIFIC_SMOKE_RESULT"
        comparison["note"] = (
            "Smoke comparison uses tiny model + mock LLM and is not a scientific result. "
            "Mock and API modes remain distinguishable via llm_mode in results.csv."
        )

    comparison["runtime_human"] = _format_duration(comparison["runtime_seconds"])
    comparison["epochs"] = int(config.get("train", {}).get("epochs", 1))
    comparison["batch_size"] = int(config.get("train", {}).get("batch_size", 8))
    comparison["output_dir"] = str(output_dir)

    comparison_path = output_dir / "comparison_summary.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
    comparison["comparison_path"] = str(comparison_path)

    from calibral.reporting import write_experiment_summary

    summary_md = write_experiment_summary(config, output_dir=output_dir)
    comparison["experiment_summary_path"] = str(summary_md)

    n_failed = sum(1 for s in run_summaries if s.get("status") == "failed")
    n_ok = sum(1 for s in run_summaries if s.get("status") == "completed")
    logger.info(
        "Comparison done | completed=%d | failed=%d | results=%s | summary=%s",
        n_ok,
        n_failed,
        comparison["results_path"],
        summary_md,
    )
    comparison["completed"] = n_ok
    comparison["failed"] = n_failed
    return comparison


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Entry point for ``python run.py experiment`` (multi-seed comparison)."""
    return run_comparison(config)


def run_sweep(config_paths: list[str | Path]) -> dict[str, Any]:
    """Run experiment for each YAML config and append all rows to results.csv."""
    logger = get_logger("experiment")
    started = time.perf_counter()
    items: list[dict[str, Any]] = []

    paths = [Path(p) for p in config_paths]
    if not paths:
        raise ValueError("No config files provided for sweep.")

    logger.info("Sweep start | n_configs=%d | device=%s", len(paths), describe_device())
    for path in paths:
        logger.info("Sweep item: %s", path)
        config = load_config(path)
        comparison = run_comparison(config)
        items.append(
            {
                "config_path": str(path),
                "profile": config.get("profile"),
                "model_name": config.get("model", {}).get("name"),
                "llm_model": config.get("llm", {}).get("model"),
                "completed": comparison.get("completed"),
                "failed": comparison.get("failed"),
                "runs": comparison.get("runs"),
            }
        )

    output_dir = _output_dir(load_config(paths[0]))
    sweep_summary = {
        "project": "CalibrAL",
        "n_configs": len(paths),
        "device": describe_device(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "results_path": str(output_dir / "results.csv"),
        "items": items,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    sweep_path = output_dir / "sweep_summary.json"
    with sweep_path.open("w", encoding="utf-8") as handle:
        json.dump(sweep_summary, handle, indent=2, ensure_ascii=False)
    sweep_summary["sweep_path"] = str(sweep_path)
    logger.info("Sweep done | summary=%s", sweep_path)
    return sweep_summary
