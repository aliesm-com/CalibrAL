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


def _cost_columns(
    config: dict[str, Any], bundle: DataBundle, al_result: dict[str, Any]
) -> dict[str, Any]:
    """USD accounting. Human cost uses ``cost.human_per_label_usd``.

    ``human_cost_usd`` covers labels in the training set; the gold validation
    split (identical for every method) is reported separately so it can be
    included or excluded in the paper.
    """
    cost_cfg = config.get("cost") or {}
    rate = cost_cfg.get("human_per_label_usd")
    usage = al_result.get("llm_usage") or {}
    human = al_result.get("human_labels_used")
    n_val = len(bundle.validation)
    llm_cost = usage.get("llm_cost_usd")
    out: dict[str, Any] = {
        "validation_gold_labels": n_val,
        "llm_live_calls": usage.get("llm_live_calls"),
        "llm_cache_hits": usage.get("llm_cache_hits"),
        "llm_errors": usage.get("llm_errors"),
        "llm_prompt_tokens": usage.get("llm_prompt_tokens"),
        "llm_completion_tokens": usage.get("llm_completion_tokens"),
        "llm_reasoning_tokens": usage.get("llm_reasoning_tokens"),
        "llm_cost_usd": llm_cost if usage else 0.0,
        "llm_cost_unknown_calls": usage.get("llm_cost_unknown_calls"),
        "human_cost_usd": None,
        "validation_cost_usd": None,
        "total_cost_usd": None,
    }
    if rate is not None and human is not None:
        rate = float(rate)
        out["human_cost_usd"] = round(float(human) * rate, 4)
        out["validation_cost_usd"] = round(n_val * rate, 4)
        out["total_cost_usd"] = round(
            out["human_cost_usd"] + float(out["llm_cost_usd"] or 0.0), 4
        )
    return out


def resolve_llm_model(config: dict[str, Any]) -> dict[str, Any]:
    """``llm.model: auto`` -> the probe winner in ``<bakeoff.output_dir>/best_llm.json``."""
    llm_cfg = config.get("llm") or {}
    if str(llm_cfg.get("model", "")).strip().lower() != "auto":
        return config
    from calibral.bakeoff import best_llm_path, configure_bakeoff

    configure_bakeoff(config)
    path = best_llm_path()
    if not path.exists():
        raise FileNotFoundError(
            f"llm.model=auto but {path} does not exist. Run the LLM probe first."
        )
    best = json.loads(path.read_text(encoding="utf-8")).get("best_model")
    if not best:
        raise ValueError(f"{path} has no best_model.")
    config = copy.deepcopy(config)
    config["llm"]["model"] = best
    get_logger("experiment").info("llm.model=auto resolved to %s (%s)", best, path)
    return config


# method -> (description, uses LLM)
METHODS: dict[str, tuple[str, bool]] = {
    "baseline": ("random AL, humans only", False),
    "proposed": ("uncertainty AL + calibrated LLM silver labels", True),
    "random_silver": ("random AL + the same LLM silver mechanism", True),
    "uncertainty_gold": ("uncertainty AL, humans only (no LLM)", False),
    "llm_distill": ("LLM labels only, no human training labels", True),
    "llm_zeroshot": ("LLM predicts the test set directly (no classifier)", True),
}


def _run_al_for_method(
    method: str,
    config: dict[str, Any],
    bundle: DataBundle,
    *,
    device: str,
) -> dict[str, Any]:
    from calibral.active_learning import (
        run_baseline_active_learning,
        run_llm_distill_training,
        run_llm_zeroshot_test,
        run_proposed_active_learning,
    )

    if method == "llm_zeroshot":
        return run_llm_zeroshot_test(config, bundle)
    tokenizer = build_tokenizer(str(config.get("model", {}).get("name")))
    if method == "baseline":
        return run_baseline_active_learning(config, bundle, tokenizer=tokenizer, device=device)
    if method == "proposed":
        return run_proposed_active_learning(config, bundle, tokenizer=tokenizer, device=device)
    if method == "random_silver":
        return run_proposed_active_learning(
            config, bundle, tokenizer=tokenizer, device=device, selection="random"
        )
    if method == "uncertainty_gold":
        return run_proposed_active_learning(
            config, bundle, tokenizer=tokenizer, device=device, use_llm=False
        )
    if method == "llm_distill":
        return run_llm_distill_training(config, bundle, tokenizer=tokenizer, device=device)
    raise ValueError(f"Unknown method: {method!r}. Available: {sorted(METHODS)}")


def run_method(
    method: str,
    config: dict[str, Any],
    bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Run one method for one seed and append a row to results.csv."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}. Available: {sorted(METHODS)}")
    logger = get_logger("experiment")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = detect_device()
    device_desc = describe_device()
    logger.info("%s start | device=%s | seed=%d", method, device_desc, seed)

    started = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_id = (
        f"{method}_{config.get('profile', 'run')}_seed{seed}_{stamp}_{uuid.uuid4().hex[:6]}"
    )
    output_dir = _output_dir(config)
    uses_llm = METHODS[method][1]
    llm_mode = str(config.get("llm", {}).get("mode", "unknown")) if uses_llm else "none"

    try:
        if bundle is None:
            bundle = prepare_data(config)
            print_data_summary(bundle)
        al_result = _run_al_for_method(method, config, bundle, device=device)
        return _evaluate_and_record(
            config=config,
            method=method,
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
        logger.exception("%s failed: %s", method, exc)
        results_path = _append_results_row(
            output_dir,
            _failed_row(
                run_id=run_id,
                method=method,
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
            "method": method,
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

    if al_result.get("predictions") is not None:  # e.g. llm_zeroshot
        predictions = al_result["predictions"]
    else:
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
        "dataset_name": (config.get("data") or {}).get("dataset_name", "imdb"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": al_result.get("n_train_final"),
        "gold_labels_total": al_result.get("gold_labels_total"),
        "silver_labels_total": al_result.get("silver_labels_total"),
        "silver_label_accuracy": al_result.get("silver_label_accuracy"),
        "rejected_llm_accuracy": al_result.get("rejected_llm_accuracy"),
        "llm_selective_coverage": al_result.get("llm_selective_coverage"),
        "llm_selective_accuracy": al_result.get("llm_selective_accuracy"),
        **_cost_columns(config, bundle, al_result),
    }

    records = al_result.get("llm_records") or []
    if records:
        pd.DataFrame(records).to_csv(
            output_dir / f"llm_labels_{method}_seed{seed}.csv", index=False
        )

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
        "llm_usage": al_result.get("llm_usage"),
        "silver_label_accuracy": al_result.get("silver_label_accuracy"),
        "rejected_llm_accuracy": al_result.get("rejected_llm_accuracy"),
        "cost": _cost_columns(config, bundle, al_result),
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
        "llm_cost_usd=%s | llm_mode=%s | runtime=%.2fs",
        method,
        metrics["macro_f1"],
        al_result["human_labels_used"],
        al_result["llm_calls"],
        result_row.get("llm_cost_usd"),
        llm_mode,
        runtime_seconds,
    )
    return summary


def run_baseline(
    config: dict[str, Any],
    bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Random active-learning baseline with the same human budget as proposed."""
    return run_method("baseline", config, bundle)


def run_proposed(
    config: dict[str, Any],
    bundle: DataBundle | None = None,
) -> dict[str, Any]:
    """Proposed hybrid active-learning method."""
    return run_method("proposed", resolve_llm_model(config), bundle)


def run_comparison(
    config: dict[str, Any],
    *,
    methods: tuple[str, ...] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run all methods on identical splits for one or more seeds.

    Methods come from the argument, else ``experiment.methods`` in the config,
    else (baseline, proposed).
    """
    logger = get_logger("experiment")
    config = resolve_llm_model(config)
    if methods is None:
        methods = tuple(
            (config.get("experiment") or {}).get("methods") or ("baseline", "proposed")
        )
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Available: {sorted(METHODS)}")
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
            run_summaries.append(run_method(method, seed_config, bundle=bundle))

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
                "cost": s.get("cost"),
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
    try:
        from calibral.significance import compute_significance

        compute_significance(output_dir)
        comparison["significance_path"] = str(output_dir / "significance.md")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Significance computation failed: %s", exc)

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
