"""Thesis bakeoff: one-call LLM probe + encoder sweeps from thesis_al."""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from calibral.config import load_config, resolve_output_dir
from calibral.data import LABEL_NEGATIVE, LABEL_POSITIVE
from calibral.experiment import run_comparison
from calibral.llm_labeler import (
    APILLMLabeler,
    MissingAPIKeyError,
    _read_api_key,
)
from calibral.reporting import build_report, format_duration
from calibral.utils import PROJECT_ROOT, describe_device, get_logger, load_dotenv

_LLM_TO_INT = {"negative": LABEL_NEGATIVE, "positive": LABEL_POSITIVE}

# LLM candidates for selection. Each model is evaluated once (not re-visited).
LLM_CANDIDATES: list[str] = [
    "openai/gpt-4o-mini",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.5-9b",
    "mistralai/ministral-3b-2512",
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o",
    "x-ai/grok-4.3",
]

# BERT-family encoders for the thesis_al comparison sweep.
ENCODER_MODELS: list[dict[str, Any]] = [
    {
        "slug": "bert_base",
        "name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 4,
    },
    {
        "slug": "distilbert",
        "name": "distilbert-base-uncased",
        "max_length": 512,
        "batch_size": 8,
    },
    {
        "slug": "roberta_base",
        "name": "roberta-base",
        "max_length": 512,
        "batch_size": 4,
    },
    {
        "slug": "electra_base",
        "name": "google/electra-base-discriminator",
        "max_length": 512,
        "batch_size": 4,
    },
    {
        "slug": "albert_base",
        "name": "albert-base-v2",
        "max_length": 512,
        "batch_size": 4,
    },
]

# Default gold-labeled probe size for LLM selection (balanced pos/neg).
DEFAULT_PROBE_SAMPLES = 20

# Fallback single example if data loading fails (not preferred).
PROBE_TEXT = (
    "This movie was an absolute masterpiece. The acting, story, and direction "
    "were outstanding from start to finish. I would watch it again tomorrow."
)
PROBE_GOLD_LABEL = LABEL_POSITIVE
PROBE_GOLD_NAME = "positive"


def bakeoff_root() -> Path:
    path = PROJECT_ROOT / "outputs" / "bakeoff"
    path.mkdir(parents=True, exist_ok=True)
    return path


def best_llm_path() -> Path:
    return bakeoff_root() / "best_llm.json"


def encoder_config_dir() -> Path:
    path = PROJECT_ROOT / "configs" / "sweeps" / "thesis_encoders"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_encoder_configs(base_config_path: str | Path) -> list[Path]:
    """Materialize per-encoder YAMLs derived from thesis_al."""
    base = load_config(base_config_path)
    out_dir = encoder_config_dir()
    written: list[Path] = []

    best = None
    if best_llm_path().exists():
        best = json.loads(best_llm_path().read_text(encoding="utf-8"))

    for enc in ENCODER_MODELS:
        cfg = copy.deepcopy(base)
        cfg["profile"] = f"thesis_al__{enc['slug']}"
        # Bakeoff default: one seed to control cost; edit YAML for full seeds.
        cfg["seeds"] = [int(cfg.get("seed", 42))]
        cfg["data"]["max_text_length"] = max(
            int(cfg.get("data", {}).get("max_text_length") or 0), 5000
        )
        cfg["model"] = {
            "name": enc["name"],
            "num_labels": 2,
            "max_length": enc["max_length"],
        }
        cfg["train"] = dict(cfg.get("train") or {})
        cfg["train"]["batch_size"] = enc["batch_size"]
        cfg["train"].setdefault("warmup_ratio", 0.1)
        cfg["train"].setdefault("max_grad_norm", 1.0)

        cfg["llm"] = dict(cfg.get("llm") or {})
        if best and best.get("best_model"):
            cfg["llm"]["model"] = best["best_model"]
            cfg["llm"]["mode"] = "api"

        cfg["paths"] = {
            "output_dir": f"outputs/bakeoff/encoders/{enc['slug']}",
            "cache_dir": "outputs/cache",
        }
        # Drop internal key before dumping.
        cfg.pop("_config_path", None)

        path = out_dir / f"{enc['slug']}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written


def _build_probe_samples(
    base_config: dict[str, Any],
    *,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Build a small balanced gold set for LLM selection (never from test)."""
    from calibral.data import prepare_data

    cfg = copy.deepcopy(base_config)
    # Keep probe cheap/deterministic; reuse thesis splits when possible.
    bundle = prepare_data(cfg)
    frame = bundle.initial_labeled.copy()
    if len(frame) < max(2, n_samples):
        frame = bundle.validation.copy()
    pos = frame[frame["label"] == LABEL_POSITIVE]
    neg = frame[frame["label"] == LABEL_NEGATIVE]
    half = max(1, n_samples // 2)
    take_pos = pos.head(half)
    take_neg = neg.head(n_samples - len(take_pos))
    chosen = (
        pd.concat([take_pos, take_neg], ignore_index=True)
        .sample(frac=1.0, random_state=int(cfg.get("seed", 42)))
        .reset_index(drop=True)
    )
    samples = [
        {
            "sample_id": str(row["sample_id"]),
            "text": str(row["text"]),
            "gold_label": int(row["label"]),
        }
        for _, row in chosen.iterrows()
    ]
    if samples:
        return samples
    return [
        {
            "sample_id": "fallback_pos",
            "text": PROBE_TEXT,
            "gold_label": PROBE_GOLD_LABEL,
        }
    ]


def run_llm_probe(
    base_config_path: str | Path,
    *,
    candidates: list[str] | None = None,
    n_probe_samples: int = DEFAULT_PROBE_SAMPLES,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate each LLM candidate once on a gold probe set.

    Each model may use as many API calls as needed for the probe samples.
    After a model's evaluation finishes, it is not re-tested in this bakeoff
    (``resume=True`` also skips already-completed models if you re-run).
    """
    logger = get_logger("bakeoff")
    load_dotenv()
    base = load_config(base_config_path)
    llm_cfg = dict(base.get("llm") or {})
    api_key_env = str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    api_key = _read_api_key(api_key_env)
    if not api_key:
        raise MissingAPIKeyError(
            f"{api_key_env} is not set. Put it in .env or the environment."
        )

    candidates = list(candidates or LLM_CANDIDATES)
    out_dir = bakeoff_root() / "llm_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "llm_probe_results.json"

    previous_by_model: dict[str, dict[str, Any]] = {}
    if resume and json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            for row in prev.get("ranking") or []:
                if (
                    row.get("status") == "completed"
                    and row.get("model")
                    and row.get("accuracy") is not None
                ):
                    previous_by_model[str(row["model"])] = row
        except Exception:  # noqa: BLE001
            previous_by_model = {}

    probe_samples = _build_probe_samples(base, n_samples=n_probe_samples)
    rows: list[dict[str, Any]] = []
    logger.info(
        "LLM probe start | n_candidates=%d | n_probe_samples=%d | "
        "one_pass_per_model=True | resume_skip_completed=%s | device=%s",
        len(candidates),
        len(probe_samples),
        resume,
        describe_device(),
    )

    for model in candidates:
        if model in previous_by_model:
            row = previous_by_model[model]
            rows.append(row)
            logger.info(
                "LLM probe skip (already completed) | model=%s | accuracy=%s",
                model,
                row.get("accuracy"),
            )
            continue

        started = time.perf_counter()
        row: dict[str, Any] = {
            "model": model,
            "status": "failed",
            "api_calls": 0,
            "n_probe_samples": len(probe_samples),
            "n_labeled": 0,
            "n_correct": 0,
            "accuracy": None,
            "mean_confidence": None,
            "correct": None,
            "error": None,
            "runtime_seconds": None,
        }
        try:
            labeler = APILLMLabeler(
                base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
                model=model,
                api_key=api_key,
                prompt_version=str(llm_cfg.get("prompt_version", "v1")),
                shot_mode=str(llm_cfg.get("shot_mode", "zero")),
                timeout_seconds=float(llm_cfg.get("timeout_seconds", 60)),
                max_retries=int(llm_cfg.get("max_retries", 3)),
                cache=None,  # fair live probe; no cross-run cache reuse
            )
            confidences: list[float] = []
            n_correct = 0
            n_labeled = 0
            api_calls = 0
            for sample in probe_samples:
                result = labeler.label_text(sample["text"])
                api_calls += 0 if result.cached else 1
                label_name = str(result.label).lower().strip()
                pred = _LLM_TO_INT[label_name]
                n_labeled += 1
                confidences.append(float(result.confidence))
                if pred == int(sample["gold_label"]):
                    n_correct += 1

            accuracy = float(n_correct / max(n_labeled, 1))
            row.update(
                {
                    "status": "completed",
                    "api_calls": api_calls,
                    "n_labeled": n_labeled,
                    "n_correct": n_correct,
                    "accuracy": round(accuracy, 6),
                    "mean_confidence": round(
                        sum(confidences) / max(len(confidences), 1), 6
                    ),
                    "correct": bool(n_correct == n_labeled),
                }
            )
        except Exception as exc:  # noqa: BLE001 — record per-model failure
            row["status"] = "failed"
            row["error"] = str(exc)
            logger.exception("LLM probe failed for %s", model)

        row["runtime_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        logger.info(
            "LLM probe done for model | model=%s | status=%s | accuracy=%s | "
            "calls=%s | runtime=%.1fs",
            model,
            row["status"],
            row.get("accuracy"),
            row["api_calls"],
            row["runtime_seconds"],
        )

    def sort_key(r: dict[str, Any]) -> tuple:
        completed = 1 if r.get("status") == "completed" else 0
        acc = float(r["accuracy"]) if r.get("accuracy") is not None else -1.0
        conf = float(r["mean_confidence"]) if r.get("mean_confidence") is not None else -1.0
        runtime = float(r["runtime_seconds"] or 1e9)
        return (-completed, -acc, -conf, runtime)

    ranked = sorted(rows, key=sort_key)
    best = next((r for r in ranked if r.get("status") == "completed"), None)

    summary = {
        "project": "CalibrAL",
        "stage": "llm_probe",
        "base_config": str(base_config_path),
        "n_probe_samples": len(probe_samples),
        "probe_sample_ids": [s["sample_id"] for s in probe_samples],
        "rule": (
            "Each candidate LLM is evaluated once on the same gold probe set "
            f"({len(probe_samples)} samples). As many API calls as needed are "
            "used during that evaluation; completed models are not re-tested."
        ),
        "device": describe_device(),
        "n_candidates": len(candidates),
        "n_completed": sum(1 for r in rows if r.get("status") == "completed"),
        "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
        "best_model": best["model"] if best else None,
        "best_row": best,
        "ranking": ranked,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    best_llm_path().write_text(
        json.dumps(
            {
                "best_model": summary["best_model"],
                "selected_from": "llm_probe",
                "rule": summary["rule"],
                "n_probe_samples": len(probe_samples),
                "ranking_top3": ranked[:3],
                "results_path": str(json_path),
                "finished_at": summary["finished_at"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path = out_dir / "llm_probe_report.md"
    md_path.write_text(_render_llm_probe_md(summary), encoding="utf-8")
    _write_bakeoff_index()
    logger.info(
        "LLM probe done | best=%s | report=%s",
        summary["best_model"],
        md_path,
    )
    return summary


def _render_llm_probe_md(summary: dict[str, Any]) -> str:
    lines = [
        "# LLM Probe Report",
        "",
        summary["rule"],
        "",
        f"- **best_model:** `{summary.get('best_model')}`",
        f"- **n_probe_samples:** {summary.get('n_probe_samples')}",
        f"- **completed / failed:** {summary.get('n_completed')} / {summary.get('n_failed')}",
        "",
        "| rank | model | status | accuracy | n_correct/n | mean_conf | api_calls | runtime_s | error |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(summary.get("ranking") or [], start=1):
        n_c = row.get("n_correct")
        n_l = row.get("n_labeled")
        frac = f"{n_c}/{n_l}" if n_c is not None and n_l is not None else ""
        lines.append(
            "| {rank} | `{model}` | {status} | {acc} | {frac} | {conf} | {calls} | {rt} | {err} |".format(
                rank=i,
                model=row.get("model"),
                status=row.get("status"),
                acc=row.get("accuracy") if row.get("accuracy") is not None else "",
                frac=frac,
                conf=row.get("mean_confidence") if row.get("mean_confidence") is not None else "",
                calls=row.get("api_calls"),
                rt=row.get("runtime_seconds"),
                err=(str(row.get("error") or "").replace("|", "/"))[:80],
            )
        )
    lines.extend(
        [
            "",
            "## How to use the winner",
            "",
            "The selected model is saved in `outputs/bakeoff/best_llm.json` and is",
            "injected into `configs/thesis_al_final.yaml` for the main experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def run_encoder_sweep(
    base_config_path: str | Path,
    *,
    methods: tuple[str, ...] = ("baseline", "proposed"),
) -> dict[str, Any]:
    """Run thesis_al-style comparison for each encoder into separate folders."""
    logger = get_logger("bakeoff")
    load_dotenv()
    paths = write_encoder_configs(base_config_path)
    items: list[dict[str, Any]] = []
    started = time.perf_counter()

    logger.warning(
        "Encoder sweep can be expensive: each proposed run may call the LLM "
        "up to active_learning.budget times. Bakeoff configs default to seeds=[42]."
    )

    for path in paths:
        config = load_config(path)
        out = resolve_output_dir(config, create=True)
        logger.info(
            "Encoder item | model=%s | llm=%s | out=%s",
            config.get("model", {}).get("name"),
            config.get("llm", {}).get("model"),
            out,
        )
        comparison = run_comparison(config, methods=methods)
        build_report(config)
        items.append(
            {
                "config_path": str(path),
                "profile": config.get("profile"),
                "encoder": config.get("model", {}).get("name"),
                "llm_model": config.get("llm", {}).get("model"),
                "output_dir": str(out),
                "completed": comparison.get("completed"),
                "failed": comparison.get("failed"),
                "runs": comparison.get("runs"),
            }
        )

    summary = {
        "project": "CalibrAL",
        "stage": "encoder_sweep",
        "base_config": str(base_config_path),
        "best_llm": json.loads(best_llm_path().read_text(encoding="utf-8"))
        if best_llm_path().exists()
        else None,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "runtime_human": format_duration(time.perf_counter() - started),
        "items": items,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = bakeoff_root() / "encoder_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = bakeoff_root() / "bakeoff_report.md"
    report_path.write_text(_render_bakeoff_report(summary), encoding="utf-8")
    _write_bakeoff_index()
    logger.info("Encoder sweep done | report=%s", report_path)
    return summary


def _render_bakeoff_report(summary: dict[str, Any]) -> str:
    best = (summary.get("best_llm") or {}).get("best_model")
    lines = [
        "# Thesis Bakeoff Report",
        "",
        f"- **device/runtime:** {summary.get('runtime_human')}",
        f"- **best LLM (from 1-call probe):** `{best}`",
        f"- **base config:** `{summary.get('base_config')}`",
        "",
        "## Encoder results",
        "",
    ]
    for item in summary.get("items") or []:
        lines.append(f"### `{item.get('encoder')}`")
        lines.append("")
        lines.append(f"- output: `{item.get('output_dir')}`")
        lines.append(f"- llm: `{item.get('llm_model')}`")
        lines.append(
            f"- completed/failed: {item.get('completed')} / {item.get('failed')}"
        )
        lines.append("")
        lines.append("| method | seed | status | macro_f1 | accuracy | llm_mode |")
        lines.append("|---|---:|---|---:|---:|---|")
        for run in item.get("runs") or []:
            lines.append(
                "| {method} | {seed} | {status} | {f1} | {acc} | {mode} |".format(
                    method=run.get("method"),
                    seed=run.get("seed"),
                    status=run.get("status"),
                    f1=run.get("macro_f1"),
                    acc=run.get("accuracy"),
                    mode=run.get("llm_mode"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `outputs/bakeoff/best_llm.json`",
            "- `outputs/bakeoff/llm_probe/llm_probe_report.md`",
            "- `outputs/bakeoff/encoders/<slug>/experiment_summary.md`",
            "- `outputs/bakeoff/encoder_sweep_summary.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_bakeoff_index() -> None:
    best = None
    if best_llm_path().exists():
        best = json.loads(best_llm_path().read_text(encoding="utf-8"))
    text = "\n".join(
        [
            "# Bakeoff index",
            "",
            f"- best LLM: `{(best or {}).get('best_model')}`",
            "- LLM probe: `outputs/bakeoff/llm_probe/`",
            "- Main experiment: `outputs/thesis_al/`",
            "- Final config: `configs/thesis_al_final.yaml`",
            "- Encoders (optional): `outputs/bakeoff/encoders/`",
            "- Full report: `outputs/bakeoff/bakeoff_report.md`",
            "",
        ]
    )
    (bakeoff_root() / "README.md").write_text(text, encoding="utf-8")


def apply_best_llm(
    base_config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write a final thesis config with the probed best LLM model."""
    if not best_llm_path().exists():
        raise FileNotFoundError(
            f"Missing {best_llm_path()}. Run llm-probe first."
        )
    best = json.loads(best_llm_path().read_text(encoding="utf-8"))
    best_model = best.get("best_model")
    if not best_model:
        raise ValueError("best_llm.json has no best_model.")

    base = load_config(base_config_path)
    cfg = copy.deepcopy(base)
    cfg["llm"] = dict(cfg.get("llm") or {})
    cfg["llm"]["model"] = best_model
    cfg["llm"]["mode"] = "api"
    cfg["paths"] = dict(cfg.get("paths") or {})
    cfg["paths"].setdefault("output_dir", "outputs/thesis_al")
    cfg["paths"].setdefault("cache_dir", "outputs/cache")
    # Keep a note for the report without breaking YAML consumers.
    cfg["notes"] = (
        f"Main thesis_al run using bakeoff-selected LLM: {best_model}"
    )
    cfg.pop("_config_path", None)

    out = Path(output_path) if output_path else PROJECT_ROOT / "configs" / "thesis_al_final.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    get_logger("bakeoff").info(
        "Wrote final thesis config | llm=%s | path=%s", best_model, out
    )
    return out


def run_main_experiment(
    base_config_path: str | Path | None = None,
    *,
    final_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the main thesis_al comparison with the selected best LLM."""
    logger = get_logger("bakeoff")
    load_dotenv()

    if final_config_path is None:
        final_config_path = apply_best_llm(
            base_config_path or (PROJECT_ROOT / "configs" / "thesis_al.yaml")
        )
    else:
        final_config_path = Path(final_config_path)

    config = load_config(final_config_path)
    out = resolve_output_dir(config, create=True)
    logger.info(
        "Main thesis experiment start | profile=%s | llm=%s | encoder=%s | "
        "seeds=%s | budget=%s | out=%s",
        config.get("profile"),
        config.get("llm", {}).get("model"),
        config.get("model", {}).get("name"),
        config.get("seeds") or [config.get("seed")],
        (config.get("active_learning") or {}).get("budget"),
        out,
    )
    comparison = run_comparison(config)
    report = build_report(config)
    summary = {
        "project": "CalibrAL",
        "stage": "main_experiment",
        "config_path": str(final_config_path),
        "llm_model": config.get("llm", {}).get("model"),
        "encoder": config.get("model", {}).get("name"),
        "output_dir": str(out),
        "comparison": {
            "completed": comparison.get("completed"),
            "failed": comparison.get("failed"),
            "seeds": comparison.get("seeds"),
            "runs": comparison.get("runs"),
            "runtime_seconds": comparison.get("runtime_seconds"),
        },
        "report_path": report.get("report_path"),
        "experiment_summary_path": report.get("experiment_summary_path"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    path = bakeoff_root() / "main_experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_bakeoff_index()
    logger.info(
        "Main thesis experiment done | completed=%s | failed=%s | report=%s",
        comparison.get("completed"),
        comparison.get("failed"),
        report.get("report_path"),
    )
    return summary


def run_all(
    base_config_path: str | Path,
    *,
    include_encoder_sweep: bool = False,
) -> dict[str, Any]:
    """Tune LLM, then run the main thesis_al experiment.

    Encoder ablation remains optional (expensive) via ``include_encoder_sweep``.
    """
    probe = run_llm_probe(base_config_path)
    write_encoder_configs(base_config_path)
    result: dict[str, Any] = {"llm_probe": probe}
    if include_encoder_sweep:
        result["encoder_sweep"] = run_encoder_sweep(base_config_path)
    result["main_experiment"] = run_main_experiment(base_config_path)
    _write_final_bakeoff_report(result)
    return result


def _write_final_bakeoff_report(result: dict[str, Any]) -> Path:
    probe = result.get("llm_probe") or {}
    main = result.get("main_experiment") or {}
    lines = [
        "# Thesis Bakeoff + Main Experiment",
        "",
        "## Selected LLM",
        "",
        f"- **best_model:** `{probe.get('best_model')}`",
        "- selection: each LLM evaluated once on a multi-sample gold probe set",
        f"- probe report: `outputs/bakeoff/llm_probe/llm_probe_report.md`",
        "",
        "## Main thesis_al experiment",
        "",
        f"- config: `{main.get('config_path')}`",
        f"- encoder: `{main.get('encoder')}`",
        f"- llm: `{main.get('llm_model')}`",
        f"- output: `{main.get('output_dir')}`",
        f"- completed/failed: {(main.get('comparison') or {}).get('completed')} / "
        f"{(main.get('comparison') or {}).get('failed')}",
        f"- report: `{main.get('report_path')}`",
        f"- summary: `{main.get('experiment_summary_path')}`",
        "",
        "This is the primary scientific comparison for the thesis "
        "(baseline vs proposed under a fixed human budget).",
        "",
    ]
    if result.get("encoder_sweep"):
        lines.extend(
            [
                "## Optional encoder ablation",
                "",
                "See `outputs/bakeoff/encoder_sweep_summary.json`.",
                "",
            ]
        )
    path = bakeoff_root() / "bakeoff_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
