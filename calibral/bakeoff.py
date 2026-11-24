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
    LLMFatalError,
    MissingAPIKeyError,
    _read_api_key,
    api_request_options,
    resolve_domain,
)
from calibral.reporting import build_report, format_duration
from calibral.utils import PROJECT_ROOT, describe_device, get_logger, load_dotenv

_LLM_TO_INT = {"negative": LABEL_NEGATIVE, "positive": LABEL_POSITIVE}

# Candidates of the original thesis bakeoff (kept for reference/reproduction).
LEGACY_LLM_CANDIDATES: list[str] = [
    "openai/gpt-4o-mini",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.5-9b",
    "mistralai/ministral-3b-2512",
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o",
    "x-ai/grok-4.3",
]

# Default candidates for new (paper) runs. Pinned snapshots only: "~...-latest"
# aliases silently move to new weights, which breaks reproducibility.
# Override per experiment with ``bakeoff.llm_candidates`` in the YAML.
LLM_CANDIDATES: list[str] = [
    # small / cheap tier
    "mistralai/ministral-3b-2512",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.6-27b",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-4o-mini",
    # strong tier
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v4-pro-0813",
    "qwen/qwen3.7-plus",
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


_BAKEOFF_ROOT_OVERRIDE: Path | None = None


def configure_bakeoff(config: dict[str, Any]) -> Path:
    """Point bakeoff artifacts at ``bakeoff.output_dir`` (default outputs/bakeoff).

    Lets a new experiment (e.g. the paper profile) keep its own probe results,
    best_llm.json and reports without overwriting the thesis bakeoff.
    """
    global _BAKEOFF_ROOT_OVERRIDE
    configured = (config.get("bakeoff") or {}).get("output_dir")
    if configured:
        path = Path(str(configured))
        _BAKEOFF_ROOT_OVERRIDE = path if path.is_absolute() else PROJECT_ROOT / path
    else:
        _BAKEOFF_ROOT_OVERRIDE = None
    return bakeoff_root()


def bakeoff_root() -> Path:
    path = _BAKEOFF_ROOT_OVERRIDE or (PROJECT_ROOT / "outputs" / "bakeoff")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bakeoff_rel() -> str:
    root = bakeoff_root()
    try:
        return root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return root.as_posix()


def best_llm_path() -> Path:
    return bakeoff_root() / "best_llm.json"


def encoder_config_dir() -> Path:
    path = PROJECT_ROOT / "configs" / "sweeps" / "thesis_encoders"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_encoder_configs(base_config_path: str | Path) -> list[Path]:
    """Materialize per-encoder YAMLs derived from thesis_al."""
    base = load_config(base_config_path)
    configure_bakeoff(base)
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
            "output_dir": f"{_bakeoff_rel()}/encoders/{enc['slug']}",
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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _calibration_metrics(
    records: list[dict[str, Any]], n_bins: int = 10
) -> dict[str, float | None]:
    """Brier score + ECE of the LLM's self-reported confidence (answered only)."""
    answered = [r for r in records if r.get("pred") is not None]
    if not answered:
        return {"brier": None, "ece": None}
    brier = sum(
        (float(r["confidence"]) - (1.0 if r["correct"] else 0.0)) ** 2 for r in answered
    ) / len(answered)
    bins: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for r in answered:
        idx = min(int(float(r["confidence"]) * n_bins), n_bins - 1)
        bins[idx].append(r)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for r in b if r["correct"]) / len(b)
        conf = sum(float(r["confidence"]) for r in b) / len(b)
        ece += len(b) / len(answered) * abs(acc - conf)
    return {"brier": round(brier, 6), "ece": round(ece, 6)}


def _probe_usage(answered: list[dict[str, Any]]) -> dict[str, Any]:
    """Token / USD totals for one probe model (OpenRouter usage.cost)."""
    if not answered:
        return {}
    known = [r for r in answered if r.get("cost_usd") is not None]
    cost = sum(float(r["cost_usd"]) for r in known)
    return {
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in answered),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in answered),
        "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in answered),
        "cost_usd": round(cost, 6) if known else None,
        "cost_unknown_calls": len(answered) - len(known),
        # Extrapolated cost of labeling 1,000 texts like these (for the paper's
        # cost axis; scale by the text-length ratio for other datasets).
        "cost_per_1k_labels_usd": round(1000 * cost / len(known), 4) if known else None,
    }


def run_llm_probe(
    base_config_path: str | Path,
    *,
    candidates: list[str] | None = None,
    n_probe_samples: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate each LLM candidate once on a gold probe set.

    Each model may use as many API calls as needed for the probe samples.
    After a model's evaluation finishes, it is not re-tested in this bakeoff
    (``resume=True`` also skips already-completed models if you re-run with
    the same prompt_version / shot_mode / probe size).

    A sample that still fails after retries is counted as an error (wrong
    answer) instead of failing the whole model, so one stubborn response no
    longer removes a candidate from the comparison. ``accuracy`` counts
    errors as wrong; ``accuracy_answered`` excludes them.
    """
    logger = get_logger("bakeoff")
    load_dotenv()
    base = load_config(base_config_path)
    configure_bakeoff(base)
    bakeoff_cfg = dict(base.get("bakeoff") or {})
    llm_cfg = dict(base.get("llm") or {})
    api_key_env = str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    api_key = _read_api_key(api_key_env)
    if not api_key:
        raise MissingAPIKeyError(
            f"{api_key_env} is not set. Put it in .env or the environment."
        )

    candidates = list(
        candidates or bakeoff_cfg.get("llm_candidates") or LLM_CANDIDATES
    )
    if n_probe_samples is None:
        n_probe_samples = int(bakeoff_cfg.get("n_probe_samples") or DEFAULT_PROBE_SAMPLES)
    prompt_version = str(llm_cfg.get("prompt_version", "v1"))
    shot_mode = str(llm_cfg.get("shot_mode", "zero"))
    request_options = api_request_options(llm_cfg)

    out_dir = bakeoff_root() / "llm_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "llm_probe_results.json"

    previous_by_model: dict[str, dict[str, Any]] = {}
    if resume and json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            same_setup = (
                str(prev.get("prompt_version", "v1")) == prompt_version
                and str(prev.get("shot_mode", "zero")) == shot_mode
                and int(prev.get("n_probe_samples") or 0) == n_probe_samples
            )
            for row in (prev.get("ranking") or []) if same_setup else []:
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
        "LLM probe start | n_candidates=%d | n_probe_samples=%d | prompt=%s | "
        "shot=%s | resume_skip_completed=%s | out=%s | device=%s",
        len(candidates),
        len(probe_samples),
        prompt_version,
        shot_mode,
        resume,
        out_dir,
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
            "n_errors": 0,
            "accuracy": None,
            "accuracy_ci95": None,
            "accuracy_answered": None,
            "mean_confidence": None,
            "brier": None,
            "ece": None,
            "served_providers": [],
            "served_models": [],
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cost_usd": None,
            "cost_unknown_calls": None,
            "cost_per_1k_labels_usd": None,
            "correct": None,
            "error": None,
            "runtime_seconds": None,
        }
        records: list[dict[str, Any]] = []
        try:
            labeler = APILLMLabeler(
                base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
                model=model,
                api_key=api_key,
                prompt_version=prompt_version,
                shot_mode=shot_mode,
                timeout_seconds=float(llm_cfg.get("timeout_seconds", 60)),
                max_retries=int(llm_cfg.get("max_retries", 3)),
                cache=None,  # fair live probe; no cross-run cache reuse
                domain=resolve_domain(base),
                **request_options,
            )
            first_error: str | None = None
            for sample in probe_samples:
                gold = int(sample["gold_label"])
                try:
                    result = labeler.label_text(sample["text"])
                except LLMFatalError:
                    raise
                except Exception as exc:  # noqa: BLE001 - count, don't abort model
                    first_error = first_error or str(exc)
                    records.append(
                        {
                            "sample_id": sample["sample_id"],
                            "gold": gold,
                            "pred": None,
                            "confidence": None,
                            "correct": False,
                            "error": str(exc)[:300],
                        }
                    )
                    continue
                pred = _LLM_TO_INT[str(result.label).lower().strip()]
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "gold": gold,
                        "pred": pred,
                        "confidence": float(result.confidence),
                        "correct": pred == gold,
                        "error": None,
                        "served_model": result.served_model,
                        "provider": result.provider,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "reasoning_tokens": result.reasoning_tokens,
                        "cost_usd": result.cost_usd,
                    }
                )

            n_total = len(records)
            answered = [r for r in records if r["pred"] is not None]
            n_correct = sum(1 for r in records if r["correct"])
            n_errors = n_total - len(answered)
            lo, hi = _wilson_interval(n_correct, n_total)
            row.update(
                {
                    "api_calls": n_total,
                    "n_labeled": len(answered),
                    "n_correct": n_correct,
                    "n_errors": n_errors,
                    "accuracy": round(n_correct / max(n_total, 1), 6),
                    "accuracy_ci95": [round(lo, 4), round(hi, 4)],
                    "accuracy_answered": round(n_correct / len(answered), 6)
                    if answered
                    else None,
                    "mean_confidence": round(
                        sum(r["confidence"] for r in answered) / len(answered), 6
                    )
                    if answered
                    else None,
                    "served_providers": sorted(
                        {str(r.get("provider")) for r in answered if r.get("provider")}
                    ),
                    "served_models": sorted(
                        {str(r.get("served_model")) for r in answered if r.get("served_model")}
                    ),
                    **_probe_usage(answered),
                    "correct": bool(n_correct == n_total),
                    "error": first_error,
                    # A model that answered nothing is a failed candidate.
                    "status": "completed" if answered else "failed",
                    **_calibration_metrics(records),
                }
            )
        except LLMFatalError:
            raise
        except Exception as exc:  # noqa: BLE001 — record per-model failure
            row["status"] = "failed"
            row["error"] = str(exc)
            logger.exception("LLM probe failed for %s", model)

        if records:
            slug = model.replace("/", "__").replace(":", "_")
            with (pred_dir / f"{slug}.jsonl").open("w", encoding="utf-8") as handle:
                for rec in records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        row["runtime_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        done_cost = sum(float(r.get("cost_usd") or 0) for r in rows)
        logger.info(
            "[%d/%d] %s | %s | accuracy=%s | errors=%s | tokens=%s+%s | cost=$%s | "
            "runtime=%.1fs | spent so far=$%.4f",
            len(rows),
            len(candidates),
            model,
            row["status"],
            row.get("accuracy"),
            row.get("n_errors"),
            row.get("prompt_tokens"),
            row.get("completion_tokens"),
            row.get("cost_usd"),
            row["runtime_seconds"],
            done_cost,
        )

    def sort_key(r: dict[str, Any]) -> tuple:
        # Accuracy first (errors count as wrong); ties broken by better
        # calibration (lower Brier), since silver labels are gated on the
        # LLM's confidence; then fewer errors, then speed.
        completed = 1 if r.get("status") == "completed" else 0
        acc = float(r["accuracy"]) if r.get("accuracy") is not None else -1.0
        brier = float(r["brier"]) if r.get("brier") is not None else 1e9
        errors = int(r.get("n_errors") or 0)
        runtime = float(r["runtime_seconds"] or 1e9)
        return (-completed, -acc, brier, errors, runtime)

    ranked = sorted(rows, key=sort_key)
    best = next((r for r in ranked if r.get("status") == "completed"), None)

    summary = {
        "project": "CalibrAL",
        "stage": "llm_probe",
        "base_config": str(base_config_path),
        "prompt_version": prompt_version,
        "shot_mode": shot_mode,
        "request_options": dict(request_options),
        "n_probe_samples": len(probe_samples),
        "probe_sample_ids": [s["sample_id"] for s in probe_samples],
        "rule": (
            "Each candidate LLM is evaluated once on the same gold probe set "
            f"({len(probe_samples)} samples). As many API calls as needed are "
            "used during that evaluation; a sample that still fails after "
            "retries counts as wrong; completed models are not re-tested. "
            "Ranking: accuracy, then Brier score (lower is better)."
        ),
        "device": describe_device(),
        "n_candidates": len(candidates),
        "n_completed": sum(1 for r in rows if r.get("status") == "completed"),
        "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
        "best_model": best["model"] if best else None,
        "best_row": best,
        "total_cost_usd": round(
            sum(float(r.get("cost_usd") or 0) for r in rows), 6
        ),
        "total_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        "ranking": ranked,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    from calibral.probe_report import run_stamp

    summary["run_stamp"] = run_stamp(summary)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    best_llm_path().write_text(
        json.dumps(
            {
                "best_model": summary["best_model"],
                "selected_from": "llm_probe",
                "probe_run": summary["run_stamp"],
                "probe_report": f"PROBE_REPORT_{summary['run_stamp']}.md",
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
    try:
        from calibral.probe_report import build_probe_report

        build_probe_report(bakeoff_root())
    except Exception as exc:  # noqa: BLE001 - presentation must not break the probe
        logger.warning("PROBE_REPORT.md generation failed: %s", exc)
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
        f"**Run:** {summary.get('run_stamp', '')} (local time) · finished_at {summary.get('finished_at', '')}",
        "",
        summary["rule"],
        "",
        f"- **best_model:** `{summary.get('best_model')}`",
        f"- **n_probe_samples:** {summary.get('n_probe_samples')}",
        f"- **completed / failed:** {summary.get('n_completed')} / {summary.get('n_failed')}",
        "",
        f"- **prompt / shot:** {summary.get('prompt_version')} / {summary.get('shot_mode')}",
        f"- **total probe cost:** ${summary.get('total_cost_usd', 0):.4f} "
        f"({summary.get('total_prompt_tokens', 0):,} prompt + "
        f"{summary.get('total_completion_tokens', 0):,} completion tokens)",
        "",
        "| rank | model | status | accuracy | 95% CI | n_correct/n | errors | mean_conf | brier | ece | "
        "prompt tok | completion tok | reasoning tok | cost $ | $ / 1k labels | runtime_s | error |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def _fmt(value: Any) -> str:
        return "" if value is None else str(value)

    for i, row in enumerate(summary.get("ranking") or [], start=1):
        n_c = row.get("n_correct")
        n_t = row.get("n_probe_samples") or row.get("n_labeled")
        frac = f"{n_c}/{n_t}" if n_c is not None and n_t is not None else ""
        ci = row.get("accuracy_ci95")
        ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        lines.append(
            f"| {i} | `{row.get('model')}` | {row.get('status')} | "
            f"{_fmt(row.get('accuracy'))} | {ci_txt} | {frac} | "
            f"{_fmt(row.get('n_errors'))} | {_fmt(row.get('mean_confidence'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('ece'))} | "
            f"{_fmt(row.get('prompt_tokens'))} | {_fmt(row.get('completion_tokens'))} | "
            f"{_fmt(row.get('reasoning_tokens'))} | {_fmt(row.get('cost_usd'))} | "
            f"{_fmt(row.get('cost_per_1k_labels_usd'))} | "
            f"{_fmt(row.get('runtime_seconds'))} | "
            f"{(str(row.get('error') or '').replace('|', '/'))[:80]} |"
        )
    lines.extend(
        [
            "",
            "## How to use the winner",
            "",
            "The selected model is saved in `best_llm.json` (next to this folder) and",
            "injected into the final config (`bakeoff.final_config`, default",
            "`configs/thesis_al_final.yaml`) for the main experiment.",
            "Per-sample answers: `llm_probe/predictions/<model>.jsonl`.",
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
    configure_bakeoff(load_config(base_config_path))
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
    base = load_config(base_config_path)
    configure_bakeoff(base)
    if not best_llm_path().exists():
        raise FileNotFoundError(
            f"Missing {best_llm_path()}. Run llm-probe first."
        )
    best = json.loads(best_llm_path().read_text(encoding="utf-8"))
    best_model = best.get("best_model")
    if not best_model:
        raise ValueError("best_llm.json has no best_model.")

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

    if output_path is None:
        output_path = (base.get("bakeoff") or {}).get("final_config")
    if output_path is None:
        out = PROJECT_ROOT / "configs" / "thesis_al_final.yaml"
    else:
        out = Path(output_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
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
    configure_bakeoff(config)
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
        f"- probe report: `{bakeoff_root() / 'llm_probe' / 'llm_probe_report.md'}`",
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
