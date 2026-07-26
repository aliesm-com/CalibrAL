"""Build tables, plots, and report.md from real experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from calibral.config import resolve_output_dir
from calibral.utils import get_logger

REQUIRED_COLUMNS = [
    "run_id",
    "method",
    "seed",
    "profile",
    "macro_f1",
    "accuracy",
    "human_labels_used",
    "llm_labels_accepted",
    "llm_calls",
    "runtime_seconds",
    "llm_mode",
    "status",
]


def load_results(results_path: str | Path) -> pd.DataFrame | None:
    """Load results.csv; return None if missing or empty."""
    path = Path(results_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].astype(str).str.lower() == "completed"].copy()


def format_duration(seconds: float | None) -> str:
    """Human-readable duration for summaries."""
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return "n/a"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def summarize_results(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate metrics from a results dataframe (numbers only from CSV)."""
    completed = _completed(df)
    failed = df[df["status"].astype(str).str.lower() == "failed"].copy()

    methods: dict[str, Any] = {}
    for method, group in completed.groupby("method"):
        methods[str(method)] = {
            "n_runs": int(len(group)),
            "n_seeds": int(group["seed"].nunique()),
            "seeds": sorted(int(s) for s in group["seed"].dropna().unique()),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=0))
            if len(group) > 1
            else 0.0,
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=0))
            if len(group) > 1
            else 0.0,
            "human_labels_used_mean": float(group["human_labels_used"].mean()),
            "llm_labels_accepted_mean": float(group["llm_labels_accepted"].mean()),
            "llm_calls_mean": float(group["llm_calls"].mean()),
            "runtime_seconds_mean": float(group["runtime_seconds"].mean()),
            "runtime_human_mean": format_duration(float(group["runtime_seconds"].mean())),
            "epochs_mean": float(group["epochs"].mean()) if "epochs" in group else None,
            "n_al_rounds_mean": float(group["n_al_rounds"].mean())
            if "n_al_rounds" in group
            else None,
            "llm_modes": sorted({str(m) for m in group["llm_mode"].dropna().unique()}),
            "profiles": sorted({str(p) for p in group["profile"].dropna().unique()}),
        }

    llm_modes_all = sorted({str(m) for m in completed["llm_mode"].dropna().unique()})
    profiles = sorted({str(p) for p in df["profile"].dropna().unique()})
    only_mock_like = bool(llm_modes_all) and set(llm_modes_all).issubset({"mock", "none"})

    return {
        "n_rows_total": int(len(df)),
        "n_completed": int(len(completed)),
        "n_failed": int(len(failed)),
        "successful_seeds": sorted(int(s) for s in completed["seed"].dropna().unique()),
        "n_successful_seeds": int(completed["seed"].nunique()) if len(completed) else 0,
        "profiles": profiles,
        "llm_modes": llm_modes_all,
        "only_mock_or_none": only_mock_like,
        "methods": methods,
        "failed_run_ids": failed["run_id"].astype(str).tolist() if len(failed) else [],
        "total_runtime_seconds": float(df["runtime_seconds"].sum())
        if "runtime_seconds" in df
        else None,
        "total_runtime_human": format_duration(
            float(df["runtime_seconds"].sum()) if "runtime_seconds" in df else None
        ),
    }


def plot_method_comparison(df: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Bar chart of Macro-F1 / Accuracy by method from completed runs."""
    completed = _completed(df)
    if completed.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'matplotlib'. Install with: pip install -r requirements.txt"
        ) from exc

    output_path = Path(output_path)
    agg = (
        completed.groupby("method")[["macro_f1", "accuracy"]]
        .mean()
        .reindex(sorted(completed["method"].unique()))
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(agg.index))
    width = 0.35
    ax.bar([i - width / 2 for i in x], agg["macro_f1"], width=width, label="macro_f1")
    ax.bar([i + width / 2 for i in x], agg["accuracy"], width=width, label="accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in agg.index])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Method comparison (mean over completed runs)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_learning_curve(df: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Learning-style curve: Macro-F1 vs human labels used, by method."""
    completed = _completed(df)
    if completed.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'matplotlib'. Install with: pip install -r requirements.txt"
        ) from exc

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in completed.groupby("method"):
        ordered = group.sort_values(["human_labels_used", "seed"])
        ax.plot(
            ordered["human_labels_used"].tolist(),
            ordered["macro_f1"].tolist(),
            marker="o",
            linestyle="-",
            label=str(method),
        )
    ax.set_xlabel("human_labels_used")
    ax.set_ylabel("macro_f1")
    ax.set_title("Learning curve proxy (from results.csv)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _profile_blurb(profile: str | None) -> str:
    key = str(profile or "").strip().lower()
    blurbs = {
        "thesis_upperbound": (
            "Supervised upper-bound check: large gold training set, full IMDB test, "
            "no AL queries (budget=0). Use this to show whether >~90% is achievable."
        ),
        "thesis_al": (
            "Main thesis AL comparison: baseline (random) vs proposed "
            "(uncertainty + LLM silver labels + limited human oracle) under a fixed "
            "human budget on large IMDB pool/test slices."
        ),
        "smoke": (
            "Software smoke test only. Tiny data/model and mock LLM — "
            "NOT a scientific thesis result."
        ),
        "default": "Default development comparison profile.",
        "colab": "Colab-oriented profile writing under /content/outputs/<profile>.",
    }
    if key in blurbs:
        return blurbs[key]
    if key.startswith("test_llm") or "llm" in key or "slm" in key:
        return "LLM / SLM ablation run (prompt shot mode and/or model family)."
    if any(m in key for m in ("bert", "roberta", "electra", "distilbert")):
        return "Encoder model sweep / ablation run."
    return "Experiment profile defined by the active YAML config."


def render_experiment_summary_markdown(
    *,
    config: dict[str, Any],
    summary: dict[str, Any] | None,
    df: pd.DataFrame | None,
    results_path: str,
    output_dir: str | Path,
) -> str:
    """English, human-readable summary for thesis folders."""
    profile = config.get("profile")
    train = config.get("train") or {}
    data = config.get("data") or {}
    model = config.get("model") or {}
    al = config.get("active_learning") or {}
    llm = config.get("llm") or {}
    output_dir = Path(output_dir)

    lines: list[str] = [
        f"# Experiment Summary — `{profile}`",
        "",
        "## What this folder is",
        "",
        _profile_blurb(str(profile) if profile else None),
        "",
        "All artifacts for this config live in this folder so runs do not overwrite "
        "other experiments. Shared dataset cache stays under `outputs/cache` "
        "(or the configured `paths.cache_dir`).",
        "",
        "## Setup (from config)",
        "",
        f"- **profile:** `{profile}`",
        f"- **config_path:** `{config.get('_config_path', 'n/a')}`",
        f"- **seeds:** `{config.get('seeds') or [config.get('seed')]}`",
        f"- **model:** `{model.get('name')}` (max_length={model.get('max_length')})",
        f"- **epochs (per retrain):** `{train.get('epochs')}`",
        f"- **batch_size:** `{train.get('batch_size')}`",
        f"- **learning_rate:** `{train.get('learning_rate')}`",
        f"- **weight_decay:** `{train.get('weight_decay')}`",
        (
            f"- **data sizes:** initial_labeled={data.get('initial_labeled_size')}, "
            f"pool={data.get('pool_size')}, validation={data.get('validation_size')}, "
            f"test={data.get('test_size')}"
        ),
        (
            f"- **active learning:** budget={al.get('budget')}, "
            f"query_batch_size={al.get('query_batch_size')}, "
            f"uncertainty={al.get('uncertainty')}, "
            f"silver_threshold={al.get('silver_threshold')}, "
            f"silver_weight={al.get('silver_weight')}"
        ),
        (
            f"- **LLM:** mode={llm.get('mode')}, model=`{llm.get('model')}`, "
            f"shot_mode={llm.get('shot_mode')}, provider={llm.get('provider')}"
        ),
        f"- **output_dir:** `{output_dir.as_posix()}`",
        "",
    ]

    if summary is None or df is None:
        lines.extend(
            [
                "## Status",
                "",
                "Incomplete — `results.csv` is missing or empty. "
                "Run the experiment for this config first, then re-run report.",
                "",
                f"- expected results path: `{results_path}`",
                "",
            ]
        )
        return "\n".join(lines)

    if summary.get("only_mock_or_none"):
        lines.extend(
            [
                "> **Disclaimer:** completed rows use mock/none LLM modes. "
                "Treat as pipeline validation, not thesis-quality scientific evidence.",
                "",
            ]
        )

    lines.extend(
        [
            "## Runtime overview",
            "",
            f"- **rows (total / completed / failed):** "
            f"{summary.get('n_rows_total')} / {summary.get('n_completed')} / "
            f"{summary.get('n_failed')}",
            f"- **sum of per-run runtime_seconds:** "
            f"{_fmt(summary.get('total_runtime_seconds'), 3)} "
            f"({summary.get('total_runtime_human')})",
            f"- **successful seeds:** {summary.get('successful_seeds')}",
            f"- **llm_modes in completed runs:** {summary.get('llm_modes')}",
            "",
            "## Per-run results (from results.csv)",
            "",
        ]
    )

    cols = [
        "method",
        "seed",
        "status",
        "macro_f1",
        "accuracy",
        "epochs",
        "n_al_rounds",
        "human_labels_used",
        "llm_labels_accepted",
        "llm_calls",
        "runtime_seconds",
        "runtime_human",
        "llm_mode",
        "llm_shot_mode",
    ]
    present = [c for c in cols if c in df.columns]
    text_cols = {"method", "status", "llm_mode", "llm_shot_mode", "runtime_human"}
    header = "| " + " | ".join(present) + " |"
    sep = "|" + "|".join(["---" if c in text_cols else "---:" for c in present]) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in df.sort_values(["method", "seed"], na_position="last").iterrows():
        cells = []
        for c in present:
            val = row.get(c)
            if c in {"macro_f1", "accuracy"}:
                cells.append(_fmt(val, 4))
            elif c == "runtime_seconds":
                cells.append(_fmt(val, 3))
            elif c == "runtime_human" and (
                val is None or (isinstance(val, float) and pd.isna(val))
            ):
                cells.append(format_duration(row.get("runtime_seconds")))
            else:
                cells.append(_fmt(val, 2) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(["## Method means (completed runs only)", ""])
    methods = summary.get("methods") or {}
    if not methods:
        lines.append("No completed runs yet.")
        lines.append("")
    else:
        lines.append(
            "| method | n_runs | macro_f1_mean | accuracy_mean | "
            "epochs_mean | n_al_rounds_mean | runtime_mean | human_labels_mean | "
            "llm_accepted_mean | llm_calls_mean |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        for method, stats in sorted(methods.items()):
            lines.append(
                "| {method} | {n_runs} | {macro_f1_mean} | {accuracy_mean} | "
                "{epochs_mean} | {n_al_rounds_mean} | {runtime_human_mean} | "
                "{human_labels_used_mean} | {llm_labels_accepted_mean} | "
                "{llm_calls_mean} |".format(
                    method=method,
                    n_runs=stats["n_runs"],
                    macro_f1_mean=_fmt(stats["macro_f1_mean"]),
                    accuracy_mean=_fmt(stats["accuracy_mean"]),
                    epochs_mean=_fmt(stats.get("epochs_mean"), 2),
                    n_al_rounds_mean=_fmt(stats.get("n_al_rounds_mean"), 2),
                    runtime_human_mean=stats.get("runtime_human_mean", "n/a"),
                    human_labels_used_mean=_fmt(stats["human_labels_used_mean"], 2),
                    llm_labels_accepted_mean=_fmt(stats["llm_labels_accepted_mean"], 2),
                    llm_calls_mean=_fmt(stats["llm_calls_mean"], 2),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Files in this folder",
            "",
            f"- `{output_dir.as_posix()}/results.csv` — source of truth for all numbers",
            f"- `{output_dir.as_posix()}/experiment_summary.md` — this file",
            f"- `{output_dir.as_posix()}/report.md` — longer report (after `report`)",
            f"- `{output_dir.as_posix()}/method_comparison.png`",
            f"- `{output_dir.as_posix()}/learning_curve.png`",
            f"- `{output_dir.as_posix()}/comparison_summary.json`",
            f"- `{output_dir.as_posix()}/run_summary.json` — last single-run JSON",
            f"- `{output_dir.as_posix()}/predictions_*.csv`",
            "",
            "## How to cite in the thesis",
            "",
            "1. Copy metrics only from `results.csv` / this summary (never invent numbers).",
            "2. Exclude smoke/mock rows from scientific claims.",
            "3. Report mean ± variation across seeds when multiple seeds completed.",
            "4. Mention epochs, human budget, test size, and wall-clock runtime.",
            "",
        ]
    )
    return "\n".join(lines)


def write_experiment_summary(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    results_path: str | Path | None = None,
) -> Path:
    """Write English ``experiment_summary.md`` into the experiment folder."""
    if output_dir is None:
        output_dir = resolve_output_dir(config, create=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results_path is None:
        results_path = output_dir / "results.csv"
    results_path = Path(results_path)

    df = load_results(results_path)
    summary = summarize_results(df) if df is not None else None
    text = render_experiment_summary_markdown(
        config=config,
        summary=summary,
        df=df,
        results_path=str(results_path),
        output_dir=output_dir,
    )
    path = output_dir / "experiment_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def render_report_markdown(
    summary: dict[str, Any] | None,
    *,
    results_path: str,
    config_snapshot: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Render report.md text. All metric numbers come from ``summary``/CSV."""
    lines: list[str] = ["# CalibrAL Report", ""]
    out = Path(output_dir) if output_dir is not None else Path("outputs")

    if incomplete_reason is not None or summary is None:
        lines.extend(
            [
                "## Status: incomplete",
                "",
                incomplete_reason
                or "Experiment results are missing or empty; report is incomplete.",
                "",
                f"- results_path: `{results_path}`",
                "",
            ]
        )
        return "\n".join(lines)

    if summary.get("only_mock_or_none"):
        lines.extend(
            [
                "> این گزارش مربوط به تست نرم‌افزاری با LLM Mock است و نتیجه علمی نهایی نیست.",
                "",
            ]
        )

    lines.extend(
        [
            "## Run configuration",
            "",
            f"- results_path: `{results_path}`",
            f"- output_dir: `{out.as_posix()}`",
            f"- profiles_in_csv: {summary.get('profiles')}",
            f"- llm_modes_in_completed_runs: {summary.get('llm_modes')}",
            f"- total_rows: {summary.get('n_rows_total')}",
            f"- completed_rows: {summary.get('n_completed')}",
            f"- failed_rows: {summary.get('n_failed')}",
            f"- total_runtime: {summary.get('total_runtime_human')} "
            f"({_fmt(summary.get('total_runtime_seconds'), 3)}s)",
            "",
        ]
    )

    if config_snapshot:
        train = config_snapshot.get("train") or {}
        lines.extend(
            [
                "### Training / AL knobs",
                "",
                f"- epochs: `{train.get('epochs')}`",
                f"- batch_size: `{train.get('batch_size')}`",
                f"- learning_rate: `{train.get('learning_rate')}`",
                f"- active_learning.budget: "
                f"`{(config_snapshot.get('active_learning') or {}).get('budget')}`",
                "",
            ]
        )
        lines.append("### Config snapshot (from active config file)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(config_snapshot, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Successful seeds",
            "",
            f"- n_successful_seeds: {summary.get('n_successful_seeds')}",
            f"- seeds: {summary.get('successful_seeds')}",
            "",
            "## Method comparison (completed runs only)",
            "",
        ]
    )

    methods = summary.get("methods", {})
    if not methods:
        lines.append("No completed runs were found in results.csv.")
        lines.append("")
    else:
        lines.append(
            "| method | n_runs | n_seeds | macro_f1_mean | accuracy_mean | "
            "human_labels_used_mean | llm_labels_accepted_mean | llm_calls_mean | "
            "runtime_seconds_mean | runtime_human_mean | llm_modes |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for method, stats in sorted(methods.items()):
            lines.append(
                "| {method} | {n_runs} | {n_seeds} | {macro_f1_mean} | {accuracy_mean} | "
                "{human_labels_used_mean} | {llm_labels_accepted_mean} | {llm_calls_mean} | "
                "{runtime_seconds_mean} | {runtime_human_mean} | {llm_modes} |".format(
                    method=method,
                    n_runs=stats["n_runs"],
                    n_seeds=stats["n_seeds"],
                    macro_f1_mean=_fmt(stats["macro_f1_mean"]),
                    accuracy_mean=_fmt(stats["accuracy_mean"]),
                    human_labels_used_mean=_fmt(stats["human_labels_used_mean"], 2),
                    llm_labels_accepted_mean=_fmt(stats["llm_labels_accepted_mean"], 2),
                    llm_calls_mean=_fmt(stats["llm_calls_mean"], 2),
                    runtime_seconds_mean=_fmt(stats["runtime_seconds_mean"], 3),
                    runtime_human_mean=stats.get("runtime_human_mean", "n/a"),
                    llm_modes=", ".join(stats["llm_modes"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- `{out.as_posix()}/method_comparison.png`",
            f"- `{out.as_posix()}/learning_curve.png`",
            f"- `{out.as_posix()}/results.csv`",
            f"- `{out.as_posix()}/experiment_summary.md`",
            "",
            "## Limitations",
            "",
            "- Numbers in this report are computed only from `results.csv`.",
            "- Failed runs are counted but excluded from mean metrics.",
            "- Smoke / mock LLM rows are not thesis-quality scientific results.",
            "- Learning curve is a proxy plot of macro_f1 vs human_labels_used from saved runs.",
            "",
        ]
    )

    if summary.get("failed_run_ids"):
        lines.append("### Failed run_ids")
        lines.append("")
        for run_id in summary["failed_run_ids"]:
            lines.append(f"- `{run_id}`")
        lines.append("")

    return "\n".join(lines)


def build_report(
    config: dict[str, Any] | None = None,
    *,
    results_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create plots and report.md from real CSV artifacts."""
    logger = get_logger("reporting")
    config = config or {}
    if output_dir is None:
        output_dir = resolve_output_dir(config, create=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results_path is None:
        results_path = output_dir / "results.csv"
    results_path = Path(results_path)

    comparison_path = output_dir / "method_comparison.png"
    learning_path = output_dir / "learning_curve.png"
    report_path = output_dir / "report.md"

    df = load_results(results_path)
    summary_md_path = write_experiment_summary(
        config,
        output_dir=output_dir,
        results_path=results_path,
    )

    if df is None:
        report_text = render_report_markdown(
            None,
            results_path=str(results_path),
            config_snapshot=_config_snapshot(config),
            incomplete_reason=(
                "Experiment has not been run or results.csv is missing/empty. "
                "Report status: incomplete."
            ),
            output_dir=output_dir,
        )
        report_path.write_text(report_text, encoding="utf-8")
        logger.warning("Report incomplete; missing results at %s", results_path)
        return {
            "status": "incomplete",
            "report_path": str(report_path),
            "experiment_summary_path": str(summary_md_path),
            "results_path": str(results_path),
            "method_comparison_path": None,
            "learning_curve_path": None,
        }

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv missing columns: {missing}")

    summary = summarize_results(df)
    method_plot = plot_method_comparison(df, comparison_path)
    learning_plot = plot_learning_curve(df, learning_path)
    report_text = render_report_markdown(
        summary,
        results_path=str(results_path),
        config_snapshot=_config_snapshot(config),
        output_dir=output_dir,
    )
    report_path.write_text(report_text, encoding="utf-8")

    logger.info(
        "Report built | completed=%d | failed=%d | report=%s | summary=%s",
        summary["n_completed"],
        summary["n_failed"],
        report_path,
        summary_md_path,
    )
    return {
        "status": "completed" if summary["n_completed"] > 0 else "incomplete",
        "summary": summary,
        "report_path": str(report_path),
        "experiment_summary_path": str(summary_md_path),
        "results_path": str(results_path),
        "method_comparison_path": str(method_plot) if method_plot else None,
        "learning_curve_path": str(learning_plot) if learning_plot else None,
    }


def _config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Keep a compact, non-secret config snapshot for the report."""
    if not config:
        return {}
    keys = [
        "profile",
        "seed",
        "seeds",
        "data",
        "model",
        "train",
        "active_learning",
        "llm",
        "paths",
        "notes",
    ]
    snap = {k: config.get(k) for k in keys if k in config}
    # Never include secrets; only the env var name lives in llm.api_key_env.
    return snap
