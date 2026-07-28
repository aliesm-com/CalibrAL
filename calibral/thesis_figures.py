"""Thesis-quality figures from saved artifacts (no re-training required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from calibral.reporting import _completed, dedupe_results, load_results

METHOD_COLORS = {
    "baseline": "#4C72B0",
    "proposed": "#55A868",
}
METHOD_LABELS = {
    "baseline": "Baseline (random AL)",
    "proposed": "Proposed (uncertainty + LLM silver)",
}


def _import_plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'matplotlib'. Install with: pip install -r requirements.txt"
        ) from exc
    return plt


def _apply_thesis_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )



def collect_run_summaries(output_dir: Path) -> list[dict[str, Any]]:
    """Load per-run JSON summaries if present, plus legacy run_summary.json."""
    output_dir = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    archive = output_dir / "run_summaries"
    if archive.is_dir():
        for path in sorted(archive.glob("*.json")):
            try:
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    legacy = output_dir / "run_summary.json"
    if legacy.is_file():
        try:
            row = json.loads(legacy.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            row = None
        if row and not any(s.get("run_id") == row.get("run_id") for s in summaries):
            summaries.append(row)
    return summaries


def _effective_train_size(row: pd.Series) -> float:
    human = float(row.get("human_labels_used") or 0)
    silver = float(row.get("llm_labels_accepted") or 0)
    if str(row.get("method")) == "baseline":
        return human
    return human + silver


def plot_macro_f1_by_seed(df: pd.DataFrame, output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    data = dedupe_results(df)
    if data.empty:
        return None

    seeds = sorted(data["seed"].unique())
    methods = [m for m in ("baseline", "proposed") if m in set(data["method"])]
    if not methods:
        methods = sorted(data["method"].unique())

    x = list(range(len(seeds)))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(methods):
        subset = data[data["method"] == method].set_index("seed")
        heights = [float(subset.loc[s, "macro_f1"]) if s in subset.index else 0 for s in seeds]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            heights,
            width=width * 0.95,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, h in zip(bars, heights):
            if h:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.002,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Test Macro-F1 by seed (same human budget)")
    ymin = max(0.0, float(data["macro_f1"].min()) - 0.02)
    ax.set_ylim(ymin, min(1.0, float(data["macro_f1"].max()) + 0.03))
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_improvement_by_seed(df: pd.DataFrame, output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    data = dedupe_results(df)
    if data.empty:
        return None
    pivot = data.pivot_table(index="seed", columns="method", values="macro_f1", aggfunc="last")
    if "baseline" not in pivot.columns or "proposed" not in pivot.columns:
        return None
    delta = (pivot["proposed"] - pivot["baseline"]).dropna()
    if delta.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in delta.values]
    ax.bar([str(s) for s in delta.index], delta.values, color=colors, edgecolor="white")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Δ Macro-F1 (proposed − baseline)")
    ax.set_title("Improvement over random-query baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(delta.values):
        ax.text(i, v + (0.0005 if v >= 0 else -0.0015), f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_size_vs_f1(df: pd.DataFrame, output_path: Path) -> Path | None:
    """Effective training set size vs test Macro-F1 (shows spread across seeds)."""
    plt = _import_plt()
    _apply_thesis_style(plt)
    data = dedupe_results(df)
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, group in data.groupby("method"):
        g = group.copy()
        g["train_size"] = g.apply(_effective_train_size, axis=1)
        g = g.sort_values("train_size")
        ax.plot(
            g["train_size"],
            g["macro_f1"],
            marker="o",
            markersize=9,
            linewidth=1.8,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
        )
        for _, row in g.iterrows():
            ax.annotate(
                f"seed {int(row['seed'])}",
                (row["train_size"], row["macro_f1"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )

    ax.set_xlabel("Effective training labels (gold + accepted silver)")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Performance vs effective supervision scale")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_al_label_composition(summaries: list[dict[str, Any]], output_path: Path) -> Path | None:
    """Stacked silver vs gold labels per AL round (from saved run summaries)."""
    plt = _import_plt()
    _apply_thesis_style(plt)
    panels = [
        s
        for s in summaries
        if s.get("method") == "proposed" and isinstance(s.get("rounds"), list) and s["rounds"]
    ]
    if not panels:
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, summary in zip(axes, panels):
        rounds = summary["rounds"]
        labels = [f"R{r.get('round', i+1)}" for i, r in enumerate(rounds)]
        silver = [int(r.get("silver", 0)) for r in rounds]
        gold = [int(r.get("gold", 0)) for r in rounds]
        x = range(len(labels))
        ax.bar(x, silver, label="LLM silver", color="#6ACC65", edgecolor="white")
        ax.bar(x, gold, bottom=silver, label="Human gold", color="#4C72B0", edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=0)
        title = f"seed {summary.get('seed')}"
        if any(r.get("phase") == "silver_expansion" for r in rounds):
            title += " (+ expansion)"
        ax.set_title(title)
        ax.set_xlabel("Active learning round")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Labels acquired in round")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Label acquisition per round (proposed method)", y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cumulative_training_size(summaries: list[dict[str, Any]], output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    proposed = [
        s
        for s in summaries
        if s.get("method") == "proposed" and isinstance(s.get("rounds"), list)
    ]
    if not proposed:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for summary in proposed:
        rounds = summary["rounds"]
        init_block = summary.get("initial_train") or {}
        start = int(init_block.get("n_train", 500))
        xs = [0]
        ys = [start]
        for r in rounds:
            if r.get("n_train_after") is not None:
                ys.append(int(r["n_train_after"]))
            else:
                ys.append(ys[-1] + int(r.get("silver", 0)) + int(r.get("gold", 0)))
            xs.append(int(r.get("round", len(xs))))
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"seed {summary.get('seed')}")

    ax.set_xlabel("AL round (0 = initial labeled set)")
    ax.set_ylabel("Cumulative training set size")
    ax.set_title("Training set growth during active learning")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_epoch_curves(summaries: list[dict[str, Any]], output_path: Path) -> Path | None:
    """Validation Macro-F1 across epochs (final retrain of each saved proposed run)."""
    plt = _import_plt()
    _apply_thesis_style(plt)
    series: list[tuple[str, list[dict[str, Any]]]] = []

    for summary in summaries:
        if summary.get("method") != "proposed":
            continue
        history: list[dict[str, Any]] | None = None
        init = summary.get("initial_train") or {}
        if isinstance(init.get("history"), list) and init["history"]:
            history = init["history"]
        rounds = summary.get("rounds") or []
        for rnd in reversed(rounds):
            eh = rnd.get("epoch_history")
            if isinstance(eh, list) and eh:
                history = eh
                break
        if history:
            series.append((f"seed {summary.get('seed')}", history))

    if not series:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_f1, ax_loss = axes

    for label, history in series:
        epochs = [int(h["epoch"]) for h in history]
        val_f1 = [float(h.get("val_macro_f1", float("nan"))) for h in history]
        train_loss = [float(h.get("train_loss", float("nan"))) for h in history]
        ax_f1.plot(epochs, val_f1, marker="o", linewidth=2, label=label)
        ax_loss.plot(epochs, train_loss, marker="s", linewidth=2, label=label)

    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Validation Macro-F1")
    ax_f1.set_title("Validation score per epoch (last retrain)")
    ax_f1.grid(True, linestyle="--", alpha=0.35)
    ax_f1.legend()

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training loss per epoch (last retrain)")
    ax_loss.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Within-retrain learning dynamics (from saved summaries)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_al_validation_trajectory(
    summaries: list[dict[str, Any]], output_path: Path
) -> Path | None:
    """Best validation Macro-F1 after each AL retrain (when saved in run summaries)."""
    plt = _import_plt()
    _apply_thesis_style(plt)
    series: list[tuple[str, list[tuple[int, float]]]] = []
    for summary in summaries:
        if summary.get("method") != "proposed":
            continue
        points: list[tuple[int, float]] = []
        init = summary.get("initial_train") or {}
        if init.get("best_val_macro_f1") is not None:
            points.append((0, float(init["best_val_macro_f1"])))
        for rnd in summary.get("rounds") or []:
            score = rnd.get("best_val_macro_f1")
            if score is not None:
                points.append((int(rnd.get("round", len(points))), float(score)))
        if len(points) >= 2:
            series.append((f"seed {summary.get('seed')}", points))
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in series:
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=label)
    ax.set_xlabel("AL round (0 = after initial train)")
    ax.set_ylabel("Best validation Macro-F1 (within 3 epochs)")
    ax.set_title("Validation quality across active learning rounds")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_llm_utilization(df: pd.DataFrame, output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    data = dedupe_results(df)
    prop = data[data["method"] == "proposed"]
    if prop.empty or "llm_calls" not in prop.columns:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 5))
    seeds = [str(int(s)) for s in prop["seed"]]
    calls = prop["llm_calls"].astype(float).tolist()
    accepted = prop["llm_labels_accepted"].astype(float).tolist()
    rate = [a / c if c else 0.0 for a, c in zip(accepted, calls)]

    x = range(len(seeds))
    ax1.bar([i - 0.2 for i in x], calls, width=0.4, label="LLM API calls", color="#8172B2")
    ax1.bar([i + 0.2 for i in x], accepted, width=0.4, label="Silver labels accepted", color="#55A868")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("Random seed")
    ax1.set_ylabel("Count")
    ax1.set_title("LLM labeling throughput (proposed method)")

    ax2 = ax1.twinx()
    ax2.plot(list(x), rate, color="#CCB974", marker="D", linewidth=2, label="Acceptance rate")
    ax2.set_ylabel("Silver acceptance rate")
    ax2.set_ylim(0.0, 1.05)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(output_dir: Path, output_path: Path, *, method: str = "proposed") -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob(f"predictions_{method}_seed*.csv"))
    if not paths:
        return None

    frames = [pd.read_csv(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    if "true_label" not in merged.columns or "pred_label" not in merged.columns:
        return None

    y_true = merged["true_label"].astype(int)
    y_pred = merged["pred_label"].astype(int)
    labels = [0, 1]
    matrix = pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["pred"]).reindex(
        index=labels, columns=labels, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({method}, pooled test predictions)")
    for i in range(2):
        for j in range(2):
            val = int(matrix.values[i, j])
            ax.text(j, i, f"{val:,}", ha="center", va="center", color="black" if val < matrix.values.max() / 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bakeoff_llm_probe(bakeoff_root: Path, output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    path = Path(bakeoff_root) / "llm_probe" / "llm_probe_results.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ranking = data.get("ranking") or []
    rows = [r for r in ranking if r.get("status") == "completed" and r.get("accuracy") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (float(r["accuracy"]), float(r.get("mean_confidence") or 0)), reverse=True)
    models = [str(r["model"]).split("/")[-1][:28] for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    conf = [float(r.get("mean_confidence") or 0) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(rows))))
    y = range(len(rows))
    ax.barh(list(y), acc, color="#4C72B0", alpha=0.85, label="Probe accuracy (%)")
    ax.scatter(conf, list(y), color="#DD8452", s=70, zorder=3, label="Mean confidence (%)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.set_xlabel("Percent on 20-sample gold probe")
    ax.set_title("LLM bakeoff: probe accuracy (bars) vs mean confidence (dots)")
    ax.set_xlim(0, 102)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bakeoff_llm_runtime(bakeoff_root: Path, output_path: Path) -> Path | None:
    plt = _import_plt()
    _apply_thesis_style(plt)
    path = Path(bakeoff_root) / "llm_probe" / "llm_probe_results.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        r
        for r in (data.get("ranking") or [])
        if r.get("status") == "completed" and r.get("runtime_seconds") is not None
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["runtime_seconds"]))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(rows))))
    labels = [str(r["model"]).split("/")[-1][:24] for r in rows]
    times = [float(r["runtime_seconds"]) / 60.0 for r in rows]
    ax.barh(labels, times, color="#8172B2")
    ax.set_xlabel("Wall-clock minutes (20 probe calls)")
    ax.set_title("LLM bakeoff: probe runtime")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_experiment_figures(
    output_dir: str | Path,
    *,
    results_path: str | Path | None = None,
) -> dict[str, str | None]:
    """Generate thesis figures under ``output_dir/figures/``."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = output_dir / "results.csv"
    df = load_results(results_path)
    if df is None:
        return {}

    summaries = collect_run_summaries(output_dir)
    paths = {
        "macro_f1_by_seed": plot_macro_f1_by_seed(df, fig_dir / "macro_f1_by_seed.png"),
        "improvement_by_seed": plot_improvement_by_seed(df, fig_dir / "improvement_by_seed.png"),
        "training_size_vs_f1": plot_training_size_vs_f1(df, fig_dir / "training_size_vs_f1.png"),
        "al_label_composition": plot_al_label_composition(
            summaries, fig_dir / "al_label_composition.png"
        ),
        "cumulative_training_size": plot_cumulative_training_size(
            summaries, fig_dir / "cumulative_training_size.png"
        ),
        "validation_epoch_curves": plot_validation_epoch_curves(
            summaries, fig_dir / "validation_epoch_curves.png"
        ),
        "al_validation_trajectory": plot_al_validation_trajectory(
            summaries, fig_dir / "al_validation_trajectory.png"
        ),
        "llm_utilization": plot_llm_utilization(df, fig_dir / "llm_utilization.png"),
        "confusion_proposed": plot_confusion_matrix(
            output_dir, fig_dir / "confusion_matrix_proposed.png", method="proposed"
        ),
    }
    return {k: str(v) if v else None for k, v in paths.items()}


def build_bakeoff_figures(bakeoff_root: str | Path = "outputs/bakeoff") -> dict[str, str | None]:
    bakeoff_root = Path(bakeoff_root)
    fig_dir = bakeoff_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "llm_probe_accuracy": plot_bakeoff_llm_probe(bakeoff_root, fig_dir / "llm_probe_accuracy.png"),
        "llm_probe_runtime": plot_bakeoff_llm_runtime(bakeoff_root, fig_dir / "llm_probe_runtime.png"),
    }
    return {k: str(v) if v else None for k, v in paths.items()}
