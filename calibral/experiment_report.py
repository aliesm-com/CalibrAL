"""Presentation report for one experiment folder (all methods x seeds).

Writes ``EXPERIMENT_REPORT.md`` (latest), a dated ``EXPERIMENT_REPORT_<date_time>.md``
and an archive ``history/<date_time>/`` with the figures, from files the run
already produced (no GPU, no API):

- ``results.csv``                       metrics, costs, runtimes per method x seed
- ``predictions_<method>_seed<s>.csv``  paired bootstrap between methods
- ``llm_labels_<method>_seed<s>.csv``   LLM accuracy on hard vs random samples

Regenerate any time:  python run.py report --config <config>
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from calibral.utils import get_logger

BLUE = "#2a78d6"    # proposed
ORANGE = "#eb6834"  # best overall / highlighted alternative
AQUA = "#1baf7a"
NEUTRAL = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

METHOD_INFO: dict[str, tuple[str, str]] = {
    "baseline": ("Random AL, humans only", "reference (thesis baseline)"),
    "uncertainty_gold": ("Uncertainty AL, humans only", "is the LLM needed?"),
    "random_silver": ("Random AL + LLM silver", "is uncertainty sampling needed?"),
    "proposed": ("Uncertainty AL + calibrated LLM silver", "the method"),
    "llm_distill": ("LLM labels only (0 human)", "are humans needed?"),
    "llm_zeroshot": ("LLM predicts test directly", "why train a classifier?"),
}
ORDER = list(METHOD_INFO)
CLASSIFIER_METHODS = [m for m in ORDER if m != "llm_zeroshot"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _load(output_dir: Path) -> pd.DataFrame | None:
    from calibral.reporting import dedupe_results, load_results

    df = load_results(output_dir / "results.csv")
    if df is None:
        return None
    df = dedupe_results(df)
    if df.empty:
        return None
    df["seed"] = df["seed"].astype(int)
    return df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _method_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [x for x in ORDER if x in set(df["method"])] + sorted(set(df["method"]) - set(ORDER)):
        sub = df[df["method"] == m]
        f1 = _num(sub["macro_f1"])

        def mean(col: str) -> float | None:
            if col not in sub.columns:
                return None
            v = _num(sub[col]).dropna()
            return float(v.mean()) if len(v) else None

        rows.append(
            {
                "method": m,
                "seeds": len(sub),
                "f1": float(f1.mean()),
                "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "human": mean("human_labels_used"),
                "llm_calls": mean("llm_calls"),
                "n_train": mean("n_train"),
                "llm_usd": mean("llm_cost_usd"),
                "human_usd": mean("human_cost_usd"),
                "total_usd": mean("total_cost_usd"),
                "runtime_s": mean("runtime_seconds"),
                "silver_acc": mean("silver_label_accuracy"),
                "rejected_acc": mean("rejected_llm_accuracy"),
                "sel_cov": mean("llm_selective_coverage"),
                "sel_acc": mean("llm_selective_accuracy"),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def _preds(output_dir: Path, method: str, seed: int) -> pd.DataFrame | None:
    p = output_dir / f"predictions_{method}_seed{seed}.csv"
    return pd.read_csv(p) if p.exists() else None


def compare(df: pd.DataFrame, output_dir: Path, a: str, b: str, n_resamples: int = 1000) -> dict[str, Any] | None:
    """Paired comparison a - b: per-seed deltas, paired t-test, per-seed bootstrap."""
    from scipy import stats

    from calibral.significance import paired_bootstrap

    da = df[df["method"] == a].set_index("seed")["macro_f1"]
    db = df[df["method"] == b].set_index("seed")["macro_f1"]
    seeds = sorted(set(da.index) & set(db.index))
    if not seeds:
        return None
    deltas = np.array([float(da[s]) - float(db[s]) for s in seeds])
    p_t = None
    if len(deltas) >= 2 and np.any(deltas != deltas[0]):
        p_t = float(stats.ttest_1samp(deltas, 0.0).pvalue)
    a_wins = b_wins = 0
    for s in seeds:
        pa, pb = _preds(output_dir, a, s), _preds(output_dir, b, s)
        if pa is None or pb is None:
            continue
        m = pa.merge(pb, on="sample_id", suffixes=("_a", "_b"))
        if m.empty:
            continue
        bt = paired_bootstrap(
            m["true_label_a"].to_numpy(), m["pred_label_a"].to_numpy(),
            m["pred_label_b"].to_numpy(), n_resamples=n_resamples, seed=int(s),
        )
        if bt["ci95_low"] > 0:
            a_wins += 1
        elif bt["ci95_high"] < 0:
            b_wins += 1
    return {
        "a": a,
        "b": b,
        "n": len(seeds),
        "delta": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "p_t": p_t,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "significant": p_t is not None and p_t < 0.05,
    }


def llm_difficulty(output_dir: Path, method: str = "proposed") -> dict[str, dict[str, float]] | None:
    """LLM accuracy / confidence / ECE on high-entropy (AL) vs random picks, pooled over seeds."""
    files = sorted(output_dir.glob(f"llm_labels_{method}_seed*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[~df["llm_error"].astype(bool)]
    out = {}
    for phase, name in (("al_round", "hard"), ("silver_expansion", "random")):
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        conf = sub["llm_confidence"].astype(float).to_numpy()
        corr = sub["llm_correct"].astype(bool).to_numpy()
        bins = np.linspace(0.5, 1.0, 6)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
        ece = 0.0
        for k in range(len(bins) - 1):
            mask = idx == k
            if mask.any():
                ece += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
        out[name] = {
            "n": int(len(sub)),
            "acc": float(corr.mean()),
            "conf": float(conf.mean()),
            "accept": float(sub["accepted_as_silver"].astype(bool).mean()),
            "ece": float(ece),
        }
    return out or None


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150, "savefig.dpi": 200, "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10.5,
            "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.labelcolor": INK_2, "axes.edgecolor": NEUTRAL, "axes.spines.top": False,
            "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": INK_2,
            "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
        }
    )
    return plt


def _label(m: str) -> str:
    return f"{m}\n{METHOD_INFO.get(m, (m, ''))[0]}"


def _color(m: str, best: str | None) -> str:
    if m == "proposed":
        return BLUE
    if m == best:
        return ORANGE
    return NEUTRAL


def plot_f1(table: pd.DataFrame, df: pd.DataFrame, best: str | None, out: Path) -> Path:
    plt = _plt()
    methods = list(table.index)
    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(methods) + 1.4))
    for i, m in enumerate(methods):
        vals = 100 * _num(df[df["method"] == m]["macro_f1"]).to_numpy()
        c = _color(m, best)
        ax.scatter(vals, [i] * len(vals), s=22, color=c if c != NEUTRAL else MUTED, alpha=0.45, zorder=2)
        mu, sd = 100 * table.loc[m, "f1"], 100 * table.loc[m, "f1_std"]
        ax.plot([mu - sd, mu + sd], [i, i], color=c if c != NEUTRAL else MUTED, lw=2.2, zorder=3,
                solid_capstyle="round")
        ax.scatter([mu], [i], s=95, color=c if c != NEUTRAL else INK_2, edgecolor=SURFACE,
                   linewidth=2, zorder=4)
        ax.text(mu, i - 0.32, f"{mu:.2f}", ha="center", fontsize=9, color=INK if c != NEUTRAL else INK_2)
    if "baseline" in table.index:
        ax.axvline(100 * table.loc["baseline", "f1"], color=MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_label(m) for m in methods], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Test Macro-F1 (%) — big dot = mean, line = ±1 std, small dots = seeds; dashed = baseline")
    ax.set_title("Test Macro-F1 by method")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost(table: pd.DataFrame, best: str | None, out: Path) -> Path | None:
    t = table.dropna(subset=["total_usd"])
    if len(t) < 2:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = np.maximum(t["total_usd"].to_numpy(dtype=float), 0.01)
    ys = 100 * t["f1"].to_numpy(dtype=float)
    yspan = max(ys.max() - ys.min(), 0.5)
    placed: list[tuple[float, float]] = []
    for m, x, y in sorted(zip(t.index, xs, ys), key=lambda z: -z[2]):
        c = _color(m, best)
        ax.scatter([x], [y], s=130 if c != NEUTRAL else 80, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        # push the label down when another point sits right above it (same cost, close F1)
        crowded = any(abs(np.log10(x / px)) < 0.15 and abs(y - py) < 0.08 * yspan for px, py in placed)
        placed.append((x, y))
        ax.annotate(m, (x, y), xytext=(8, -13 if crowded else 6), textcoords="offset points", fontsize=9,
                    color=INK if c != NEUTRAL else INK_2, fontweight="bold" if c != NEUTRAL else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Total labeling cost per run (USD, log scale): human training labels + LLM API")
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_title("Cost vs quality — up and to the left is better")
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta(comps: list[dict[str, Any]], out: Path) -> Path | None:
    comps = [c for c in comps if c]
    if not comps:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(comps) + 1.4))
    for i, c in enumerate(comps):
        mu, sd = 100 * c["delta"], 100 * c["delta_std"]
        col = BLUE if c["significant"] and mu > 0 else (ORANGE if c["significant"] else NEUTRAL)
        ax.plot([mu - sd, mu + sd], [i, i], color=col, lw=2.2, solid_capstyle="round")
        ax.scatter([mu], [i], s=90, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        p = c["p_t"]
        ptxt = "p<0.001" if p is not None and p < 0.001 else (f"p={p:.3f}" if p is not None else "")
        ax.text(mu + sd + 0.05, i, f"{mu:+.2f} pts  {ptxt}", va="center", fontsize=9, color=INK_2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([f"{c['a']} − {c['b']}" for c in comps], fontsize=9.5)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ Macro-F1 (points), mean ± std over paired seeds · blue = significant gain (paired t, p<0.05)")
    ax.set_title("Head-to-head: proposed vs every other method")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_difficulty(diff: dict[str, dict[str, float]], gate: tuple[float, float] | None, out: Path) -> Path | None:
    if not diff or "hard" not in diff or "random" not in diff:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    groups = ["LLM accuracy", "LLM mean confidence", "Accepted as silver"]
    keys = ["acc", "conf", "accept"]
    x = np.arange(len(groups))
    w = 0.36
    for j, (name, col, lab) in enumerate((("hard", BLUE, "High-entropy picks (AL rounds)"),
                                          ("random", ORANGE, "Random picks (silver expansion)"))):
        vals = [100 * diff[name][k] for k in keys]
        ax.bar(x + (j - 0.5) * w, vals, width=w - 0.04, color=col, label=f"{lab} · n={diff[name]['n']:,}")
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    title = "The LLM is much less reliable on the samples AL selects"
    if gate:
        title += f"\n(confidence gate: accepted {gate[0]:.1f}% correct vs rejected {gate[1]:.1f}%)"
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _badge(label: str, value: str, color: str) -> str:
    def esc(s: str) -> str:
        return quote(str(s).replace("-", "--").replace("_", "__"), safe="")

    return (f"![{label}: {value}](https://img.shields.io/badge/{esc(label)}-{esc(value)}-"
            f"{color.lstrip('#')}?style=for-the-badge)")


def _bar(v: float, lo: float, hi: float, width: int = 14) -> str:
    frac = 1.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    full = frac * width
    blocks = "▏▎▍▌▋▊▉█"
    n = int(full)
    rem = full - n
    return "█" * n + (blocks[int(rem * 8) - 1] if rem >= 0.125 else "")


def _p(p: float | None) -> str:
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _usd(x: float | None, d: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:,.{d}f}"


def _n(x: float | None) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}"


def _dur(s: float | None) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "—"
    s = int(s)
    h, r = divmod(s, 3600)
    m = r // 60
    return f"{h}h {m}m" if h else f"{m}m"


def _verdict(c: dict[str, Any] | None) -> str:
    if not c:
        return "—"
    if c["significant"]:
        return "✅ better" if c["delta"] > 0 else "❌ worse"
    return "🤝 tied"


def render(ctx: dict[str, Any], figs: dict[str, str | None]) -> str:
    t: pd.DataFrame = ctx["table"]
    cfg = ctx["config"]
    df = ctx["df"]
    best = ctx["best"]
    best_clf = ctx["best_clf"]
    comps = ctx["comps"]
    by_b = {c["b"]: c for c in comps if c}
    diff = ctx["difficulty"]
    data_name = (cfg.get("data") or {}).get("dataset_name", "?")
    encoder = df["model_name"].dropna().iloc[0] if "model_name" in df else "?"
    llm = df["llm_model"].dropna().iloc[0] if "llm_model" in df else "?"
    seeds = sorted(df["seed"].unique().tolist())
    n_test = None
    L: list[str] = []

    L += [
        '<div align="center">',
        "",
        "# 📊 Experiment Results",
        "",
        f"**{data_name.upper()}** · encoder `{encoder}` · LLM `{llm}` · {len(seeds)} seeds · {ctx['when']}",
        "",
        "<br/>",
        "",
        "| 🏆 Best overall | 🎯 Best trained classifier | 🧪 Proposed vs baseline |",
        "|:---:|:---:|:---:|",
        f"| <h2>{best}</h2> | <h2>{best_clf}</h2> | <h2>{100 * by_b['baseline']['delta']:+.2f} pts</h2> |"
        if "baseline" in by_b else f"| <h2>{best}</h2> | <h2>{best_clf}</h2> | <h2>—</h2> |",
        f"| {100 * t.loc[best, 'f1']:.2f}% Macro-F1 | {100 * t.loc[best_clf, 'f1']:.2f}% Macro-F1 | "
        + (f"paired t p={_p(by_b['baseline']['p_t'])} · {by_b['baseline']['a_wins']}/{by_b['baseline']['n']} seeds sig. |"
           if "baseline" in by_b else "— |"),
        "",
    ]
    if "proposed" in t.index:
        L.append(" ".join([
            _badge("proposed F1", f"{100 * t.loc['proposed', 'f1']:.2f}%", BLUE),
            _badge("human labels", _n(t.loc["proposed", "human"]), "52514e"),
            _badge("LLM cost/run", _usd(t.loc["proposed", "llm_usd"]), ORANGE),
        ]))
    L += ["", "</div>", "", "---", ""]

    # podium
    top = list(t.sort_values("f1", ascending=False).index[:3])
    medals = ["🥇", "🥈", "🥉"]
    L += ["## 🏅 Podium (mean Macro-F1 over seeds)", "",
          "| | " + " | ".join(medals[: len(top)]) + " |", "|---|" + "|".join([":---:"] * len(top)) + "|",
          "| **Method** | " + " | ".join(f"**{m}**" for m in top) + " |",
          "| What it is | " + " | ".join(METHOD_INFO.get(m, (m, ""))[0] for m in top) + " |",
          "| Macro-F1 | " + " | ".join(f"{100 * t.loc[m, 'f1']:.2f} ± {100 * t.loc[m, 'f1_std']:.2f}" for m in top) + " |",
          "| Human labels | " + " | ".join(_n(t.loc[m, "human"]) for m in top) + " |",
          "| Total $ / run | " + " | ".join(_usd(t.loc[m, "total_usd"]) for m in top) + " |", ""]

    # findings
    L += ["## 💡 Key findings", ""]
    L += [f"- {x}" for x in ctx["findings"]]
    L += ["", "### 📝 Suggested paper story", "", f"> {ctx['story']}", ""]

    # leaderboard
    lo = 100 * t["f1"].min() - 1.5
    L += ["## 📋 Leaderboard", "",
          "| Method | Question it answers | Macro-F1 | | Proposed is… | Human labels | LLM calls | Train size | LLM $ | Total $ | Time / run |",
          "|---|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for m in t.sort_values("f1", ascending=False).index:
        r = t.loc[m]
        name = f"**{m}** 🧪" if m == "proposed" else (f"{m} 🏆" if m == best else m)
        L.append(
            f"| {name} | {METHOD_INFO.get(m, ('', '—'))[1]} | `{_bar(100 * r['f1'], lo, 100 * t['f1'].max())}` | "
            f"{100 * r['f1']:.2f} ± {100 * r['f1_std']:.2f} | "
            f"{'—' if m == 'proposed' else _verdict(by_b.get(m))} | {_n(r['human'])} | {_n(r['llm_calls'])} | "
            f"{_n(r['n_train'])} | {_usd(r['llm_usd'])} | {_usd(r['total_usd'])} | {_dur(r['runtime_s'])} |"
        )
    L += ["", "🧪 proposed · 🏆 best overall · *Proposed is…* = paired t-test over seeds (p<0.05): "
          "is the proposed method better than, worse than, or tied with that row?", ""]

    # head-to-head
    if comps:
        L += ["## ⚔️ Head-to-head: proposed vs …", "",
              "| Opponent | Δ Macro-F1 (pts) | paired t p | seeds where proposed sig. better | seeds where opponent sig. better | Verdict |",
              "|---|---:|---:|:---:|:---:|:---:|"]
        for c in comps:
            if not c:
                continue
            L.append(f"| {c['b']} | {100 * c['delta']:+.2f} ± {100 * c['delta_std']:.2f} | {_p(c['p_t'])} | "
                     f"{c['a_wins']}/{c['n']} | {c['b_wins']}/{c['n']} | {_verdict(c)} |")
        L += ["", "<sub>Per-seed significance: paired bootstrap over test items (95% CI excludes 0).</sub>", ""]

    # LLM difficulty
    if diff and "hard" in diff and "random" in diff:
        h, r = diff["hard"], diff["random"]
        L += ["## 🧠 LLM as annotator: hard vs random samples", "",
              "| Samples | n | LLM accuracy | mean confidence | ECE | accepted as silver |",
              "|---|---:|---:|---:|---:|---:|",
              f"| High-entropy picks (AL rounds) | {h['n']:,} | {100 * h['acc']:.1f}% | {100 * h['conf']:.1f}% | {h['ece']:.3f} | {100 * h['accept']:.1f}% |",
              f"| Random picks (silver expansion) | {r['n']:,} | {100 * r['acc']:.1f}% | {100 * r['conf']:.1f}% | {r['ece']:.3f} | {100 * r['accept']:.1f}% |",
              "", "<sub>From `llm_labels_proposed_seed*.csv`, pooled over seeds. Gold labels are used here for evaluation only.</sub>", ""]

    # charts
    caps = {
        "f1": ("🎯 Macro-F1 by method", "Every seed is shown; the dashed line is the baseline."),
        "delta": ("⚔️ Proposed vs every other method", "Paired over seeds; blue = significant gain for proposed."),
        "cost": ("💵 Cost vs quality", "Log scale: LLM-only methods cost cents, human labels cost dollars."),
        "difficulty": ("🧠 LLM reliability on hard vs random samples", "Why a confidence gate is needed on AL picks."),
    }
    if any(figs.values()):
        L += ["## 📈 Charts", ""]
        for k, (title, cap) in caps.items():
            if figs.get(k):
                L += [f"### {title}", "", f"![{title}]({figs[k]})", "", f"<sub>{cap}</sub>", ""]

    # setup
    al = cfg.get("active_learning") or {}
    tr = cfg.get("train") or {}
    dc = cfg.get("data") or {}
    cost = cfg.get("cost") or {}
    L += ["<details>", "<summary><b>🧾 Setup & reproducibility</b></summary>", "",
          f"- Dataset `{data_name}` · initial {dc.get('initial_labeled_size')} · pool {dc.get('pool_size')} · "
          f"validation {dc.get('validation_size')} (gold, same for all methods) · test {dc.get('test_size')}",
          f"- Encoder `{encoder}` · max_length {(cfg.get('model') or {}).get('max_length')} · "
          f"epochs {tr.get('epochs')} · batch {tr.get('batch_size')} · lr {tr.get('learning_rate')}",
          f"- AL: budget {al.get('budget')} · query batch {al.get('query_batch_size')} · "
          f"silver threshold {al.get('silver_threshold')} · silver weight {al.get('silver_weight')} · "
          f"post-budget silver {al.get('post_budget_silver')}",
          f"- LLM `{llm}` · prompt {(cfg.get('llm') or {}).get('prompt_version')} · "
          f"reasoning {(cfg.get('llm') or {}).get('reasoning')}",
          f"- Seeds: {', '.join(map(str, seeds))}",
          f"- Human cost rate: ${cost.get('human_per_label_usd')} per label "
          "(placeholder unless you set a cited rate) · LLM $ = attributable API cost incl. cached labels",
          "- Tables: `significance.md` · raw: `results.csv`, `predictions_*.csv`, `llm_labels_*.csv`",
          "", "</details>", "", "---", "",
          "<sub>Generated by CalibrAL · regenerate (no GPU/API): <code>python run.py report --config "
          f"{cfg.get('_config_path', '<config>')}</code></sub>", ""]
    return "\n".join(L)


def _findings(ctx: dict[str, Any]) -> tuple[list[str], str]:
    t: pd.DataFrame = ctx["table"]
    by_b = {c["b"]: c for c in ctx["comps"] if c}
    F: list[str] = []

    def pts(c: dict[str, Any]) -> str:
        return f"{100 * c['delta']:+.2f} pts (p={_p(c['p_t'])}, {c['a_wins']}/{c['n']} seeds significant)"

    if "baseline" in by_b:
        c = by_b["baseline"]
        F.append(("✅" if c["significant"] and c["delta"] > 0 else "⚠️")
                 + f" **Proposed vs baseline:** {pts(c)} with the **same {_n(t.loc['proposed', 'human'])} human labels**.")
    if "random_silver" in by_b and "random_silver" in t.index:
        c = by_b["random_silver"]
        calls = t.loc["random_silver", "llm_calls"] / max(t.loc["proposed", "llm_calls"], 1)
        time_ratio = t.loc["random_silver", "runtime_s"] / max(t.loc["proposed", "runtime_s"], 1)
        if c["significant"]:
            F.append(("✅" if c["delta"] > 0 else "⚠️") + f" **Uncertainty sampling vs random:** {pts(c)}.")
        else:
            F.append(f"🤝 **Uncertainty sampling vs random selection (both with LLM silver):** no significant accuracy "
                     f"difference ({100 * c['delta']:+.2f} pts, p={_p(c['p_t'])}), but proposed needs "
                     f"**{calls:.1f}× fewer LLM calls** and **{time_ratio:.1f}× less compute time** "
                     f"(train set {_n(t.loc['proposed', 'n_train'])} vs {_n(t.loc['random_silver', 'n_train'])}).")
    if "llm_distill" in by_b:
        c = by_b["llm_distill"]
        if c["significant"] and c["delta"] > 0:
            F.append(f"👥 **Humans still add value:** proposed beats LLM-only distillation by {pts(c)}.")
        else:
            F.append(f"👥 **Humans add little here:** proposed vs LLM-only distillation {pts(c)}.")
        if "baseline" in t.index:
            d = t.loc["llm_distill", "f1"] - t.loc["baseline", "f1"]
            F.append(f"💸 **LLM-only distillation** (0 human labels, {_usd(t.loc['llm_distill', 'total_usd'])}/run) "
                     f"is {100 * d:+.2f} pts vs the human baseline ({_usd(t.loc['baseline', 'total_usd'])}/run).")
    if "uncertainty_gold" in by_b:
        c = by_b["uncertainty_gold"]
        F.append(f"🤖 **The LLM is what helps:** proposed vs uncertainty AL with humans only {pts(c)}"
                 + (f"; uncertainty_gold itself is unstable (std {100 * t.loc['uncertainty_gold', 'f1_std']:.2f})."
                    if t.loc["uncertainty_gold", "f1_std"] > 2 * max(t.loc["baseline", "f1_std"], 1e-9) else "."))
    if "proposed" in t.index and not math.isnan(t.loc["proposed", "silver_acc"] or float("nan")):
        F.append(f"🧭 **Confidence gate works:** accepted silver labels are {100 * t.loc['proposed', 'silver_acc']:.1f}% "
                 f"correct vs {100 * t.loc['proposed', 'rejected_acc']:.1f}% for answers sent to humans.")
    diff = ctx["difficulty"]
    if diff and "hard" in diff and "random" in diff:
        F.append(f"🧠 **AL picks are hard for the LLM:** {100 * diff['hard']['acc']:.1f}% accuracy on high-entropy "
                 f"samples vs {100 * diff['random']['acc']:.1f}% on random ones (paper-2 material).")
    story = "Not enough methods to suggest a story."
    if "llm_zeroshot" in t.index and "proposed" in t.index:
        zs, pr = t.loc["llm_zeroshot"], t.loc["proposed"]
        per_1k = (zs["llm_usd"] or 0) / max((zs["llm_calls"] or 1) / 1000, 1e-9)
        if zs["f1"] > pr["f1"]:
            F.append(f"🔮 **The LLM itself is more accurate:** zero-shot {100 * zs['f1']:.2f}% vs proposed "
                     f"{100 * pr['f1']:.2f}%, but costs ~{_usd(per_1k, 3)} per 1,000 predictions forever, "
                     "whereas the trained classifier runs locally for free. "
                     f"Selective LLM: {100 * (zs['sel_acc'] or 0):.1f}% accurate on the "
                     f"{100 * (zs['sel_cov'] or 0):.0f}% it is confident about.")
    rs = by_b.get("random_silver")
    di = by_b.get("llm_distill")
    zs_better = "llm_zeroshot" in t.index and t.loc["llm_zeroshot", "f1"] > t.loc["proposed", "f1"]
    parts = []
    if "baseline" in by_b and by_b["baseline"]["significant"] and by_b["baseline"]["delta"] > 0:
        parts.append("Under a fixed human budget, calibrated LLM silver labels significantly improve the classifier")
    if di and di["significant"] and di["delta"] > 0:
        parts.append("human labels on the uncertain samples still matter (LLM-only distillation is worse)")
    if rs and not rs["significant"]:
        parts.append("uncertainty sampling matches random selection in accuracy while using fewer LLM labels and "
                     "less compute — frame it as an efficiency gain, not an accuracy gain")
    if zs_better:
        parts.append("the LLM alone is more accurate, so the case for the classifier is inference cost and "
                     "independence from the API — make that comparison explicit")
    if parts:
        story = "; ".join(parts) + "."
    return F, story


def build_experiment_report(output_dir: str | Path, config: dict[str, Any] | None = None) -> Path | None:
    """Write EXPERIMENT_REPORT.md (+ dated copy and history archive) for one experiment folder."""
    output_dir = Path(output_dir)
    df = _load(output_dir)
    if df is None:
        return None
    config = config or {}
    table = _method_table(df)
    best = str(table["f1"].idxmax())
    clf = table.loc[[m for m in table.index if m in CLASSIFIER_METHODS]] if any(
        m in CLASSIFIER_METHODS for m in table.index) else table
    best_clf = str(clf["f1"].idxmax())
    comps = []
    if "proposed" in table.index:
        for other in [m for m in ORDER if m in table.index and m != "proposed"]:
            comps.append(compare(df, output_dir, "proposed", other))
    now = datetime.now()
    ctx = {
        "df": df, "table": table, "config": config, "best": best, "best_clf": best_clf, "comps": comps,
        "difficulty": llm_difficulty(output_dir), "when": now.strftime("%Y-%m-%d %H:%M"),
    }
    ctx["findings"], ctx["story"] = _findings(ctx)

    stamp = now.strftime("%Y-%m-%d_%H%M")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gate = None
    if "proposed" in table.index and table.loc["proposed", "silver_acc"] is not None:
        try:
            gate = (100 * float(table.loc["proposed", "silver_acc"]), 100 * float(table.loc["proposed", "rejected_acc"]))
        except (TypeError, ValueError):
            gate = None
    makers = {
        "f1": lambda p: plot_f1(table, df, best, p),
        "delta": lambda p: plot_delta(comps, p),
        "cost": lambda p: plot_cost(table, best, p),
        "difficulty": lambda p: plot_difficulty(ctx["difficulty"] or {}, gate, p),
    }
    rel: dict[str, str | None] = {}
    for key, fn in makers.items():
        name = f"report_{key}.png"
        try:
            rel[key] = f"figures/{name}" if fn(fig_dir / name) else None
        except Exception as exc:  # noqa: BLE001 - a chart must not block the report
            get_logger("reporting").warning("Report chart %s failed: %s", key, exc)
            rel[key] = None

    latest = output_dir / "EXPERIMENT_REPORT.md"
    latest.write_text(render(ctx, rel), encoding="utf-8")
    hist = output_dir / "history" / stamp
    (hist / "figures").mkdir(parents=True, exist_ok=True)
    for r in rel.values():
        if r:
            shutil.copy2(output_dir / r, hist / "figures" / Path(r).name)
    (hist / "EXPERIMENT_REPORT.md").write_text(
        render(ctx, {k: (f"figures/{Path(v).name}" if v else None) for k, v in rel.items()}), encoding="utf-8")
    dated = output_dir / f"EXPERIMENT_REPORT_{stamp}.md"
    dated.write_text(
        render(ctx, {k: (f"history/{stamp}/figures/{Path(v).name}" if v else None) for k, v in rel.items()}),
        encoding="utf-8")
    (hist / "summary.json").write_text(json.dumps({
        "generated_at": now.isoformat(timespec="seconds"), "best_overall": best, "best_classifier": best_clf,
        "story": ctx["story"], "findings": ctx["findings"],
        "comparisons": [c for c in comps if c],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    get_logger("reporting").info("Experiment report written | latest=%s | dated=%s", latest, dated)
    return dated
