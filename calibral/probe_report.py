"""Presentation report for the LLM probe: winner banner, leaderboard, charts.

Reads ``<bakeoff_root>/llm_probe/llm_probe_results.json`` and the per-sample
``predictions/*.jsonl`` files, writes four figures to
``<bakeoff_root>/llm_probe/figures/`` and ``<bakeoff_root>/PROBE_REPORT.md``.

Needs no API calls, so it can be regenerated at any time:
``python run.py bakeoff --stage probe-report --config configs/paper_al.yaml``.

Statistics added on top of the probe results:
- exact McNemar test (paired, same samples) of every model vs the winner,
- "statistically tied with the winner" = McNemar p >= 0.05,
- "best value" = cheapest model (per 1k labels) that is tied with the winner.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from calibral.utils import get_logger

# Validated categorical slots 1-3 (all-pairs, light surface) + chart chrome.
BLUE = "#2a78d6"    # winner
ORANGE = "#eb6834"  # best value
AQUA = "#1baf7a"    # third series (reliability diagram only; direct-labeled)
SERIES = [BLUE, ORANGE, AQUA]
NEUTRAL = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _short(model: str) -> str:
    return str(model).split("/")[-1]


def _slug(model: str) -> str:
    return str(model).replace("/", "__").replace(":", "_")


def _load_records(pred_dir: Path, model: str) -> list[dict[str, Any]]:
    path = pred_dir / f"{_slug(model)}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> float | None:
    """Two-sided exact McNemar p-value on paired correctness vectors."""
    if len(a_correct) != len(b_correct) or not a_correct:
        return None
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _paired_correct(
    winner_recs: list[dict[str, Any]], recs: list[dict[str, Any]]
) -> tuple[list[bool], list[bool]]:
    by_id = {r["sample_id"]: bool(r.get("correct")) for r in recs}
    a, b = [], []
    for r in winner_recs:
        if r["sample_id"] in by_id:
            a.append(bool(r.get("correct")))
            b.append(by_id[r["sample_id"]])
    return a, b


def analyse(bakeoff_root: Path) -> dict[str, Any] | None:
    probe_dir = bakeoff_root / "llm_probe"
    path = probe_dir / "llm_probe_results.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    ranking = list(summary.get("ranking") or [])
    completed = [r for r in ranking if r.get("status") == "completed"]
    failed = [r for r in ranking if r.get("status") != "completed"]
    if not completed:
        return {"summary": summary, "completed": [], "failed": failed, "winner": None}

    pred_dir = probe_dir / "predictions"
    records = {r["model"]: _load_records(pred_dir, r["model"]) for r in ranking}
    winner = completed[0]
    w_recs = records.get(winner["model"]) or []
    for r in completed:
        if r is winner:
            r["_p_vs_winner"] = None
            r["_tied"] = True
            continue
        a, b = _paired_correct(w_recs, records.get(r["model"]) or [])
        p = mcnemar_exact(a, b)
        r["_p_vs_winner"] = p
        r["_tied"] = p is not None and p >= 0.05

    priced = [r for r in completed if r.get("_tied") and r.get("cost_per_1k_labels_usd")]
    best_value = min(priced, key=lambda r: float(r["cost_per_1k_labels_usd"])) if priced else None
    calibrated = [r for r in completed if r.get("ece") is not None]
    best_cal = min(calibrated, key=lambda r: float(r["ece"])) if calibrated else None
    fastest = min(completed, key=lambda r: float(r.get("runtime_seconds") or 1e12))
    with_cost = [r for r in completed if r.get("cost_per_1k_labels_usd")]
    cheapest = min(with_cost, key=lambda r: float(r["cost_per_1k_labels_usd"])) if with_cost else None
    priciest = max(with_cost, key=lambda r: float(r["cost_per_1k_labels_usd"])) if with_cost else None
    return {
        "summary": summary,
        "completed": completed,
        "failed": failed,
        "records": records,
        "winner": winner,
        "best_value": best_value,
        "best_calibrated": best_cal,
        "fastest": fastest,
        "cheapest": cheapest,
        "priciest": priciest,
        "tied": [r for r in completed if r.get("_tied") and r is not winner],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": NEUTRAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": INK_2,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    return plt


def _color_for(row: dict[str, Any], a: dict[str, Any]) -> str:
    if row is a["winner"]:
        return BLUE
    if a.get("best_value") is not None and row is a["best_value"]:
        return ORANGE
    return NEUTRAL


def plot_accuracy(a: dict[str, Any], out: Path) -> Path:
    plt = _plt()
    rows = a["completed"]
    n = a["summary"].get("n_probe_samples")
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(rows) + 1.6))
    w = a["winner"]
    lo_w, hi_w = (w.get("accuracy_ci95") or [None, None])
    if lo_w is not None:
        ax.axvspan(100 * lo_w, 100 * hi_w, color=BLUE, alpha=0.07, lw=0, zorder=0)
    for i, r in enumerate(rows):
        acc = 100 * float(r["accuracy"])
        ci = r.get("accuracy_ci95") or [r["accuracy"], r["accuracy"]]
        color = _color_for(r, a)
        ax.plot([100 * ci[0], 100 * ci[1]], [i, i], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([acc], [i], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(100 * ci[1] + 0.4, i, f"{acc:.1f}%", va="center", fontsize=9.5,
                color=INK if color != NEUTRAL else INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_short(r["model"]) for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r is w:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    lo = min(100 * float((r.get("accuracy_ci95") or [r["accuracy"]])[0]) for r in rows)
    ax.set_xlim(max(0, lo - 3), 102)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Accuracy on {n} gold samples (dot) with 95% Wilson CI (line)")
    ax.set_title("Probe accuracy — shaded band = winner's 95% CI")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_accuracy(a: dict[str, Any], out: Path) -> Path | None:
    rows = [r for r in a["completed"] if r.get("cost_per_1k_labels_usd")]
    if len(rows) < 2:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = [float(r["cost_per_1k_labels_usd"]) for r in rows]
    ys = [100 * float(r["accuracy"]) for r in rows]
    # Pareto frontier: no other model is both cheaper and more accurate.
    order = sorted(range(len(rows)), key=lambda i: (xs[i], -ys[i]))
    frontier, best = [], -1.0
    for i in order:
        if ys[i] > best:
            frontier.append(i)
            best = ys[i]
    ax.step([xs[i] for i in frontier], [ys[i] for i in frontier], where="post",
            color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1, label="Pareto frontier")
    for r, x, y in zip(rows, xs, ys):
        color = _color_for(r, a)
        size = 120 if color != NEUTRAL else 70
        ax.scatter([x], [y], s=size, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        weight = "bold" if color != NEUTRAL else "normal"
        ax.annotate(_short(r["model"]), (x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=8.8, color=INK if color != NEUTRAL else INK_2, fontweight=weight)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 labels (USD, log scale) — from OpenRouter usage.cost")
    ax.set_ylabel("Probe accuracy (%)")
    ax.set_title("Cost vs accuracy — up and to the left is better")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="Winner"),
    ]
    if a.get("best_value") is not None and a["best_value"] is not a["winner"]:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=9, label="Best value"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color=NEUTRAL, ms=8, label="Other models"))
    handles.append(plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="Pareto frontier"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _reliability_points(recs: list[dict[str, Any]], edges: list[float]):
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [
            r for r in recs
            if r.get("pred") is not None and r.get("confidence") is not None
            and (lo <= float(r["confidence"]) < hi or (hi == edges[-1] and float(r["confidence"]) == hi))
        ]
        if b:
            conf = sum(float(r["confidence"]) for r in b) / len(b)
            acc = sum(1 for r in b if r.get("correct")) / len(b)
            pts.append((conf, acc, len(b)))
    return pts


def plot_reliability(a: dict[str, Any], out: Path) -> Path | None:
    top = [r for r in a["completed"] if a["records"].get(r["model"])][:3]
    if not top:
        return None
    plt = _plt()
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1,
            label="Perfect calibration")
    markers = ["o", "s", "^"]
    all_pts = {r["model"]: _reliability_points(a["records"][r["model"]], edges) for r in top}
    max_n = max((n for pts in all_pts.values() for _, _, n in pts), default=1)
    for idx, r in enumerate(top):
        pts = all_pts[r["model"]]
        if not pts:
            continue
        xs, ys, ns = zip(*pts)
        # small horizontal offset so overlapping series stay visible
        xs = [x + (idx - 1) * 0.004 for x in xs]
        ax.plot(xs, ys, color=SERIES[idx], lw=2, zorder=2)
        ax.scatter(xs, ys, s=[30 + 140 * math.sqrt(n / max_n) for n in ns], marker=markers[idx],
                   color=SERIES[idx], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"#{idx + 1} {_short(r['model'])} · ECE {float(r.get('ece') or 0):.3f}")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Stated confidence (bin mean)\nmarker area ∝ samples in bin · below the diagonal = overconfident")
    ax.set_ylabel("Observed accuracy in bin")
    ax.set_title("Reliability of LLM confidence — top 3 models")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.8, markerscale=0.7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ece(a: dict[str, Any], out: Path) -> Path | None:
    rows = [r for r in a["completed"] if r.get("ece") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["ece"]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.5))
    vals = [float(r["ece"]) for r in rows]
    colors = [_color_for(r, a) for r in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_short(r["model"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.15 or 1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Expected Calibration Error (lower = confidence matches accuracy)")
    ax.set_title("Calibration of stated confidence")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _pct(x: Any, d: int = 1) -> str:
    return "—" if x is None else f"{100 * float(x):.{d}f}%"


def _money(x: Any, d: int = 4) -> str:
    return "—" if x is None else f"${float(x):.{d}f}"


def _badge(label: str, value: str, color: str) -> str:
    def esc(s: str) -> str:
        return quote(str(s).replace("-", "--").replace("_", "__"), safe="")

    return (
        f"![{label}: {value}](https://img.shields.io/badge/"
        f"{esc(label)}-{esc(value)}-{color.lstrip('#')}?style=for-the-badge)"
    )


def _bar(value: float, lo: float, hi: float, width: int = 16) -> str:
    if hi <= lo:
        frac = 1.0
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    full = frac * width
    blocks = "▏▎▍▌▋▊▉█"
    n_full = int(full)
    rem = full - n_full
    tail = blocks[int(rem * 8) - 1] if rem >= 0.125 else ""
    return "█" * n_full + tail


def _p(p: Any) -> str:
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def render_markdown(a: dict[str, Any], fig_rel: dict[str, str | None]) -> str:
    s = a["summary"]
    n = s.get("n_probe_samples")
    when = s.get("finished_at", "")
    try:
        when = datetime.fromisoformat(when).astimezone().strftime("%Y-%m-%d %H:%M (%Z)")
    except (TypeError, ValueError):
        pass
    L: list[str] = []
    w = a.get("winner")

    if w is None:
        L += ["# LLM probe results", "", "> ❌ **No model completed the probe.** See failures below.", ""]
    else:
        acc = float(w["accuracy"])
        ci = w.get("accuracy_ci95") or [acc, acc]
        L += [
            '<div align="center">',
            "",
            "# 🏆 LLM Probe Results",
            "",
            f"**{n} gold-labeled samples · {len(a['completed']) + len(a['failed'])} candidate models · {when}**",
            "",
            "<br/>",
            "",
            "### THE WINNER IS",
            "",
            f"<h1>✨ {_short(w['model'])} ✨</h1>",
            "",
            f"<code>{w['model']}</code>",
            "",
            "<br/><br/>",
            "",
            " ".join(
                [
                    _badge("accuracy", _pct(acc), BLUE),
                    _badge("ECE", f"{float(w.get('ece') or 0):.3f}", "1baf7a"),
                    _badge("cost / 1k labels", _money(w.get("cost_per_1k_labels_usd"), 3), "eb6834"),
                ]
            ),
            "",
            "</div>",
            "",
            "---",
            "",
            "## 🎯 At a glance",
            "",
            "| 🎯 Accuracy | 📏 95% CI | 🧭 ECE | 📐 Brier | 💵 $ / 1k labels | ⏱️ Runtime | ⚠️ Errors |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
            f"| **{_pct(acc)}** | {_pct(ci[0])} – {_pct(ci[1])} | {float(w.get('ece') or 0):.3f} | "
            f"{float(w.get('brier') or 0):.3f} | {_money(w.get('cost_per_1k_labels_usd'), 3)} | "
            f"{float(w.get('runtime_seconds') or 0):.0f}s | {w.get('n_errors', 0)}/{n} |",
            "",
        ]

        # podium
        medals = ["🥇", "🥈", "🥉"]
        top = a["completed"][:3]
        L += ["## 🏅 Podium", "", "| | " + " | ".join(medals[: len(top)]) + " |",
              "|---|" + "|".join([":---:"] * len(top)) + "|"]
        L.append("| **Model** | " + " | ".join(f"**{_short(r['model'])}**" for r in top) + " |")
        L.append("| Accuracy | " + " | ".join(_pct(r["accuracy"]) for r in top) + " |")
        L.append("| ECE | " + " | ".join(f"{float(r.get('ece') or 0):.3f}" for r in top) + " |")
        L.append("| $ / 1k labels | " + " | ".join(_money(r.get("cost_per_1k_labels_usd"), 3) for r in top) + " |")
        L.append("| vs winner (McNemar p) | " + " | ".join(
            "—" if r is w else _p(r.get("_p_vs_winner")) for r in top) + " |")
        L.append("")

        # findings
        L += ["## 💡 Key findings", ""]
        L.append(
            f"- 🏆 **Winner: `{_short(w['model'])}`** with **{_pct(acc)}** accuracy "
            f"(95% CI {_pct(ci[0])}–{_pct(ci[1])}). Ranking rule: accuracy first, "
            "then Brier score (calibration) as tie-breaker."
        )
        tied = a.get("tied") or []
        if tied:
            names = ", ".join(f"`{_short(r['model'])}`" for r in tied)
            L.append(
                f"- 🤝 **Statistically tied with the winner** (paired McNemar p ≥ 0.05): {names}. "
                f"With {n} samples these differences are not significant — report them as equivalent."
            )
        else:
            L.append("- 📈 The winner is **significantly better** than every other model (McNemar p < 0.05).")
        bv = a.get("best_value")
        if bv is not None and bv is not w:
            wc = w.get("cost_per_1k_labels_usd")
            ratio = (float(wc) / float(bv["cost_per_1k_labels_usd"])) if wc else None
            L.append(
                f"- 💰 **Best value: `{_short(bv['model'])}`** — statistically tied with the winner "
                f"at {_money(bv['cost_per_1k_labels_usd'], 3)} per 1k labels"
                + (f" (**{ratio:.1f}× cheaper**)" if ratio and ratio > 1 else "")
                + ". For a cost-focused paper this is a strong default."
            )
        elif bv is w:
            L.append("- 💰 The winner is **also the best value**: no cheaper model is statistically tied with it.")
        bc = a.get("best_calibrated")
        if bc is not None:
            L.append(
                f"- 🧭 **Best calibrated: `{_short(bc['model'])}`** (ECE {float(bc['ece']):.3f}) — its "
                "stated confidence tracks real accuracy most closely, which matters because silver "
                "labels are accepted by a confidence threshold."
            )
        if a.get("cheapest") and a.get("priciest") and a["cheapest"] is not a["priciest"]:
            c, p = a["cheapest"], a["priciest"]
            spread = float(p["cost_per_1k_labels_usd"]) / float(c["cost_per_1k_labels_usd"])
            L.append(
                f"- 💵 Price spread: `{_short(c['model'])}` {_money(c['cost_per_1k_labels_usd'], 3)} → "
                f"`{_short(p['model'])}` {_money(p['cost_per_1k_labels_usd'], 3)} per 1k labels "
                f"(**{spread:.0f}×**)."
            )
        f = a.get("fastest")
        if f is not None:
            L.append(f"- ⚡ Fastest: `{_short(f['model'])}` ({float(f['runtime_seconds']):.0f}s for {n} labels).")
        think = [r for r in a["completed"] if int(r.get("reasoning_tokens") or 0) > 0]
        if think:
            L.append(
                "- 🧠 Reasoning tokens despite `reasoning: off`: "
                + ", ".join(f"`{_short(r['model'])}` ({int(r['reasoning_tokens']):,})" for r in think)
                + ". Mention this in the paper's setup."
            )
        if a["failed"]:
            L.append(f"- ❌ {len(a['failed'])} model(s) failed completely — see *Failures* below.")
        L += [
            f"- 💸 Total probe spend: **{_money(s.get('total_cost_usd'), 4)}** "
            f"({int(s.get('total_prompt_tokens') or 0):,} prompt + "
            f"{int(s.get('total_completion_tokens') or 0):,} completion tokens).",
            "",
        ]

        # leaderboard
        accs = [100 * float(r["accuracy"]) for r in a["completed"]]
        lo, hi = max(0.0, min(accs) - 6), 100.0
        L += [
            "## 📊 Leaderboard",
            "",
            "| # | Model | Accuracy | | 95% CI | p vs 🏆 | ECE | Brier | $ / 1k | Errors | Time |",
            "|---:|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(a["completed"], start=1):
            ci_r = r.get("accuracy_ci95") or [r["accuracy"], r["accuracy"]]
            tag = " 🏆" if r is w else (" 💰" if r is bv else "")
            name = f"**{_short(r['model'])}**{tag}" if r is w else f"{_short(r['model'])}{tag}"
            L.append(
                f"| {i} | {name} | `{_bar(100 * float(r['accuracy']), lo, hi)}` | "
                f"{_pct(r['accuracy'])} | {_pct(ci_r[0])}–{_pct(ci_r[1])} | "
                f"{'—' if r is w else _p(r.get('_p_vs_winner'))} | "
                f"{float(r.get('ece') or 0):.3f} | {float(r.get('brier') or 0):.3f} | "
                f"{_money(r.get('cost_per_1k_labels_usd'), 3)} | {r.get('n_errors', 0)} | "
                f"{float(r.get('runtime_seconds') or 0):.0f}s |"
            )
        L += ["", "🏆 winner · 💰 best value (cheapest model statistically tied with the winner)", ""]

    # figures
    captions = {
        "accuracy": ("🎯 Accuracy with 95% confidence intervals",
                     "Overlapping intervals mean the probe cannot separate those models."),
        "cost": ("💵 Cost vs accuracy",
                 "The dashed Pareto frontier shows models no other model beats on both price and accuracy."),
        "reliability": ("🧭 Reliability diagram",
                        "Points on the diagonal = honest confidence. Below it = overconfident."),
        "ece": ("📐 Calibration error by model", "Lower is better; drives silver-label quality."),
    }
    if any(fig_rel.values()):
        L += ["## 📈 Charts", ""]
        for key, (title, cap) in captions.items():
            rel = fig_rel.get(key)
            if rel:
                L += [f"### {title}", "", f"![{title}]({rel})", "", f"<sub>{cap}</sub>", ""]

    # pipeline diagram
    L += [
        "## 🔬 How the winner was selected",
        "",
        "```mermaid",
        "flowchart LR",
        f'    A["{n} gold samples<br/>balanced pos/neg<br/>(never from test)"] --> B["Each candidate LLM<br/>labels every sample once"]',
        '    B --> C["Accuracy + 95% CI<br/>Brier · ECE<br/>tokens · USD"]',
        '    C --> D{"Rank: accuracy,<br/>then Brier"}',
        f'    D --> E["🏆 {_short(w["model"]) if w else "none"}"]',
        '    E --> F["best_llm.json →<br/>main experiment"]',
        "```",
        "",
    ]

    if a["failed"]:
        L += ["## ❌ Failures", "", "| Model | Errors | First error |", "|---|---:|---|"]
        for r in a["failed"]:
            err = str(r.get("error") or "").replace("|", "/").replace("\n", " ")[:140]
            L.append(f"| `{r['model']}` | {r.get('n_errors', '—')} | {err} |")
        L.append("")

    # reproducibility
    L += [
        "<details>",
        "<summary><b>🧾 Reproducibility details</b></summary>",
        "",
        f"- Prompt version / shot mode: `{s.get('prompt_version')}` / `{s.get('shot_mode')}`",
        f"- Request options: `{json.dumps(s.get('request_options'), ensure_ascii=False)}`",
        f"- Probe samples: {n} (ids in `llm_probe/llm_probe_results.json`)",
        f"- Base config: `{s.get('base_config')}`",
        f"- Finished: {s.get('finished_at')}",
        "- Providers actually served (OpenRouter):",
    ]
    for r in a["completed"]:
        prov = ", ".join(r.get("served_providers") or []) or "—"
        served = ", ".join(r.get("served_models") or []) or "—"
        L.append(f"  - `{r['model']}` → providers: {prov}; served model: {served}")
    L += [
        "- Per-sample answers: `llm_probe/predictions/<model>.jsonl`",
        "- Statistics: Wilson 95% CI; exact two-sided McNemar test vs the winner on paired samples.",
        "",
        "</details>",
        "",
        "---",
        "",
        "### ▶️ Next step",
        "",
        "```bash",
        f"python run.py bakeoff --stage main --config {s.get('base_config')}",
        "```",
        "",
        "<sub>Generated by CalibrAL · regenerate without API calls: "
        f"<code>python run.py bakeoff --stage probe-report --config {s.get('base_config')}</code></sub>",
        "",
    ]
    return "\n".join(L)


def run_stamp(summary: dict[str, Any]) -> str:
    """Stable timestamp of a probe run (from its finished_at), e.g. 2026-11-23_1645."""
    try:
        # Local time of the machine that ran the probe (folder names read naturally).
        return (
            datetime.fromisoformat(str(summary.get("finished_at")))
            .astimezone()
            .strftime("%Y-%m-%d_%H%M")
        )
    except (TypeError, ValueError):
        return datetime.now().strftime("%Y-%m-%d_%H%M")


def build_probe_report(bakeoff_root: str | Path) -> Path | None:
    """Write figures + PROBE_REPORT.md for the probe in ``bakeoff_root``.

    The top-level ``PROBE_REPORT.md`` is always the latest run. A dated,
    self-contained copy (report, figures, results JSON, per-sample answers)
    is kept in ``llm_probe/history/<YYYY-MM-DD_HHMM>/`` so runs never
    overwrite each other. The stamp comes from the run's ``finished_at``, so
    regenerating the report for the same run reuses the same folder.
    """
    import shutil

    bakeoff_root = Path(bakeoff_root)
    a = analyse(bakeoff_root)
    if a is None:
        return None
    fig_dir = bakeoff_root / "llm_probe" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel: dict[str, str | None] = {}
    if a.get("completed"):
        makers = {
            "accuracy": (plot_accuracy, "probe_accuracy_ci.png"),
            "cost": (plot_cost_accuracy, "probe_cost_vs_accuracy.png"),
            "reliability": (plot_reliability, "probe_reliability.png"),
            "ece": (plot_ece, "probe_ece.png"),
        }
        for key, (fn, name) in makers.items():
            try:
                path = fn(a, fig_dir / name)
            except Exception as exc:  # noqa: BLE001 - a chart must not block the report
                get_logger("bakeoff").warning("Probe chart %s failed: %s", key, exc)
                path = None
            fig_rel[key] = f"llm_probe/figures/{name}" if path else None
    out = bakeoff_root / "PROBE_REPORT.md"
    out.write_text(render_markdown(a, fig_rel), encoding="utf-8")

    # Dated archive of this run.
    probe_dir = bakeoff_root / "llm_probe"
    hist = probe_dir / "history" / run_stamp(a["summary"])
    (hist / "figures").mkdir(parents=True, exist_ok=True)
    for key, rel in fig_rel.items():
        if rel:
            shutil.copy2(bakeoff_root / rel, hist / "figures" / Path(rel).name)
    for name in ("llm_probe_results.json", "llm_probe_report.md"):
        if (probe_dir / name).exists():
            shutil.copy2(probe_dir / name, hist / name)
    if (probe_dir / "predictions").is_dir():
        shutil.copytree(probe_dir / "predictions", hist / "predictions", dirs_exist_ok=True)
    hist_rel = {k: (f"figures/{Path(v).name}" if v else None) for k, v in fig_rel.items()}
    (hist / "PROBE_REPORT.md").write_text(render_markdown(a, hist_rel), encoding="utf-8")

    # Dated report next to the latest one; its charts point into the archive,
    # so it keeps showing this run's figures after later runs.
    stamp = run_stamp(a["summary"])
    dated_rel = {
        k: (f"llm_probe/history/{stamp}/figures/{Path(v).name}" if v else None)
        for k, v in fig_rel.items()
    }
    dated = bakeoff_root / f"PROBE_REPORT_{stamp}.md"
    dated.write_text(render_markdown(a, dated_rel), encoding="utf-8")
    get_logger("bakeoff").info(
        "Probe report written | latest=%s | dated=%s | archive=%s", out, dated, hist
    )
    return dated
