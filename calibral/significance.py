"""Paper statistics: per-method summary, paired tests and cost table.

Reads ``results.csv`` + ``predictions_<method>_seed<seed>.csv`` from one
experiment folder and writes ``significance.json`` / ``significance.md``.

Two complementary tests against a reference method (default ``baseline``):

- Across seeds: paired t-test and Wilcoxon signed-rank on per-seed Macro-F1.
  With 5 seeds the smallest possible two-sided Wilcoxon p is 0.0625, so the
  t-test (and the bootstrap below) carry the inference; Wilcoxon is reported
  for completeness.
- Within each seed: paired bootstrap over test examples of the Macro-F1
  difference (same test items for both methods), giving a 95% CI and a
  one-sided p-value for "method > reference".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibral.utils import get_logger


def _binary_macro_f1_batch(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Macro-F1 for binary labels, vectorised over rows of (B, N) arrays."""
    t1 = true == 1
    p1 = pred == 1
    tp1 = np.sum(t1 & p1, axis=1)
    fp1 = np.sum(~t1 & p1, axis=1)
    fn1 = np.sum(t1 & ~p1, axis=1)
    tp0 = np.sum(~t1 & ~p1, axis=1)
    fp0 = fn1
    fn0 = fp1

    def _f1(tp, fp, fn):
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    return (_f1(tp1, fp1, fn1) + _f1(tp0, fp0, fn0)) / 2.0


def paired_bootstrap(
    true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    chunk: int = 200,
) -> dict[str, float]:
    """Bootstrap the Macro-F1 difference (a - b) over shared test items."""
    true = np.asarray(true, dtype=np.int8)
    pred_a = np.asarray(pred_a, dtype=np.int8)
    pred_b = np.asarray(pred_b, dtype=np.int8)
    n = len(true)
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int64)
        t = true[idx]
        diffs.append(
            _binary_macro_f1_batch(t, pred_a[idx]) - _binary_macro_f1_batch(t, pred_b[idx])
        )
        done += b
    d = np.concatenate(diffs)
    observed = float(
        _binary_macro_f1_batch(true[None, :], pred_a[None, :])[0]
        - _binary_macro_f1_batch(true[None, :], pred_b[None, :])[0]
    )
    return {
        "delta_macro_f1": round(observed, 6),
        "ci95_low": round(float(np.percentile(d, 2.5)), 6),
        "ci95_high": round(float(np.percentile(d, 97.5)), 6),
        # One-sided: share of resamples where a is not better than b.
        "p_one_sided": round(float(np.mean(d <= 0.0)), 6),
        "n_resamples": int(n_resamples),
        "n_test": int(n),
    }


def _seed_tests(diffs: list[float]) -> dict[str, Any]:
    from scipy import stats

    arr = np.asarray(diffs, dtype=float)
    out: dict[str, Any] = {
        "n_seeds": int(len(arr)),
        "delta_mean": round(float(arr.mean()), 6) if len(arr) else None,
        "delta_std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else None,
        "ttest_p": None,
        "wilcoxon_p": None,
    }
    if len(arr) >= 2 and np.any(arr != arr[0]):
        out["ttest_p"] = round(float(stats.ttest_1samp(arr, 0.0).pvalue), 6)
    if len(arr) >= 2 and np.any(arr != 0):
        try:
            out["wilcoxon_p"] = round(float(stats.wilcoxon(arr).pvalue), 6)
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def _load_preds(output_dir: Path, method: str, seed: int) -> pd.DataFrame | None:
    path = output_dir / f"predictions_{method}_seed{seed}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _mean_std(series: pd.Series) -> tuple[float | None, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.mean()), (float(s.std(ddof=1)) if len(s) > 1 else 0.0)


def compute_significance(
    output_dir: str | Path,
    *,
    reference: str = "baseline",
    n_resamples: int = 2000,
) -> dict[str, Any] | None:
    from calibral.reporting import dedupe_results, load_results

    output_dir = Path(output_dir)
    df = load_results(output_dir / "results.csv")
    if df is None:
        return None
    df = dedupe_results(df)
    if df.empty:
        return None
    df["seed"] = df["seed"].astype(int)

    methods = list(dict.fromkeys(df["method"].astype(str)))
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        sub = df[df["method"] == method]
        row: dict[str, Any] = {"method": method, "n_seeds": int(len(sub))}
        for col in (
            "macro_f1",
            "accuracy",
            "human_labels_used",
            "llm_labels_accepted",
            "llm_calls",
            "n_train",
            "silver_label_accuracy",
            "rejected_llm_accuracy",
            "llm_cost_usd",
            "human_cost_usd",
            "total_cost_usd",
        ):
            if col in sub.columns:
                mean, std = _mean_std(sub[col])
                row[f"{col}_mean"] = None if mean is None else round(mean, 6)
                row[f"{col}_std"] = None if std is None else round(std, 6)
        summary_rows.append(row)

    comparisons: list[dict[str, Any]] = []
    ref = df[df["method"] == reference].set_index("seed")
    if not ref.empty:
        for method in methods:
            if method == reference:
                continue
            cur = df[df["method"] == method].set_index("seed")
            seeds = sorted(set(cur.index) & set(ref.index))
            if not seeds:
                continue
            diffs = [float(cur.loc[s, "macro_f1"]) - float(ref.loc[s, "macro_f1"]) for s in seeds]
            per_seed: list[dict[str, Any]] = []
            for s in seeds:
                a = _load_preds(output_dir, method, s)
                b = _load_preds(output_dir, reference, s)
                if a is None or b is None:
                    continue
                merged = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
                if merged.empty:
                    continue
                boot = paired_bootstrap(
                    merged["true_label_a"].to_numpy(),
                    merged["pred_label_a"].to_numpy(),
                    merged["pred_label_b"].to_numpy(),
                    n_resamples=n_resamples,
                    seed=int(s),
                )
                per_seed.append({"seed": int(s), **boot})
            comparisons.append(
                {
                    "method": method,
                    "reference": reference,
                    "seeds": seeds,
                    "per_seed_delta": [round(d, 6) for d in diffs],
                    **_seed_tests(diffs),
                    "bootstrap": per_seed,
                    "bootstrap_seeds_significant_p05": sum(
                        1 for b in per_seed if b["p_one_sided"] < 0.05
                    ),
                }
            )

    from datetime import datetime

    now = datetime.now()
    result = {
        "project": "CalibrAL",
        "generated_at": now.isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "reference": reference,
        "methods": summary_rows,
        "comparisons": comparisons,
        "notes": [
            "Deltas are Macro-F1 (method - reference); positive = better than reference.",
            "ttest_p: two-sided one-sample t-test on per-seed deltas (= paired t-test).",
            "wilcoxon_p: with 5 seeds the minimum two-sided p is 0.0625.",
            "bootstrap: paired over test items within each seed; p_one_sided tests "
            "method > reference.",
            "human_cost_usd counts training labels only; validation_gold_labels "
            "(identical for all methods) are in results.csv.",
        ],
    }
    (output_dir / "significance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = render_significance_md(result)
    (output_dir / "significance.md").write_text(md, encoding="utf-8")
    # Dated copy so successive analyses never overwrite each other.
    hist = output_dir / "history"
    hist.mkdir(parents=True, exist_ok=True)
    (hist / f"significance_{now.strftime('%Y-%m-%d_%H%M')}.md").write_text(md, encoding="utf-8")
    get_logger("reporting").info("Significance written | %s", output_dir / "significance.md")
    return result


def _pct(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * float(x):.2f}"


def _num(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):,.{digits}f}"


def render_significance_md(result: dict[str, Any]) -> str:
    lines = [
        "# Paper statistics",
        "",
        f"Generated: {result.get('generated_at', '')} · Reference method: `{result['reference']}`",
        "",
        "## Methods (mean ± std over seeds)",
        "",
        "| method | seeds | Macro-F1 % | human labels | LLM calls | train size | "
        "silver acc % | rejected-LLM acc % | LLM $ | human $ | total $ |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["methods"]:
        f1 = r.get("macro_f1_mean")
        f1s = r.get("macro_f1_std")
        lines.append(
            f"| `{r['method']}` | {r['n_seeds']} | "
            f"{_pct(f1)} ± {_pct(f1s)} | {_num(r.get('human_labels_used_mean'), 0)} | "
            f"{_num(r.get('llm_calls_mean'), 0)} | {_num(r.get('n_train_mean'), 0)} | "
            f"{_pct(r.get('silver_label_accuracy_mean'))} | "
            f"{_pct(r.get('rejected_llm_accuracy_mean'))} | "
            f"{_num(r.get('llm_cost_usd_mean'), 3)} | {_num(r.get('human_cost_usd_mean'))} | "
            f"{_num(r.get('total_cost_usd_mean'))} |"
        )
    lines += [
        "",
        "## Paired comparison vs reference",
        "",
        "| method | Δ Macro-F1 (pts) mean ± std | paired t p | Wilcoxon p | "
        "seeds with bootstrap p<0.05 |",
        "|---|---|---:|---:|---:|",
    ]
    for c in result["comparisons"]:
        d = c.get("delta_mean")
        ds = c.get("delta_std")
        lines.append(
            f"| `{c['method']}` | {_num(None if d is None else 100 * d)} ± "
            f"{_num(None if ds is None else 100 * ds)} | {_num(c.get('ttest_p'), 4)} | "
            f"{_num(c.get('wilcoxon_p'), 4)} | "
            f"{c.get('bootstrap_seeds_significant_p05')}/{len(c.get('bootstrap') or [])} |"
        )
    lines += ["", "## Per-seed paired bootstrap (Δ Macro-F1, 95% CI)", ""]
    for c in result["comparisons"]:
        for b in c.get("bootstrap") or []:
            lines.append(
                f"- `{c['method']}` seed {b['seed']}: Δ={100 * b['delta_macro_f1']:.2f} pts "
                f"[{100 * b['ci95_low']:.2f}, {100 * b['ci95_high']:.2f}], "
                f"p(one-sided)={b['p_one_sided']:.4f}"
            )
    lines += ["", "## Notes", ""] + [f"- {n}" for n in result["notes"]] + [""]
    return "\n".join(lines)
