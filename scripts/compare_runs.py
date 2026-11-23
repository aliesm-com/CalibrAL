#!/usr/bin/env python3
"""Side-by-side comparison of several experiment folders (e.g. an encoder ablation).

Usage:
    python scripts/compare_runs.py outputs/paper_snappfood outputs/paper_snappfood_xlmr

Prints and writes ``<first folder>/comparison_<names>.md`` with, per folder:
mean ± std Macro-F1 per method, and Δ vs baseline (mean, and how many seeds
the method beat the baseline). If the ranking of methods / the sign of Δ is
the same in every folder, the conclusions do not depend on that setting.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from calibral.reporting import dedupe_results, load_results  # noqa: E402


def _summary(folder: Path) -> tuple[str, pd.DataFrame] | None:
    df = load_results(folder / "results.csv")
    if df is None:
        return None
    df = dedupe_results(df)
    if df.empty:
        return None
    model = str(df["model_name"].dropna().iloc[0]) if "model_name" in df else "?"
    base = df[df["method"] == "baseline"].set_index("seed")["macro_f1"]
    rows = []
    for method, sub in df.groupby("method", sort=False):
        f1 = sub["macro_f1"].astype(float)
        row = {
            "method": method,
            "seeds": len(sub),
            "f1": f"{100 * f1.mean():.2f} ± {100 * (f1.std(ddof=1) if len(f1) > 1 else 0):.2f}",
        }
        if method != "baseline" and not base.empty:
            s = sub.set_index("seed")["macro_f1"]
            common = s.index.intersection(base.index)
            d = (s.loc[common] - base.loc[common]).astype(float)
            row["delta"] = f"{100 * d.mean():+.2f}" if len(d) else ""
            row["wins"] = f"{int((d > 0).sum())}/{len(d)}" if len(d) else ""
        else:
            row["delta"], row["wins"] = "—", "—"
        rows.append(row)
    return model, pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    folders = [Path(a) for a in argv]
    lines = [
        "# Run comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Δ = Macro-F1 points vs `baseline` (paired by seed); wins = seeds where the method beat it.",
        "",
    ]
    for folder in folders:
        res = _summary(folder)
        lines.append(f"## `{folder.as_posix()}`")
        lines.append("")
        if res is None:
            lines += ["_no completed results_", ""]
            continue
        model, table = res
        lines += [
            f"Encoder: `{model}`",
            "",
            "| method | seeds | Macro-F1 % | Δ vs baseline | wins |",
            "|---|---:|---|---:|---:|",
        ]
        for _, r in table.iterrows():
            lines.append(f"| `{r['method']}` | {r['seeds']} | {r['f1']} | {r['delta']} | {r['wins']} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    names = "_vs_".join(f.name for f in folders)
    out = folders[0] / f"comparison_{names}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
