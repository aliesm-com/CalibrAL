#!/usr/bin/env python3
"""CLI wrapper for thesis bakeoff (LLM probe + main thesis experiment)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibral.bakeoff import (  # noqa: E402
    run_all,
    run_encoder_sweep,
    run_llm_probe,
    run_main_experiment,
    write_encoder_configs,
)
from calibral.utils import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Thesis bakeoff: 1-call LLM probe, then main thesis_al experiment. "
            "Encoder sweep is optional/expensive."
        )
    )
    parser.add_argument(
        "stage",
        choices=("llm-probe", "write-configs", "encoder-sweep", "main", "all"),
        help=(
            "llm-probe | write-configs | encoder-sweep | main | all "
            "(all = probe + main thesis_al)"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/thesis_al.yaml",
        help="Base thesis config (default: configs/thesis_al.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(level=10 if args.verbose else 20)

    if args.stage == "llm-probe":
        summary = run_llm_probe(args.config)
        print(f"best_llm={summary.get('best_model')}")
        print("report: outputs/bakeoff/llm_probe/llm_probe_report.md")
        return 0 if summary.get("best_model") else 1

    if args.stage == "write-configs":
        paths = write_encoder_configs(args.config)
        print(f"wrote {len(paths)} encoder configs:")
        for path in paths:
            print(f"  - {path}")
        return 0

    if args.stage == "encoder-sweep":
        summary = run_encoder_sweep(args.config)
        print("report: outputs/bakeoff/bakeoff_report.md")
        failed = sum(int(i.get("failed") or 0) for i in summary.get("items") or [])
        return 0 if failed == 0 else 1

    if args.stage == "main":
        summary = run_main_experiment(args.config)
        print(f"llm={summary.get('llm_model')}")
        print(f"report: {summary.get('report_path')}")
        failed = int((summary.get("comparison") or {}).get("failed") or 0)
        return 0 if failed == 0 else 1

    summary = run_all(args.config, include_encoder_sweep=False)
    print(f"best_llm={summary['llm_probe'].get('best_model')}")
    main = summary.get("main_experiment") or {}
    print(f"main_report: {main.get('report_path')}")
    print("report: outputs/bakeoff/bakeoff_report.md")
    failed = int((main.get("comparison") or {}).get("failed") or 0)
    ok = bool(summary["llm_probe"].get("best_model")) and failed == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
