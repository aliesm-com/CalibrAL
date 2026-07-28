#!/usr/bin/env python3
"""Entry point for the CalibrAL research prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from calibral import __project_name__, __version__
from calibral.config import load_config
from calibral.utils import describe_device, get_logger, load_dotenv, setup_logging


COMMANDS = ("smoke", "experiment", "report", "figures", "all", "sweep", "bakeoff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "CalibrAL - minimal research prototype for hybrid active learning "
            "on binary IMDB sentiment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py smoke\n"
            "  python run.py experiment --config configs/default.yaml\n"
            "  python run.py sweep --config-dir configs/models\n"
            "  python run.py bakeoff --stage llm-probe --config configs/thesis_al.yaml\n"
            "  python run.py bakeoff --stage encoder-sweep --config configs/thesis_al.yaml\n"
            "  python run.py bakeoff --stage all --config configs/thesis_al.yaml\n"
            "  python run.py report\n"
            "  python run.py all\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__project_name__} {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        help="Command to run: smoke | experiment | report | figures | all | sweep | bakeoff",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config profile name (smoke/default/colab) or path to a YAML file.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory of YAML configs for sweep (e.g. configs/models).",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help=(
            "Optional stage filter. For smoke: data|baseline|llm|proposed|all. "
            "For bakeoff: llm-probe|write-configs|encoder-sweep|main|all."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _not_ready(command: str, milestone: str) -> int:
    logger = get_logger("run")
    logger.error(
        "Command %r is not implemented yet. It will be available after %s.",
        command,
        milestone,
    )
    print(
        f"Not implemented yet: '{command}'. "
        f"This command becomes available after {milestone}.",
        file=sys.stderr,
    )
    return 2


def cmd_smoke(args: argparse.Namespace, config: dict[str, Any]) -> int:
    stage = args.stage or "all"
    logger = get_logger("run")

    if stage == "data":
        from calibral.data import prepare_data, print_data_summary

        logger.info("Running smoke stage: data")
        bundle = prepare_data(config)
        print_data_summary(bundle)
        return 0

    if stage == "baseline":
        from calibral.experiment import run_baseline

        logger.info("Running smoke stage: baseline")
        summary = run_baseline(config)
        print(
            "Baseline smoke finished | "
            f"status={summary.get('status')} | "
            f"device={summary.get('device')} | "
            f"macro_f1={summary.get('macro_f1')} | "
            f"accuracy={summary.get('accuracy')} | "
            f"runtime_seconds={summary.get('runtime_seconds')} | "
            f"tag={summary.get('result_tag', 'n/a')}"
        )
        print(f"predictions: {summary.get('predictions_path')}")
        print(f"summary: {summary.get('summary_path')}")
        print(f"results: {summary.get('results_path')}")
        return 0 if summary.get("status") == "completed" else 1

    if stage == "llm":
        from calibral.llm_labeler import run_llm_smoke

        logger.info("Running smoke stage: llm")
        summary = run_llm_smoke(config)
        print(
            "LLM smoke finished | "
            f"llm_mode={summary.get('llm_mode')} | "
            f"n_texts={summary.get('n_texts')} | "
            f"tag={summary.get('result_tag', 'n/a')}"
        )
        for item in summary.get("items", []):
            print(
                f"  - label={item['label']} conf={item['confidence']:.3f} "
                f"source={item['source']} preview={item['text_preview']!r}"
            )
        print(f"artifact: {summary.get('artifact_path')}")
        return 0

    if stage == "proposed":
        from calibral.experiment import run_proposed

        logger.info("Running smoke stage: proposed")
        summary = run_proposed(config)
        print(
            "Proposed smoke finished | "
            f"status={summary.get('status')} | "
            f"macro_f1={summary.get('macro_f1')} | "
            f"accuracy={summary.get('accuracy')} | "
            f"gold={summary.get('gold_labels_total')} | "
            f"silver={summary.get('silver_labels_total')} | "
            f"llm_calls={summary.get('llm_calls')} | "
            f"human_labels_used={summary.get('human_labels_used')} | "
            f"llm_mode={summary.get('llm_mode')} | "
            f"tag={summary.get('result_tag', 'n/a')}"
        )
        print(f"predictions: {summary.get('predictions_path')}")
        print(f"summary: {summary.get('summary_path')}")
        print(f"results: {summary.get('results_path')}")
        return 0 if summary.get("status") == "completed" else 1

    if stage == "all":
        from calibral.experiment import run_comparison

        logger.info("Running full smoke comparison (baseline + proposed)")
        comparison = run_comparison(config, seeds=[int(config.get("seed", 42))])
        print(
            "Smoke comparison finished | "
            f"completed={comparison.get('completed')} | "
            f"failed={comparison.get('failed')} | "
            f"tag={comparison.get('result_tag', 'n/a')}"
        )
        for run in comparison.get("runs", []):
            print(
                f"  - {run.get('method')} seed={run.get('seed')} "
                f"status={run.get('status')} macro_f1={run.get('macro_f1')} "
                f"human={run.get('human_labels_used')} "
                f"silver={run.get('llm_labels_accepted')} "
                f"llm_calls={run.get('llm_calls')} "
                f"llm_mode={run.get('llm_mode')}"
            )
        print(f"results: {comparison.get('results_path')}")
        print(f"comparison: {comparison.get('comparison_path')}")
        return 0 if comparison.get("failed", 0) == 0 else 1

    print(
        f"Unknown smoke stage: {stage!r}. "
        "Available now: data, baseline, llm, proposed, all",
        file=sys.stderr,
    )
    return 2


def cmd_experiment(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from calibral.experiment import run_experiment

    logger = get_logger("run")
    logger.info("Running experiment comparison")
    comparison = run_experiment(config)
    print(
        "Experiment finished | "
        f"completed={comparison.get('completed')} | "
        f"failed={comparison.get('failed')} | "
        f"seeds={comparison.get('seeds')}"
    )
    for run in comparison.get("runs", []):
        print(
            f"  - {run.get('method')} seed={run.get('seed')} "
            f"status={run.get('status')} macro_f1={run.get('macro_f1')} "
            f"llm_mode={run.get('llm_mode')}"
        )
    print(f"results: {comparison.get('results_path')}")
    print(f"comparison: {comparison.get('comparison_path')}")
    print(f"experiment_summary: {comparison.get('experiment_summary_path')}")
    print(f"output_dir: {comparison.get('output_dir')}")
    return 0 if comparison.get("failed", 0) == 0 else 1


def cmd_report(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from calibral.reporting import build_report

    logger = get_logger("run")
    out = config.get("paths", {}).get("output_dir", "outputs")
    logger.info("Building report from %s/results.csv", out)
    result = build_report(config)
    print(
        "Report finished | "
        f"status={result.get('status')} | "
        f"report={result.get('report_path')}"
    )
    print(f"experiment_summary: {result.get('experiment_summary_path')}")
    print(f"method_comparison: {result.get('method_comparison_path')}")
    print(f"learning_curve: {result.get('learning_curve_path')}")
    print(f"results: {result.get('results_path')}")
    for name, path in (result.get("thesis_figures") or {}).items():
        if path:
            print(f"figure/{name}: {path}")
    for name, path in (result.get("bakeoff_figures") or {}).items():
        if path:
            print(f"bakeoff/{name}: {path}")
    return 0 if result.get("status") in {"completed", "incomplete"} else 1


def cmd_figures(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Regenerate thesis plots from saved CSV/JSON (no GPU re-run)."""
    from calibral.config import resolve_output_dir
    from calibral.thesis_figures import build_bakeoff_figures, build_experiment_figures, sync_docs_thesis_figures

    logger = get_logger("run")
    out = resolve_output_dir(config, create=True)
    logger.info("Building thesis figures from %s", out)
    exp = build_experiment_figures(out)
    bake = build_bakeoff_figures("outputs/bakeoff")
    docs_dir = sync_docs_thesis_figures(out)
    print(f"Experiment figures: {out / 'figures'}")
    for name, path in exp.items():
        if path:
            print(f"  {name}: {path}")
    print("Bakeoff figures: outputs/bakeoff/figures")
    for name, path in bake.items():
        if path:
            print(f"  {name}: {path}")
    print(f"README assets: {docs_dir}")
    return 0


def cmd_sweep(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from calibral.experiment import run_sweep

    logger = get_logger("run")
    config_dir = Path(args.config_dir) if args.config_dir else Path("configs/models")
    if not config_dir.exists():
        print(f"Config dir not found: {config_dir}", file=sys.stderr)
        return 1
    yaml_files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not yaml_files:
        print(f"No YAML files in {config_dir}", file=sys.stderr)
        return 1

    logger.info("Running sweep over %d configs in %s", len(yaml_files), config_dir)
    summary = run_sweep(yaml_files)
    print(
        "Sweep finished | "
        f"n_configs={summary.get('n_configs')} | "
        f"runtime_seconds={summary.get('runtime_seconds')}"
    )
    for item in summary.get("items", []):
        print(
            f"  - {item.get('profile')} | model={item.get('model_name')} | "
            f"llm={item.get('llm_model')} | "
            f"completed={item.get('completed')} failed={item.get('failed')}"
        )
    print(f"results: {summary.get('results_path')}")
    print(f"sweep: {summary.get('sweep_path')}")
    n_failed = sum(int(i.get("failed") or 0) for i in summary.get("items", []))
    return 0 if n_failed == 0 else 1


def cmd_all(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Run experiment then report, and print final artifact paths."""
    logger = get_logger("run")
    logger.info("Running all = experiment + report")
    exp_code = cmd_experiment(args, config)
    rep_code = cmd_report(args, config)

    from calibral.config import resolve_output_dir

    output_dir = resolve_output_dir(config, create=False)
    final_files = [
        "results.csv",
        "predictions.csv",
        "run_summary.json",
        "comparison_summary.json",
        "experiment_summary.md",
        "learning_curve.png",
        "method_comparison.png",
        "report.md",
    ]
    print("Final artifacts:")
    for name in final_files:
        path = output_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"  - {path} | exists={exists} | size={size}")

    return exp_code if exp_code != 0 else rep_code


def cmd_bakeoff(args: argparse.Namespace, config: dict[str, Any]) -> int:
    from calibral.bakeoff import (
        run_all,
        run_encoder_sweep,
        run_llm_probe,
        run_main_experiment,
        write_encoder_configs,
    )

    stage = args.stage or "all"
    config_path = args.config or "configs/thesis_al.yaml"

    if stage == "llm-probe":
        summary = run_llm_probe(config_path)
        print(f"best_llm={summary.get('best_model')}")
        print("report: outputs/bakeoff/llm_probe/llm_probe_report.md")
        print("best_json: outputs/bakeoff/best_llm.json")
        return 0 if summary.get("best_model") else 1

    if stage == "write-configs":
        paths = write_encoder_configs(config_path)
        print(f"wrote {len(paths)} encoder configs under configs/sweeps/thesis_encoders/")
        for path in paths:
            print(f"  - {path}")
        return 0

    if stage == "encoder-sweep":
        summary = run_encoder_sweep(config_path)
        print("report: outputs/bakeoff/bakeoff_report.md")
        failed = sum(int(i.get("failed") or 0) for i in summary.get("items") or [])
        return 0 if failed == 0 else 1

    if stage == "main":
        summary = run_main_experiment(config_path)
        print(f"llm={summary.get('llm_model')}")
        print(f"output: {summary.get('output_dir')}")
        print(f"report: {summary.get('report_path')}")
        failed = int((summary.get("comparison") or {}).get("failed") or 0)
        return 0 if failed == 0 else 1

    if stage == "all":
        # Default path for thesis: probe best LLM, then main thesis_al experiment.
        # Encoder ablation is optional and expensive; use --stage encoder-sweep.
        summary = run_all(config_path, include_encoder_sweep=False)
        print(f"best_llm={summary['llm_probe'].get('best_model')}")
        main = summary.get("main_experiment") or {}
        print(f"main_output: {main.get('output_dir')}")
        print(f"main_report: {main.get('report_path')}")
        print("report: outputs/bakeoff/bakeoff_report.md")
        failed = int((main.get("comparison") or {}).get("failed") or 0)
        ok = bool(summary["llm_probe"].get("best_model")) and failed == 0
        return 0 if ok else 1

    print(
        f"Unknown bakeoff stage: {stage!r}. "
        "Use: llm-probe | write-configs | encoder-sweep | main | all",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv()
    level = 10 if args.verbose else 20  # DEBUG / INFO
    setup_logging(level=level)
    logger = get_logger("run")

    if args.command is None:
        parser.print_help()
        return 0

    config_ref = args.config
    if config_ref is None and args.command == "smoke":
        config_ref = "smoke"
    if config_ref is None and args.command in {"report", "all", "figures"}:
        # Prefer smoke unless the user explicitly passes --config.
        config_ref = "smoke"
    if args.command == "bakeoff":
        # Bakeoff reloads configs itself; placeholder only for boot logging.
        if config_ref is None:
            config_ref = "configs/thesis_al.yaml"
    if args.command == "sweep":
        # Sweep loads each YAML itself; use a lightweight placeholder for boot logging.
        if args.config_dir:
            first = sorted(Path(args.config_dir).glob("*.yaml"))
            config_ref = str(first[0]) if first else "smoke"
        elif config_ref is None:
            config_ref = "smoke"

    try:
        config = load_config(config_ref)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    device = describe_device()
    logger.info(
        "%s %s | command=%s | profile=%s | device=%s | config=%s",
        __project_name__,
        __version__,
        args.command,
        config.get("profile", "unknown"),
        device,
        config.get("_config_path"),
    )

    handlers = {
        "smoke": cmd_smoke,
        "experiment": cmd_experiment,
        "report": cmd_report,
        "figures": cmd_figures,
        "all": cmd_all,
        "sweep": cmd_sweep,
        "bakeoff": cmd_bakeoff,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    # Ensure project root is on sys.path when invoked as a script.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
