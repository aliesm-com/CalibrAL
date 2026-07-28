"""Tests for reporting helpers (numbers must come from CSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibral.reporting import build_report, render_report_markdown, summarize_results


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "b1",
                "method": "baseline",
                "seed": 42,
                "profile": "smoke",
                "macro_f1": 0.4,
                "accuracy": 0.5,
                "human_labels_used": 16,
                "llm_labels_accepted": 0,
                "llm_calls": 0,
                "runtime_seconds": 3.0,
                "llm_mode": "none",
                "status": "completed",
            },
            {
                "run_id": "p1",
                "method": "proposed",
                "seed": 42,
                "profile": "smoke",
                "macro_f1": 0.6,
                "accuracy": 0.7,
                "human_labels_used": 16,
                "llm_labels_accepted": 5,
                "llm_calls": 10,
                "runtime_seconds": 4.0,
                "llm_mode": "mock",
                "status": "completed",
            },
            {
                "run_id": "p2",
                "method": "proposed",
                "seed": 43,
                "profile": "smoke",
                "macro_f1": None,
                "accuracy": None,
                "human_labels_used": None,
                "llm_labels_accepted": None,
                "llm_calls": None,
                "runtime_seconds": 1.0,
                "llm_mode": "mock",
                "status": "failed",
            },
        ]
    )


def test_summarize_results_uses_completed_only():
    summary = summarize_results(_sample_results())
    assert summary["n_completed"] == 2
    assert summary["n_failed"] == 1
    assert summary["methods"]["proposed"]["macro_f1_mean"] == 0.6
    assert summary["only_mock_or_none"] is True


def test_render_report_includes_mock_disclaimer_and_csv_numbers():
    summary = summarize_results(_sample_results())
    text = render_report_markdown(summary, results_path="outputs/results.csv")
    assert "تست نرم‌افزاری با LLM Mock" in text
    assert "0.6000" in text
    assert "baseline" in text


def test_build_report_incomplete_without_csv(tmp_path: Path):
    result = build_report(
        {"profile": "smoke", "paths": {"output_dir": str(tmp_path)}},
        results_path=tmp_path / "missing.csv",
        output_dir=tmp_path,
    )
    assert result["status"] == "incomplete"
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "incomplete" in report.lower()


def test_build_experiment_figures_from_csv(tmp_path: Path):
    from calibral.thesis_figures import build_experiment_figures

    csv_path = tmp_path / "results.csv"
    sample = _sample_results()
    sample["epochs"] = [1, 1, 1]
    sample["n_al_rounds"] = [2, 2, 0]
    sample["llm_labels_accepted"] = [0, 5, 0]
    sample.to_csv(csv_path, index=False)
    paths = build_experiment_figures(tmp_path, results_path=csv_path)
    assert paths.get("macro_f1_by_seed")
    assert Path(paths["macro_f1_by_seed"]).exists()

    csv_path = tmp_path / "results.csv"
    sample = _sample_results()
    sample["epochs"] = [1, 1, 1]
    sample["n_al_rounds"] = [2, 2, 0]
    sample["runtime_human"] = ["3s", "4s", "1s"]
    sample.to_csv(csv_path, index=False)
    result = build_report(
        {
            "profile": "smoke",
            "paths": {"output_dir": str(tmp_path)},
            "notes": "NOT_SCIENTIFIC_SMOKE_RESULT",
            "train": {"epochs": 1, "batch_size": 4, "learning_rate": 1e-4},
            "data": {
                "initial_labeled_size": 8,
                "pool_size": 32,
                "validation_size": 8,
                "test_size": 16,
            },
            "model": {"name": "prajjwal1/bert-tiny", "max_length": 64},
            "active_learning": {"budget": 8, "query_batch_size": 4},
            "llm": {"mode": "mock", "model": "mock", "shot_mode": "zero"},
        },
        results_path=csv_path,
        output_dir=tmp_path,
    )
    assert result["status"] == "completed"
    assert Path(result["method_comparison_path"]).exists()
    assert Path(result["learning_curve_path"]).exists()
    assert Path(result["report_path"]).exists()
    summary_path = Path(result["experiment_summary_path"])
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "Experiment Summary" in text
    assert "epochs (per retrain)" in text
    assert "Runtime overview" in text
