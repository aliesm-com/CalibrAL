"""Final delivery checks for CalibrAL (milestone M8)."""

from __future__ import annotations

import ast
from pathlib import Path

from calibral.reporting import build_report, load_results


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_trainer_source_does_not_reference_test_split():
    source = (PROJECT_ROOT / "calibral" / "trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "test" not in names
    assert "bundle" not in names
    assert "test_df" not in source


def test_report_disclaimer_when_results_are_mock_only(tmp_path: Path):
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "method": "proposed",
                "seed": 42,
                "profile": "smoke",
                "macro_f1": 0.5,
                "accuracy": 0.5,
                "human_labels_used": 16,
                "llm_labels_accepted": 2,
                "llm_calls": 4,
                "runtime_seconds": 1.0,
                "llm_mode": "mock",
                "status": "completed",
            }
        ]
    ).to_csv(csv_path, index=False)

    result = build_report(
        {"profile": "smoke", "notes": "NOT_SCIENTIFIC_SMOKE_RESULT"},
        results_path=csv_path,
        output_dir=tmp_path,
    )
    text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "LLM Mock" in text
    assert Path(result["method_comparison_path"]).exists()
    assert Path(result["learning_curve_path"]).exists()


def test_existing_results_csv_has_required_columns_if_present():
    path = PROJECT_ROOT / "outputs" / "results.csv"
    if not path.exists():
        return
    df = load_results(path)
    assert df is not None
    required = {
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
    }
    assert required.issubset(df.columns)
