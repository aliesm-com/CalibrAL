"""Smoke-oriented CLI and package bootstrap tests for M0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from calibral import __version__
from calibral.utils import detect_device, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_package_version_is_set():
    assert __version__
    assert isinstance(__version__, str)


def test_setup_logging_returns_logger():
    logger = setup_logging()
    assert logger.name == "calibral"


def test_detect_device_returns_known_value():
    assert detect_device() in {"cpu", "cuda"}


def test_package_exposes_calibr_al_name():
    from calibral import __project_name__

    assert __project_name__ == "CalibrAL"


def test_run_py_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "smoke" in result.stdout
    assert "experiment" in result.stdout
