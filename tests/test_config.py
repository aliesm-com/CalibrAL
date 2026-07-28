"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibral.config import (
    get_nested,
    load_config,
    resolve_config_path,
    resolve_output_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_smoke_config_by_profile():
    config = load_config("smoke")
    assert config["profile"] == "smoke"
    assert config["llm"]["mode"] == "mock"
    assert config["model"]["name"] == "prajjwal1/bert-tiny"
    assert Path(config["_config_path"]).name == "smoke.yaml"
    assert Path(config["paths"]["output_dir"]).as_posix() == "outputs/smoke"


def test_resolve_output_dir_nests_flat_outputs(tmp_path: Path):
    config = {"profile": "thesis_al", "paths": {"output_dir": "outputs"}}
    path = resolve_output_dir(config, create=True)
    assert path.as_posix().endswith("outputs/thesis_al")
    assert config["paths"]["output_dir"] == str(path)


def test_load_default_config_by_path():
    path = PROJECT_ROOT / "configs" / "default.yaml"
    config = load_config(path)
    assert config["profile"] == "default"
    assert config["seeds"] == [42, 43, 44]
    assert get_nested(config, "train", "batch_size") == 16


def test_resolve_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        resolve_config_path("does-not-exist.yaml")


def test_get_nested_default():
    config = load_config("colab")
    assert get_nested(config, "paths", "drive_output_dir") is None
    assert get_nested(config, "missing", "key", default="fallback") == "fallback"
