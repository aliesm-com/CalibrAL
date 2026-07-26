"""Tests for LLM labeler modes, validation, and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibral.config import load_config
from calibral.llm_labeler import (
    LLMLabelError,
    MissingAPIKeyError,
    MockLLMLabeler,
    ReplayLLMLabeler,
    build_prompt,
    create_labeler,
    normalize_shot_mode,
    resolve_labeler_mode,
    validate_llm_response,
)


def test_validate_llm_response_accepts_valid_payload():
    label, conf = validate_llm_response({"label": "positive", "confidence": 0.91})
    assert label == "positive"
    assert conf == 0.91


def test_validate_llm_response_rejects_bad_label():
    with pytest.raises(LLMLabelError):
        validate_llm_response({"label": "neutral", "confidence": 0.5})


def test_validate_llm_response_rejects_bad_confidence():
    with pytest.raises(LLMLabelError):
        validate_llm_response({"label": "negative", "confidence": 1.5})


def test_validate_parses_fenced_json_string():
    raw = '```json\n{"label": "negative", "confidence": 0.7}\n```'
    label, conf = validate_llm_response(raw)
    assert label == "negative"
    assert conf == 0.7


def test_normalize_shot_mode_aliases():
    assert normalize_shot_mode("zero") == "zero"
    assert normalize_shot_mode("1-shot") == "one"
    assert normalize_shot_mode("few-shot") == "multi"


def test_build_prompt_zero_has_no_examples():
    prompt = build_prompt("Target review", shot_mode="zero")
    assert "Example" not in prompt
    assert "Target review" in prompt


def test_build_prompt_one_includes_single_example():
    prompt = build_prompt("Target review", shot_mode="one")
    assert "Example:" in prompt
    assert prompt.count("Output:") == 1
    assert "Target review" in prompt


def test_build_prompt_multi_includes_multiple_examples():
    prompt = build_prompt("Target review", shot_mode="multi")
    assert "Examples:" in prompt
    assert prompt.count("Output:") >= 3
    assert "Target review" in prompt


def test_mock_labeler_is_deterministic():
    labeler = MockLLMLabeler()
    a = labeler.label_text("Great film")
    b = labeler.label_text("Great film")
    assert a == b
    assert a.mode == "mock"
    assert 0.0 <= a.confidence <= 1.0


def test_replay_labeler_reads_file(tmp_path: Path):
    path = tmp_path / "replay.json"
    path.write_text(
        json.dumps(
            [
                {"text": "hello", "label": "positive", "confidence": 0.88},
                {"text": "bad", "label": "negative", "confidence": 0.77},
            ]
        ),
        encoding="utf-8",
    )
    labeler = ReplayLLMLabeler(path, model="gpt-x", prompt_version="v1")
    result = labeler.label_text("hello")
    assert result.label == "positive"
    assert result.confidence == 0.88
    assert result.mode == "replay"


def test_smoke_api_without_key_falls_back_to_mock(monkeypatch):
    config = load_config("smoke")
    config["llm"]["mode"] = "api"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mode, warning = resolve_labeler_mode(config, allow_mock_fallback=True)
    assert mode == "mock"
    assert warning is not None
    labeler = create_labeler(config, allow_mock_fallback=True)
    assert labeler.mode == "mock"


def test_scientific_api_without_key_does_not_mock(monkeypatch):
    config = load_config("default")
    config["llm"]["mode"] = "api"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_labeler(config, allow_mock_fallback=False)


def test_create_labeler_mock_from_smoke_config():
    config = load_config("smoke")
    labeler = create_labeler(config)
    assert labeler.mode == "mock"
    result = labeler.label_text("An amazing movie with a weak ending.")
    assert result.label in {"positive", "negative"}
