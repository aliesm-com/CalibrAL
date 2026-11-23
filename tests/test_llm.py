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


# --- prompt v2 / robust parsing / request options -------------------------


def test_prompt_v1_is_unchanged_for_reproducibility():
    prompt = build_prompt("Target review", prompt_version="v1")
    assert '{{"label"' in prompt


def test_prompt_v2_has_single_brace_json_template():
    prompt = build_prompt("Target review", prompt_version="v2")
    assert "{{" not in prompt
    assert '{"label"' in prompt
    assert "Target review" in prompt


def test_build_prompt_rejects_unknown_version():
    with pytest.raises(ValueError):
        build_prompt("x", prompt_version="v9")


def test_validate_strips_think_block():
    raw = '<think>The reviewer hated it.</think>\n{"label": "negative", "confidence": 0.9}'
    assert validate_llm_response(raw) == ("negative", 0.9)


def test_validate_handles_doubled_braces_from_v1_template():
    raw = '{{"label": "positive", "confidence": 0.8}}'
    assert validate_llm_response(raw) == ("positive", 0.8)


def test_validate_handles_trailing_text_after_object():
    raw = 'Sure! {"label": "positive", "confidence": 0.7} Hope this helps {:)'
    assert validate_llm_response(raw) == ("positive", 0.7)


def test_api_request_body_defaults_match_legacy():
    from calibral.llm_labeler import APILLMLabeler

    labeler = APILLMLabeler(base_url="http://x", model="m", api_key="k")
    body = labeler._build_body("p", with_optional=True)
    assert set(body) == {"model", "temperature", "messages"}


def test_api_request_body_with_paper_options():
    from calibral.llm_labeler import APILLMLabeler

    labeler = APILLMLabeler(
        base_url="http://x",
        model="m",
        api_key="k",
        reasoning="off",
        json_mode=True,
        max_tokens=512,
        seed=42,
        extra_body={"provider": {"allow_fallbacks": False}},
    )
    body = labeler._build_body("p", with_optional=True)
    assert body["reasoning"] == {"enabled": False}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 512
    assert body["seed"] == 42
    assert body["provider"] == {"allow_fallbacks": False}
    plain = labeler._build_body("p", with_optional=False)
    assert "reasoning" not in plain and "response_format" not in plain


def test_empty_content_raises_clear_error(monkeypatch):
    from calibral.llm_labeler import APILLMLabeler

    labeler = APILLMLabeler(base_url="http://x", model="m", api_key="k", max_retries=0)
    monkeypatch.setattr(
        labeler,
        "_post",
        lambda body: {"choices": [{"message": {"content": None}, "finish_reason": "length"}]},
    )
    with pytest.raises(LLMLabelError, match="Empty content"):
        labeler.label_text("review")


def test_probe_metrics_helpers():
    from calibral.bakeoff import _calibration_metrics, _wilson_interval

    lo, hi = _wilson_interval(19, 20)
    assert 0.7 < lo < 0.95 < hi <= 1.0
    records = [
        {"pred": 1, "confidence": 1.0, "correct": True},
        {"pred": 0, "confidence": 1.0, "correct": False},
        {"pred": None, "confidence": None, "correct": False},
    ]
    metrics = _calibration_metrics(records)
    assert metrics["brier"] == 0.5
    assert metrics["ece"] == 0.5


def test_paper_config_loads_with_bakeoff_section():
    config = load_config("configs/paper_al.yaml")
    assert config["llm"]["prompt_version"] == "v2"
    assert config["bakeoff"]["n_probe_samples"] >= 100
    assert all("~" not in m and "latest" not in m for m in config["bakeoff"]["llm_candidates"])
