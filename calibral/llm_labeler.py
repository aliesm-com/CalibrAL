"""LLM labeling backends: mock, replay, and OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from calibral.utils import get_logger

LabelName = Literal["positive", "negative"]
ShotMode = Literal["zero", "one", "multi"]
VALID_LABELS = {"positive", "negative"}
VALID_SHOT_MODES = {"zero", "one", "multi"}

# Fixed demonstration examples for one-shot / multi-shot prompts (not from the test set).
SHOT_EXAMPLES = [
    {
        "text": "An outstanding film with brilliant performances and a moving story.",
        "label": "positive",
        "confidence": 0.95,
    },
    {
        "text": "A dull waste of time with weak acting and a pointless plot.",
        "label": "negative",
        "confidence": 0.93,
    },
    {
        "text": "I loved every minute of this movie; highly recommended.",
        "label": "positive",
        "confidence": 0.92,
    },
]

PROMPT_HEADER = (
    "You are a sentiment labeler for movie reviews.\n"
    "Classify the review as positive or negative.\n"
    "Respond with ONLY valid JSON of the form:\n"
    '{{"label": "positive"|"negative", "confidence": 0.0-1.0}}'
)


def normalize_shot_mode(value: str | None) -> ShotMode:
    """Normalize YAML aliases to zero | one | multi."""
    raw = str(value or "zero").strip().lower().replace("_", "-")
    aliases = {
        "zero": "zero",
        "0": "zero",
        "0-shot": "zero",
        "zeroshot": "zero",
        "one": "one",
        "1": "one",
        "1-shot": "one",
        "oneshot": "one",
        "multi": "multi",
        "few": "multi",
        "few-shot": "multi",
        "fewshot": "multi",
        "multishot": "multi",
        "multi-shot": "multi",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(
            f"Unsupported llm.shot_mode={value!r}. Use zero, one, or multi."
        )
    return mode  # type: ignore[return-value]


def _format_example(example: dict[str, Any]) -> str:
    return (
        f"Review:\n{example['text']}\n"
        f'Output: {{"label": "{example["label"]}", "confidence": {example["confidence"]}}}'
    )


def build_prompt(
    text: str,
    prompt_version: str = "v1",
    shot_mode: str = "zero",
) -> str:
    """Build the user prompt for zero/one/multi-shot labeling."""
    if prompt_version != "v1":
        raise ValueError(f"Unsupported prompt_version: {prompt_version!r}")

    mode = normalize_shot_mode(shot_mode)
    parts = [PROMPT_HEADER]

    if mode == "one":
        parts.append("\n\nExample:")
        parts.append(_format_example(SHOT_EXAMPLES[0]))
    elif mode == "multi":
        parts.append("\n\nExamples:")
        for ex in SHOT_EXAMPLES:
            parts.append(_format_example(ex))
            parts.append("")

    parts.append(f"\n\nReview:\n{text}")
    return "\n".join(parts).strip() + "\n"


@dataclass
class LLMLabelResult:
    label: LabelName
    confidence: float
    mode: str
    cached: bool = False
    source: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMLabelError(Exception):
    """Raised when labeling fails after retries or validation."""


class MissingAPIKeyError(LLMLabelError):
    """Raised when a scientific/API run requires a key that is absent."""


def validate_llm_response(payload: Any) -> tuple[LabelName, float]:
    """Validate LLM JSON payload and return (label, confidence)."""
    if isinstance(payload, str):
        payload = _extract_json_object(payload)

    if not isinstance(payload, dict):
        raise LLMLabelError(f"LLM response must be a JSON object, got {type(payload).__name__}")

    label = payload.get("label")
    confidence = payload.get("confidence")

    if not isinstance(label, str) or label.lower().strip() not in VALID_LABELS:
        raise LLMLabelError(
            f"Invalid label {label!r}. Expected one of {sorted(VALID_LABELS)}."
        )
    try:
        conf = float(confidence)
    except (TypeError, ValueError) as exc:
        raise LLMLabelError(f"Invalid confidence {confidence!r}.") from exc
    if not 0.0 <= conf <= 1.0:
        raise LLMLabelError(f"Confidence out of range [0, 1]: {conf}")

    return label.lower().strip(), conf  # type: ignore[return-value]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    raise LLMLabelError(f"Could not parse JSON from LLM text: {text[:200]!r}")


def cache_key(text: str, model: str, prompt_version: str, shot_mode: str = "zero") -> str:
    digest = hashlib.sha256(
        f"{prompt_version}\n{normalize_shot_mode(shot_mode)}\n{model}\n{text}".encode(
            "utf-8"
        )
    ).hexdigest()
    return digest


class ResponseCache:
    """Disk cache keyed by text + model + prompt version + shot mode."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        text: str,
        model: str,
        prompt_version: str,
        shot_mode: str = "zero",
    ) -> dict[str, Any] | None:
        path = self.directory / f"{cache_key(text, model, prompt_version, shot_mode)}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(
        self,
        text: str,
        model: str,
        prompt_version: str,
        payload: dict[str, Any],
        shot_mode: str = "zero",
    ) -> None:
        path = self.directory / f"{cache_key(text, model, prompt_version, shot_mode)}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


class BaseLLMLabeler:
    mode: str = "base"

    def label_text(self, text: str) -> LLMLabelResult:
        raise NotImplementedError

    def label_many(self, texts: list[str]) -> list[LLMLabelResult]:
        return [self.label_text(t) for t in texts]


class MockLLMLabeler(BaseLLMLabeler):
    """Deterministic fake labeler for software smoke tests only."""

    mode = "mock"

    def __init__(self, model: str = "mock", prompt_version: str = "v1") -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.logger = get_logger("llm")

    def label_text(self, text: str) -> LLMLabelResult:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        label: LabelName = "positive" if int(digest[:8], 16) % 2 == 0 else "negative"
        # Map hash to a reproducible confidence in [0.55, 0.99]
        confidence = 0.55 + (int(digest[8:16], 16) % 45) / 100.0
        self.logger.debug("Mock label generated (NOT scientific)")
        return LLMLabelResult(
            label=label,
            confidence=float(confidence),
            mode=self.mode,
            cached=False,
            source="mock",
        )


class ReplayLLMLabeler(BaseLLMLabeler):
    """Replay previously stored responses for cheap reproducible runs."""

    mode = "replay"

    def __init__(
        self,
        replay_path: str | Path,
        *,
        model: str,
        prompt_version: str,
    ) -> None:
        self.replay_path = Path(replay_path)
        self.model = model
        self.prompt_version = prompt_version
        self.logger = get_logger("llm")
        self._index = self._load_index(self.replay_path)

    @staticmethod
    def _load_index(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Replay file not found: {path}. "
                "Create a JSON/JSONL file of prior LLM responses."
            )
        index: dict[str, dict[str, Any]] = {}
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = row.get("text")
                    if key is None:
                        raise LLMLabelError(f"Replay JSONL line {line_no} missing 'text'")
                    index[str(key)] = row
        else:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "items" in data:
                data = data["items"]
            if not isinstance(data, list):
                raise LLMLabelError("Replay JSON must be a list or {'items': [...]}")
            for row in data:
                index[str(row["text"])] = row
        return index

    def label_text(self, text: str) -> LLMLabelResult:
        row = self._index.get(text)
        if row is None:
            # Also allow lookup by cache key for compact replay files
            key = cache_key(text, self.model, self.prompt_version)
            row = self._index.get(key)
        if row is None:
            raise LLMLabelError(
                f"No replay entry for text (len={len(text)}). "
                f"Add it to {self.replay_path}"
            )
        label, confidence = validate_llm_response(
            {"label": row.get("label"), "confidence": row.get("confidence")}
        )
        return LLMLabelResult(
            label=label,
            confidence=confidence,
            mode=self.mode,
            cached=False,
            source="replay",
        )


class APILLMLabeler(BaseLLMLabeler):
    """OpenAI-compatible Chat Completions labeler."""

    mode = "api"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        prompt_version: str = "v1",
        shot_mode: str = "zero",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        cache: ResponseCache | None = None,
    ) -> None:
        if not api_key:
            raise MissingAPIKeyError("API key is empty.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.prompt_version = prompt_version
        self.shot_mode = normalize_shot_mode(shot_mode)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.cache = cache
        self.logger = get_logger("llm")
        # Never log the key itself.
        self.logger.info(
            "API labeler ready | base_url=%s | model=%s | shot_mode=%s | key_present=True",
            self.base_url,
            self.model,
            self.shot_mode,
        )

    def label_text(self, text: str) -> LLMLabelResult:
        if self.cache is not None:
            cached = self.cache.get(
                text, self.model, self.prompt_version, self.shot_mode
            )
            if cached is not None:
                label, confidence = validate_llm_response(cached)
                return LLMLabelResult(
                    label=label,
                    confidence=confidence,
                    mode=self.mode,
                    cached=True,
                    source="cache",
                )

        prompt = build_prompt(text, self.prompt_version, self.shot_mode)
        last_error: Exception | None = None
        attempts = max(self.max_retries, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                payload = self._call_api(prompt)
                label, confidence = validate_llm_response(payload)
                result_payload = {"label": label, "confidence": confidence}
                if self.cache is not None:
                    self.cache.set(
                        text,
                        self.model,
                        self.prompt_version,
                        result_payload,
                        self.shot_mode,
                    )
                return LLMLabelResult(
                    label=label,
                    confidence=confidence,
                    mode=self.mode,
                    cached=False,
                    source="api",
                )
            except Exception as exc:  # noqa: BLE001 - retry transient/API/parse errors
                last_error = exc
                self.logger.warning(
                    "API label attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    type(exc).__name__,
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise LLMLabelError(f"API labeling failed after {attempts} attempts: {last_error}")

    def _call_api(self, prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # Avoid echoing secrets if servers reflect headers.
            raise LLMLabelError(f"HTTP {exc.code} from LLM API: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise LLMLabelError(f"LLM API connection error: {exc.reason}") from exc

        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMLabelError(f"Unexpected API response shape: {str(data)[:300]}") from exc
        return _extract_json_object(content)


def _read_api_key(api_key_env: str) -> str | None:
    value = os.environ.get(api_key_env)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_labeler_mode(
    config: dict[str, Any],
    *,
    allow_mock_fallback: bool | None = None,
) -> tuple[str, str | None]:
    """Resolve effective LLM mode.

    Returns (mode, warning_message).
    For scientific profiles, missing API keys do not silently fall back to mock.
    """
    llm_cfg = config.get("llm", {})
    requested = str(llm_cfg.get("mode", "mock")).lower().strip()
    profile = str(config.get("profile", "default")).lower()
    is_smoke = profile == "smoke" or "NOT_SCIENTIFIC_SMOKE_RESULT" in str(
        config.get("notes", "")
    )

    if allow_mock_fallback is None:
        allow_mock_fallback = is_smoke

    if requested not in {"mock", "replay", "api"}:
        raise ValueError(
            f"Unsupported llm.mode={requested!r}. Use mock, replay, or api."
        )

    if requested == "api":
        api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if _read_api_key(api_key_env) is None:
            message = (
                f"Environment variable {api_key_env} is not set. "
                "Real LLM results were not produced."
            )
            if allow_mock_fallback:
                return "mock", message + " Falling back to mock for smoke/testing only."
            raise MissingAPIKeyError(
                message
                + " For scientific runs set the API key, or set llm.mode to replay "
                "with a documented replay file. Refusing silent mock fallback."
            )
    return requested, None


def create_labeler(
    config: dict[str, Any],
    *,
    allow_mock_fallback: bool | None = None,
) -> BaseLLMLabeler:
    """Factory for mock / replay / api labelers based on config."""
    logger = get_logger("llm")
    llm_cfg = config.get("llm", {})
    mode, warning = resolve_labeler_mode(
        config, allow_mock_fallback=allow_mock_fallback
    )
    if warning:
        logger.warning(warning)

    model = str(llm_cfg.get("model", "unknown"))
    prompt_version = str(llm_cfg.get("prompt_version", "v1"))
    shot_mode = normalize_shot_mode(llm_cfg.get("shot_mode", "zero"))
    cache_root = Path(config.get("paths", {}).get("cache_dir", "outputs/cache"))
    cache = ResponseCache(cache_root / "llm")

    logger.info(
        "Creating LLM labeler | requested_mode=%s | effective_mode=%s | shot_mode=%s",
        llm_cfg.get("mode"),
        mode,
        shot_mode,
    )

    if mode == "mock":
        return MockLLMLabeler(model=model, prompt_version=prompt_version)

    if mode == "replay":
        replay_path = llm_cfg.get("replay_path")
        if not replay_path:
            raise ValueError(
                "llm.mode=replay requires llm.replay_path in the config."
            )
        return ReplayLLMLabeler(
            replay_path,
            model=model,
            prompt_version=prompt_version,
        )

    # api
    api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
    api_key = _read_api_key(api_key_env)
    assert api_key is not None  # ensured by resolve_labeler_mode
    return APILLMLabeler(
        base_url=str(llm_cfg.get("base_url", "https://api.openai.com/v1")),
        model=model,
        api_key=api_key,
        prompt_version=prompt_version,
        shot_mode=shot_mode,
        timeout_seconds=float(llm_cfg.get("timeout_seconds", 60)),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        cache=cache,
    )


def run_llm_smoke(config: dict[str, Any], texts: list[str] | None = None) -> dict[str, Any]:
    """Label a few texts for smoke testing and write a small artifact."""
    logger = get_logger("llm")
    if texts is None:
        from calibral.data import prepare_data

        bundle = prepare_data(config)
        texts = bundle.pool_features["text"].astype(str).head(4).tolist()

    labeler = create_labeler(config, allow_mock_fallback=True)
    results = labeler.label_many(texts)

    from calibral.config import resolve_output_dir

    output_dir = resolve_output_dir(config, create=True)
    artifact = {
        "project": "CalibrAL",
        "stage": "llm_smoke",
        "llm_mode": labeler.mode,
        "result_tag": "NOT_SCIENTIFIC_SMOKE_RESULT"
        if labeler.mode == "mock" or config.get("profile") == "smoke"
        else None,
        "n_texts": len(texts),
        "items": [
            {
                "text_preview": t[:120],
                "label": r.label,
                "confidence": r.confidence,
                "mode": r.mode,
                "cached": r.cached,
                "source": r.source,
            }
            for t, r in zip(texts, results)
        ],
        "note": (
            "Mock/smoke LLM outputs are for software path testing only "
            "and must not be mixed with scientific API/replay results."
        ),
    }
    path = output_dir / "llm_smoke.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)

    logger.info(
        "LLM smoke done | mode=%s | n=%d | artifact=%s",
        labeler.mode,
        len(results),
        path,
    )
    return {"artifact_path": str(path), "llm_mode": labeler.mode, **artifact}
