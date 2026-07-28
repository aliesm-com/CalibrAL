"""Text classification model helpers for CalibrAL."""

from __future__ import annotations

from typing import Any

from calibral.utils import get_logger


def build_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer for ``model_name``."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'transformers'. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    logger = get_logger("model")
    logger.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str, num_labels: int = 2):
    """Load a sequence-classification transformer."""
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'transformers'. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    logger = get_logger("model")
    logger.info("Loading model: %s (num_labels=%d)", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model


def move_model_to_device(model: Any, device: str) -> Any:
    """Move model to the requested device and return it."""
    return model.to(device)
