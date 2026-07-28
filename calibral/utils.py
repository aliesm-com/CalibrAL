"""Shared utilities: logging and device helpers."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


_LOGGING_CONFIGURED = False
_DOTENV_LOADED = False
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no extra dependency). Does not override existing env."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED and path is None:
        return
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    if path is None:
        _DOTENV_LOADED = True


def setup_logging(level: int = logging.INFO, name: str = "calibral") -> logging.Logger:
    """Configure a simple console logger once and return the package logger."""
    global _LOGGING_CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _LOGGING_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(handler)
            root.setLevel(level)
        else:
            logger.addHandler(handler)
            logger.propagate = False
        _LOGGING_CONFIGURED = True

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the calibral namespace."""
    setup_logging()
    if name is None or name == "calibral":
        return logging.getLogger("calibral")
    if name.startswith("calibral."):
        return logging.getLogger(name)
    return logging.getLogger(f"calibral.{name}")


def detect_device() -> str:
    """Prefer CUDA when a GPU is usable; otherwise fall back to CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def describe_device() -> str:
    """Human-readable device string for logs (includes GPU name when CUDA)."""
    device = detect_device()
    if device != "cuda":
        return device

    try:
        import torch

        name = torch.cuda.get_device_name(0)
        return f"cuda ({name})"
    except Exception:
        return "cuda"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms can be slower; prefer reproducibility for research.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
