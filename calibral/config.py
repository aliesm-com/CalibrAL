"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = {
    "smoke": PROJECT_ROOT / "configs" / "smoke.yaml",
    "default": PROJECT_ROOT / "configs" / "default.yaml",
    "colab": PROJECT_ROOT / "configs" / "colab.yaml",
}


def resolve_config_path(config: str | Path | None = None) -> Path:
    """Resolve a profile name or filesystem path to a config YAML file."""
    if config is None:
        return DEFAULT_CONFIGS["default"]

    path = Path(config)
    if path.exists() and path.is_file():
        return path.resolve()

    key = str(config).strip().lower()
    if key in DEFAULT_CONFIGS:
        resolved = DEFAULT_CONFIGS[key]
        if not resolved.exists():
            raise FileNotFoundError(f"Built-in config not found: {resolved}")
        return resolved

    raise FileNotFoundError(
        f"Config not found: {config!r}. "
        f"Use a path or one of: {', '.join(sorted(DEFAULT_CONFIGS))}."
    )


def load_config(config: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config into a plain dict."""
    path = resolve_config_path(config)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    data["_config_path"] = str(path)
    return data


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested config value without raising on missing keys."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_output_dir(config: dict[str, Any], *, create: bool = True) -> Path:
    """Resolve experiment output folder, namespaced by profile when needed.

    Keeps each experiment's CSV/plots/summaries separate so runs do not overwrite
    each other. Shared dataset cache stays under ``paths.cache_dir`` (typically
    ``outputs/cache``).
    """
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        config["paths"] = paths

    profile = str(config.get("profile") or "run").strip() or "run"
    configured = paths.get("output_dir", "outputs")
    path = Path(str(configured))

    # Flat "outputs" / "/content/outputs" → nest under profile.
    posix = path.as_posix().rstrip("/")
    if posix in {"outputs", "./outputs", "/content/outputs"}:
        path = path / profile

    paths["output_dir"] = str(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

