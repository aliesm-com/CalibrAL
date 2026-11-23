"""Dataset loading (IMDB + other binary sentiment sets), cleaning, and splits."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibral.config import get_nested
from calibral.utils import get_logger

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_NAMES = {LABEL_NEGATIVE: "negative", LABEL_POSITIVE: "positive"}

@dataclass(frozen=True)
class DatasetSpec:
    """How to load a binary sentiment dataset from the Hugging Face Hub.

    Labels are mapped to 0=negative / 1=positive for every dataset.
    Schemas checked against the Hub dataset cards (Nov 2026).
    """

    hf_id: str
    hf_config: str | None
    text_fields: tuple[str, ...]
    label_field: str
    train_split: str
    test_split: str
    domain: str
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    # 25k/25k long movie reviews. 0=neg, 1=pos.
    "imdb": DatasetSpec("imdb", None, ("text",), "label", "train", "test", "movie reviews"),
    # 67k short movie-review sentences. Public test labels are hidden (-1),
    # so the 872-sample validation split is used as the test set.
    "sst2": DatasetSpec(
        "stanfordnlp/sst2", None, ("sentence",), "label", "train", "validation",
        "movie review sentences",
        note="test = official validation split (872); use data.test_size: all",
    ),
    # 560k/38k Yelp business reviews. Class names "1"/"2" = negative/positive.
    "yelp_polarity": DatasetSpec(
        "fancyzhx/yelp_polarity", None, ("text",), "label", "train", "test",
        "business reviews (Yelp)",
    ),
    # 3.6M/400k Amazon product reviews; text = title + content.
    "amazon_polarity": DatasetSpec(
        "fancyzhx/amazon_polarity", None, ("title", "content"), "label", "train", "test",
        "product reviews (Amazon)",
    ),
    # 8.5k/1k/1k movie-review snippets.
    "rotten_tomatoes": DatasetSpec(
        "cornell-movie-review-data/rotten_tomatoes", None, ("text",), "label",
        "train", "test", "movie review snippets",
        note="small: pool + validation + initial must fit in 8,530 train rows",
    ),
}


def dataset_spec(name: str) -> DatasetSpec:
    key = str(name or "imdb").strip().lower()
    if key not in DATASETS:
        raise ValueError(
            f"Unknown data.dataset_name={name!r}. Available: {sorted(DATASETS)}"
        )
    return DATASETS[key]


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass
class DataBundle:
    """Prepared splits for CalibrAL experiments.

    Pool gold labels are stored separately so selection algorithms only see
    ``pool_features`` (sample_id + text).
    """

    initial_labeled: pd.DataFrame
    pool_features: pd.DataFrame
    pool_labels: pd.Series
    validation: pd.DataFrame
    test: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def pool_gold_for_ids(self, sample_ids: list[str] | pd.Index) -> pd.Series:
        """Reveal gold labels only for explicitly queried pool sample IDs."""
        return self.pool_labels.loc[list(sample_ids)]


def clean_text(text: str, max_length: int | None = None) -> str:
    """Minimal HTML/whitespace cleanup suitable for IMDB reviews."""
    if text is None:
        return ""
    cleaned = _BR_RE.sub(" ", str(text))
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if max_length is not None and max_length > 0 and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _stratified_take(
    df: pd.DataFrame,
    size: int,
    rng: np.random.Generator,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take a roughly class-balanced subset; return (taken, remainder)."""
    if size < 0:
        raise ValueError(f"{name} size must be >= 0, got {size}")
    if size == 0:
        empty = df.iloc[0:0].copy()
        return empty, df.copy()
    if size > len(df):
        raise ValueError(
            f"Cannot take {size} rows for {name}; only {len(df)} available. "
            "Reduce split sizes in the config."
        )

    n_pos = size // 2
    n_neg = size - n_pos
    pos = df[df["label"] == LABEL_POSITIVE]
    neg = df[df["label"] == LABEL_NEGATIVE]
    if len(pos) < n_pos or len(neg) < n_neg:
        raise ValueError(
            f"Not enough class-balanced rows for {name}: "
            f"need {n_pos} positive / {n_neg} negative, "
            f"have {len(pos)} / {len(neg)}."
        )

    pos_idx = rng.choice(pos.index.to_numpy(), size=n_pos, replace=False)
    neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=False)
    taken_idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(taken_idx)

    taken = df.loc[taken_idx].copy().reset_index(drop=True)
    remainder = df.drop(index=taken_idx).copy().reset_index(drop=True)
    return taken, remainder


def create_splits_from_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    initial_labeled_size: int,
    pool_size: int,
    validation_size: int,
    test_size: int | str | None,
    seed: int,
    max_text_length: int | None = None,
) -> DataBundle:
    """Build deterministic splits from already-loaded train/test frames."""
    for frame, name in ((train_df, "train_df"), (test_df, "test_df")):
        _require_columns(frame, ["sample_id", "text", "label"], name)

    train = train_df.copy()
    test_source = test_df.copy()
    train["text"] = train["text"].map(lambda t: clean_text(t, max_text_length))
    test_source["text"] = test_source["text"].map(
        lambda t: clean_text(t, max_text_length)
    )
    train["label"] = train["label"].astype(int)
    test_source["label"] = test_source["label"].astype(int)

    rng = np.random.default_rng(seed)

    if test_size is None or str(test_size).strip().lower() in {"all", "-1"}:
        # Use the complete official test split (keeps its natural class ratio).
        test = test_source.copy().reset_index(drop=True)
    else:
        test, _ = _stratified_take(test_source, int(test_size), rng, "test")
    validation, rest = _stratified_take(train, validation_size, rng, "validation")
    initial_labeled, rest = _stratified_take(
        rest, initial_labeled_size, rng, "initial_labeled"
    )
    pool, _ = _stratified_take(rest, pool_size, rng, "pool")

    pool_features = pool[["sample_id", "text"]].copy()
    pool_labels = pool.set_index("sample_id")["label"].astype(int)

    _assert_no_id_overlap(
        {
            "initial_labeled": initial_labeled,
            "pool": pool_features,
            "validation": validation,
            "test": test,
        }
    )

    meta = {
        "seed": seed,
        "sizes": {
            "initial_labeled": len(initial_labeled),
            "pool": len(pool_features),
            "validation": len(validation),
            "test": len(test),
        },
        "class_ratios": {
            name: _class_ratio(frame)
            for name, frame in (
                ("initial_labeled", initial_labeled),
                ("pool", pool.assign(label=pool_labels.values)),
                ("validation", validation),
                ("test", test),
            )
        },
    }
    return DataBundle(
        initial_labeled=initial_labeled.reset_index(drop=True),
        pool_features=pool_features.reset_index(drop=True),
        pool_labels=pool_labels,
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        meta=meta,
    )


def _class_ratio(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0 or "label" not in df.columns:
        return {"negative": 0.0, "positive": 0.0}
    counts = df["label"].value_counts(normalize=True)
    return {
        "negative": float(counts.get(LABEL_NEGATIVE, 0.0)),
        "positive": float(counts.get(LABEL_POSITIVE, 0.0)),
    }


def _assert_no_id_overlap(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for split_name, frame in splits.items():
        for sample_id in frame["sample_id"].astype(str):
            if sample_id in seen:
                raise ValueError(
                    f"Sample ID overlap: {sample_id!r} appears in "
                    f"{seen[sample_id]!r} and {split_name!r}."
                )
            seen[sample_id] = split_name


def load_dataset_frames(
    dataset_name: str, cache_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load any registered dataset as (train, test) frames: sample_id/text/label."""
    spec = dataset_spec(dataset_name)
    if spec.hf_id == "imdb":
        return load_imdb_frames(cache_dir)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("data")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install with: pip install -r requirements.txt"
        ) from exc

    logger.info("Loading %s (%s) from Hugging Face...", dataset_name, spec.hf_id)
    try:
        dataset = load_dataset(spec.hf_id, spec.hf_config, cache_dir=str(cache_dir / "hf"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download/load {spec.hf_id}. Check network access to "
            f"Hugging Face. Original error: {exc}"
        ) from exc

    prefix = str(dataset_name).strip().lower()

    def _to_frame(split_name: str) -> pd.DataFrame:
        split = dataset[split_name]
        columns = [list(split[f]) for f in spec.text_fields]
        texts = [
            ". ".join(str(part).strip() for part in parts if str(part or "").strip())
            for parts in zip(*columns)
        ]
        frame = pd.DataFrame(
            {
                "sample_id": [f"{prefix}_{split_name}_{i}" for i in range(len(split))],
                "text": texts,
                "label": list(split[spec.label_field]),
            }
        )
        bad = frame[~frame["label"].isin([LABEL_NEGATIVE, LABEL_POSITIVE])]
        if len(bad):
            raise ValueError(
                f"{spec.hf_id}:{split_name} has {len(bad)} rows with labels outside "
                "{0, 1} (hidden test labels?). Use another split as test."
            )
        return frame

    train_df = _to_frame(spec.train_split)
    test_df = _to_frame(spec.test_split)
    logger.info("%s loaded: train=%d, test=%d", dataset_name, len(train_df), len(test_df))
    return train_df, test_df


def load_imdb_frames(cache_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download/load IMDB via Hugging Face Datasets and return train/test frames."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("data")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    logger.info("Loading IMDB dataset (Hugging Face datasets)...")
    try:
        dataset = load_dataset(
            "imdb",
            cache_dir=str(cache_dir / "hf"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download/load IMDB. Check network access to Hugging Face, "
            "then retry: python run.py smoke --stage data\n"
            f"Original error: {exc}"
        ) from exc

    def _to_frame(split_name: str) -> pd.DataFrame:
        split = dataset[split_name]
        frame = pd.DataFrame(
            {
                "sample_id": [f"{split_name}_{i}" for i in range(len(split))],
                "text": list(split["text"]),
                "label": list(split["label"]),
            }
        )
        return frame

    train_df = _to_frame("train")
    test_df = _to_frame("test")
    logger.info(
        "IMDB loaded: train=%d, test=%d",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


def _cache_key(config: dict[str, Any]) -> str:
    profile = config.get("profile", "default")
    seed = int(config.get("seed", 42))
    data = config.get("data", {})
    parts = [
        profile,
        f"seed{seed}",
        f"i{int(data.get('initial_labeled_size', 0))}",
        f"p{int(data.get('pool_size', 0))}",
        f"v{int(data.get('validation_size', 0))}",
        f"t{data.get('test_size', 0)}",
    ]
    name = str(data.get("dataset_name", "imdb")).strip().lower()
    if name != "imdb":  # keep existing IMDB cache keys unchanged
        parts.insert(1, name)
    return "_".join(parts)


def _split_cache_dir(config: dict[str, Any]) -> Path:
    cache_root = Path(get_nested(config, "paths", "cache_dir", default="outputs/cache"))
    return cache_root / "splits" / _cache_key(config)


def save_data_bundle(bundle: DataBundle, directory: str | Path) -> Path:
    """Persist splits and metadata under directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    bundle.initial_labeled.to_csv(directory / "initial_labeled.csv", index=False)
    bundle.pool_features.to_csv(directory / "pool_features.csv", index=False)
    bundle.validation.to_csv(directory / "validation.csv", index=False)
    bundle.test.to_csv(directory / "test.csv", index=False)

    # Gold pool labels kept for oracle use only; not passed to selectors.
    pool_labels_df = bundle.pool_labels.rename("label").reset_index()
    pool_labels_df.to_csv(directory / "pool_labels_oracle_only.csv", index=False)

    with (directory / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle.meta, handle, indent=2, ensure_ascii=False)

    return directory


def load_data_bundle(directory: str | Path) -> DataBundle:
    """Load a previously saved DataBundle from directory."""
    directory = Path(directory)
    initial_labeled = pd.read_csv(directory / "initial_labeled.csv")
    pool_features = pd.read_csv(directory / "pool_features.csv")
    validation = pd.read_csv(directory / "validation.csv")
    test = pd.read_csv(directory / "test.csv")
    pool_labels_df = pd.read_csv(directory / "pool_labels_oracle_only.csv")
    pool_labels = pool_labels_df.set_index("sample_id")["label"].astype(int)
    with (directory / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return DataBundle(
        initial_labeled=initial_labeled,
        pool_features=pool_features,
        pool_labels=pool_labels,
        validation=validation,
        test=test,
        meta=meta,
    )


def prepare_data(
    config: dict[str, Any],
    *,
    force_reload: bool = False,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> DataBundle:
    """Prepare IMDB splits according to config; cache results on disk."""
    logger = get_logger("data")
    cache_dir = Path(get_nested(config, "paths", "cache_dir", default="outputs/cache"))
    split_dir = _split_cache_dir(config)

    if split_dir.exists() and not force_reload and train_df is None and test_df is None:
        logger.info("Loading cached splits from %s", split_dir)
        return load_data_bundle(split_dir)

    data_cfg = config.get("data", {})
    if train_df is None or test_df is None:
        train_df, test_df = load_dataset_frames(
            str(data_cfg.get("dataset_name", "imdb")), cache_dir
        )

    bundle = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=int(data_cfg.get("initial_labeled_size", 0)),
        pool_size=int(data_cfg.get("pool_size", 0)),
        validation_size=int(data_cfg.get("validation_size", 0)),
        test_size=data_cfg.get("test_size", 0),
        seed=int(config.get("seed", 42)),
        max_text_length=data_cfg.get("max_text_length"),
    )
    bundle.meta["profile"] = config.get("profile")
    bundle.meta["dataset_name"] = data_cfg.get("dataset_name", "imdb")
    bundle.meta["cache_dir"] = str(split_dir)

    save_data_bundle(bundle, split_dir)
    logger.info("Saved splits to %s", split_dir)
    return bundle


def format_data_summary(bundle: DataBundle) -> str:
    """Return a human-readable summary of split sizes and class ratios."""
    lines = ["Data split summary"]
    sizes = bundle.meta.get("sizes", {})
    ratios = bundle.meta.get("class_ratios", {})
    for name in ("initial_labeled", "pool", "validation", "test"):
        size = sizes.get(name, "?")
        ratio = ratios.get(name, {})
        neg = ratio.get("negative", float("nan"))
        pos = ratio.get("positive", float("nan"))
        lines.append(
            f"  - {name}: n={size} | neg={neg:.2f} | pos={pos:.2f}"
        )
    lines.append(f"  seed: {bundle.meta.get('seed')}")
    lines.append(f"  cache: {bundle.meta.get('cache_dir', 'n/a')}")
    lines.append(
        "  note: pool gold labels are hidden from selection "
        "(oracle-only file on disk)."
    )
    return "\n".join(lines)


def print_data_summary(bundle: DataBundle) -> None:
    """Print split summary to stdout."""
    print(format_data_summary(bundle))
