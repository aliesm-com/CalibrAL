"""Tests for IMDB cleaning, splitting, and leakage controls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibral.config import load_config
from calibral.data import (
    clean_text,
    create_splits_from_frames,
    prepare_data,
)


def _synthetic_frames(n_train: int = 200, n_test: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    for i in range(n_train):
        label = i % 2
        train_rows.append(
            {
                "sample_id": f"train_{i}",
                "text": f"<br />Review {i} class {label}",
                "label": label,
            }
        )
    test_rows = []
    for i in range(n_test):
        label = i % 2
        test_rows.append(
            {
                "sample_id": f"test_{i}",
                "text": f"<p>Test {i}</p>",
                "label": label,
            }
        )
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def test_clean_text_strips_html_and_whitespace():
    raw = "<br /><b>Great</b>   movie&amp;show"
    cleaned = clean_text(raw)
    assert cleaned == "Great movie&show"
    assert "<" not in cleaned
    assert "  " not in cleaned


def test_splits_have_no_overlapping_ids():
    train_df, test_df = _synthetic_frames()
    bundle = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    ids = (
        set(bundle.initial_labeled["sample_id"])
        | set(bundle.pool_features["sample_id"])
        | set(bundle.validation["sample_id"])
        | set(bundle.test["sample_id"])
    )
    total = (
        len(bundle.initial_labeled)
        + len(bundle.pool_features)
        + len(bundle.validation)
        + len(bundle.test)
    )
    assert len(ids) == total


def test_splits_are_deterministic_with_same_seed():
    train_df, test_df = _synthetic_frames()
    kwargs = dict(
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    a = create_splits_from_frames(train_df, test_df, **kwargs)
    b = create_splits_from_frames(train_df, test_df, **kwargs)
    assert list(a.initial_labeled["sample_id"]) == list(b.initial_labeled["sample_id"])
    assert list(a.pool_features["sample_id"]) == list(b.pool_features["sample_id"])
    assert list(a.validation["sample_id"]) == list(b.validation["sample_id"])
    assert list(a.test["sample_id"]) == list(b.test["sample_id"])


def test_different_seed_changes_split():
    train_df, test_df = _synthetic_frames()
    a = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    b = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=99,
    )
    assert list(a.pool_features["sample_id"]) != list(b.pool_features["sample_id"])


def test_class_balance_is_approximate():
    train_df, test_df = _synthetic_frames()
    bundle = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    for name, frame in (
        ("initial_labeled", bundle.initial_labeled),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        ratio = frame["label"].mean()
        assert 0.4 <= ratio <= 0.6, f"{name} imbalance: {ratio}"
    pool_ratio = bundle.pool_labels.mean()
    assert 0.4 <= pool_ratio <= 0.6


def test_pool_features_hide_labels():
    train_df, test_df = _synthetic_frames()
    bundle = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    assert "label" not in bundle.pool_features.columns
    assert list(bundle.pool_features.columns) == ["sample_id", "text"]
    revealed = bundle.pool_gold_for_ids(bundle.pool_features["sample_id"].head(3))
    assert len(revealed) == 3


def test_test_ids_never_appear_in_train_side_splits():
    train_df, test_df = _synthetic_frames()
    bundle = create_splits_from_frames(
        train_df,
        test_df,
        initial_labeled_size=8,
        pool_size=32,
        validation_size=16,
        test_size=16,
        seed=42,
    )
    test_ids = set(bundle.test["sample_id"])
    train_side = (
        set(bundle.initial_labeled["sample_id"])
        | set(bundle.pool_features["sample_id"])
        | set(bundle.validation["sample_id"])
    )
    assert test_ids.isdisjoint(train_side)
    assert all(str(i).startswith("test_") for i in test_ids)


def test_prepare_data_caches_to_disk(tmp_path: Path):
    train_df, test_df = _synthetic_frames()
    config = load_config("smoke")
    config["paths"] = {"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path)}
    bundle = prepare_data(config, train_df=train_df, test_df=test_df)
    cache_dir = Path(bundle.meta["cache_dir"])
    assert (cache_dir / "initial_labeled.csv").exists()
    assert (cache_dir / "pool_features.csv").exists()
    assert (cache_dir / "pool_labels_oracle_only.csv").exists()
    assert (cache_dir / "validation.csv").exists()
    assert (cache_dir / "test.csv").exists()
    assert (cache_dir / "meta.json").exists()

    # Second call should load cache (still valid with injected frames skipped when cache hits)
    cached = prepare_data(config)
    assert list(cached.initial_labeled["sample_id"]) == list(
        bundle.initial_labeled["sample_id"]
    )
