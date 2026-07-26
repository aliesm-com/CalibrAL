"""Uncertainty-based active learning with LLM silver labels and human oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from calibral.data import LABEL_NEGATIVE, LABEL_POSITIVE, DataBundle
from calibral.evaluation import predict
from calibral.llm_labeler import create_labeler
from calibral.model import build_model, move_model_to_device
from calibral.trainer import train_model
from calibral.utils import get_logger

_LLM_TO_INT = {"negative": LABEL_NEGATIVE, "positive": LABEL_POSITIVE}


def uncertainty_from_prob_positive(
    prob_positive: np.ndarray | list[float],
    method: str = "entropy",
) -> np.ndarray:
    """Score uncertainty; higher means more uncertain."""
    p = np.asarray(prob_positive, dtype=float)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    method = method.lower().strip()
    if method == "entropy":
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    if method in {"margin", "distance", "distance_to_half"}:
        # Distance from 0.5 inverted: closer to 0.5 => higher uncertainty
        return 1.0 - (2.0 * np.abs(p - 0.5))
    raise ValueError(
        f"Unknown uncertainty method {method!r}. Use 'entropy' or 'margin'."
    )


def select_uncertain_ids(
    pool_features: pd.DataFrame,
    model,
    tokenizer,
    *,
    device: str,
    max_length: int,
    batch_size: int,
    k: int,
    method: str = "entropy",
) -> pd.DataFrame:
    """Return the ``k`` most uncertain pool rows (sample_id, text, uncertainty)."""
    if k <= 0 or len(pool_features) == 0:
        return pool_features.iloc[0:0].copy()

    preds = predict(
        model,
        tokenizer,
        pool_features,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    scores = uncertainty_from_prob_positive(
        preds["prob_positive"].to_numpy(), method=method
    )
    ranked = preds.copy()
    ranked["uncertainty"] = scores
    ranked = ranked.sort_values("uncertainty", ascending=False)
    top = ranked.head(min(k, len(ranked)))
    merged = pool_features.merge(
        top[["sample_id", "uncertainty", "prob_positive"]],
        on="sample_id",
        how="inner",
    )
    return merged.sort_values("uncertainty", ascending=False).reset_index(drop=True)


def query_human_oracle(bundle: DataBundle, sample_ids: list[str]) -> pd.Series:
    """Reveal gold pool labels only for explicitly requested IDs."""
    return bundle.pool_gold_for_ids(sample_ids)


def _ensure_weight_column(df: pd.DataFrame, default: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    if "weight" not in out.columns:
        out["weight"] = default
    if "label_source" not in out.columns:
        out["label_source"] = "gold"
    return out


def run_proposed_active_learning(
    config: dict[str, Any],
    bundle: DataBundle,
    *,
    tokenizer,
    device: str,
) -> dict[str, Any]:
    """Run the proposed hybrid AL loop and return model + accounting stats."""
    logger = get_logger("active_learning")
    al_cfg = config.get("active_learning", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    budget = int(al_cfg.get("budget", 0))
    query_batch_size = int(al_cfg.get("query_batch_size", 4))
    uncertainty = str(al_cfg.get("uncertainty", "entropy"))
    silver_threshold = float(al_cfg.get("silver_threshold", 0.8))
    silver_weight = float(al_cfg.get("silver_weight", 0.5))
    if silver_weight > 1.0:
        raise ValueError("silver_weight must be <= gold weight (1.0).")

    max_length = int(model_cfg.get("max_length", 128))
    batch_size = int(train_cfg.get("batch_size", 8))
    model_name = str(model_cfg.get("name"))
    num_labels = int(model_cfg.get("num_labels", 2))

    labeler = None
    llm_mode = "none"
    if budget > 0:
        labeler = create_labeler(config)
        llm_mode = labeler.mode

    labeled = _ensure_weight_column(bundle.initial_labeled, default=1.0)
    labeled["label_source"] = "gold"
    labeled["weight"] = 1.0

    pool = bundle.pool_features.copy().reset_index(drop=True)

    human_oracle_used = 0
    llm_labels_accepted = 0
    llm_calls = 0
    round_logs: list[dict[str, Any]] = []

    def _fresh_model():
        model = build_model(model_name, num_labels=num_labels)
        return move_model_to_device(model, device)

    logger.info(
        "Proposed AL start | budget=%d | batch=%d | uncertainty=%s | "
        "silver_threshold=%.3f | silver_weight=%.3f | llm_mode=%s",
        budget,
        query_batch_size,
        uncertainty,
        silver_threshold,
        silver_weight,
        llm_mode,
    )

    model = _fresh_model()
    train_model(
        model,
        tokenizer,
        labeled,
        config=config,
        device=device,
        validation_df=bundle.validation,
    )

    round_idx = 0
    while human_oracle_used < budget and len(pool) > 0:
        round_idx += 1
        k = min(query_batch_size, len(pool))
        selected = select_uncertain_ids(
            pool,
            model,
            tokenizer,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            k=k,
            method=uncertainty,
        )
        if len(selected) == 0:
            break

        round_silver = 0
        round_gold = 0
        acquired_rows: list[dict[str, Any]] = []
        consumed_ids: list[str] = []

        for _, row in selected.iterrows():
            sample_id = str(row["sample_id"])
            text = str(row["text"])

            assert labeler is not None  # budget > 0 guarantees labeler
            llm_result = labeler.label_text(text)
            llm_calls += 1

            if llm_result.confidence >= silver_threshold:
                label = _LLM_TO_INT[llm_result.label]
                acquired_rows.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "label": label,
                        "weight": silver_weight,
                        "label_source": "silver",
                        "llm_confidence": llm_result.confidence,
                    }
                )
                llm_labels_accepted += 1
                round_silver += 1
                consumed_ids.append(sample_id)
                continue

            if human_oracle_used >= budget:
                # Leave remaining selected items in the pool.
                break

            gold = int(query_human_oracle(bundle, [sample_id]).iloc[0])
            human_oracle_used += 1
            acquired_rows.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "label": gold,
                    "weight": 1.0,
                    "label_source": "gold",
                    "llm_confidence": llm_result.confidence,
                }
            )
            round_gold += 1
            consumed_ids.append(sample_id)

        if acquired_rows:
            labeled = pd.concat(
                [labeled, pd.DataFrame(acquired_rows)],
                ignore_index=True,
            )
            pool = pool[~pool["sample_id"].astype(str).isin(consumed_ids)].reset_index(
                drop=True
            )

            model = _fresh_model()
            train_model(
                model,
                tokenizer,
                labeled,
                config=config,
                device=device,
                validation_df=bundle.validation,
            )

        round_logs.append(
            {
                "round": round_idx,
                "queried": len(consumed_ids),
                "silver": round_silver,
                "gold": round_gold,
                "pool_remaining": len(pool),
                "human_oracle_used": human_oracle_used,
                "llm_calls": llm_calls,
            }
        )
        logger.info("AL round %s", round_logs[-1])

        if not acquired_rows:
            break

    human_labels_used = int(len(bundle.initial_labeled) + human_oracle_used)
    gold_total = int((labeled["label_source"] == "gold").sum())
    silver_total = int((labeled["label_source"] == "silver").sum())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "labeled": labeled,
        "pool_remaining": pool,
        "human_labels_used": human_labels_used,
        "human_oracle_used": human_oracle_used,
        "gold_labels_total": gold_total,
        "llm_labels_accepted": llm_labels_accepted,
        "silver_labels_total": silver_total,
        "llm_calls": llm_calls,
        "llm_mode": llm_mode,
        "rounds": round_logs,
        "n_train_final": len(labeled),
    }


def run_baseline_active_learning(
    config: dict[str, Any],
    bundle: DataBundle,
    *,
    tokenizer,
    device: str,
) -> dict[str, Any]:
    """Random-query baseline using the same human budget as the proposed method.

    No LLM is used. Selected pool samples receive gold labels from the human
    oracle only.
    """
    logger = get_logger("active_learning")
    al_cfg = config.get("active_learning", {})
    model_cfg = config.get("model", {})

    budget = int(al_cfg.get("budget", 0))
    query_batch_size = int(al_cfg.get("query_batch_size", 4))
    model_name = str(model_cfg.get("name"))
    num_labels = int(model_cfg.get("num_labels", 2))
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    labeled = _ensure_weight_column(bundle.initial_labeled, default=1.0)
    labeled["label_source"] = "gold"
    labeled["weight"] = 1.0
    pool = bundle.pool_features.copy().reset_index(drop=True)

    human_oracle_used = 0
    round_logs: list[dict[str, Any]] = []

    def _fresh_model():
        model = build_model(model_name, num_labels=num_labels)
        return move_model_to_device(model, device)

    logger.info(
        "Baseline AL start | budget=%d | batch=%d | selection=random | llm_mode=none",
        budget,
        query_batch_size,
    )

    model = _fresh_model()
    train_model(
        model,
        tokenizer,
        labeled,
        config=config,
        device=device,
        validation_df=bundle.validation,
    )

    round_idx = 0
    while human_oracle_used < budget and len(pool) > 0:
        round_idx += 1
        k = min(query_batch_size, budget - human_oracle_used, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        selected = pool.iloc[chosen]
        acquired_rows: list[dict[str, Any]] = []
        consumed_ids: list[str] = []

        for _, row in selected.iterrows():
            sample_id = str(row["sample_id"])
            text = str(row["text"])
            gold = int(query_human_oracle(bundle, [sample_id]).iloc[0])
            human_oracle_used += 1
            acquired_rows.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "label": gold,
                    "weight": 1.0,
                    "label_source": "gold",
                    "llm_confidence": None,
                }
            )
            consumed_ids.append(sample_id)

        labeled = pd.concat([labeled, pd.DataFrame(acquired_rows)], ignore_index=True)
        pool = pool[~pool["sample_id"].astype(str).isin(consumed_ids)].reset_index(
            drop=True
        )

        model = _fresh_model()
        train_model(
            model,
            tokenizer,
            labeled,
            config=config,
            device=device,
            validation_df=bundle.validation,
        )

        round_logs.append(
            {
                "round": round_idx,
                "queried": len(consumed_ids),
                "silver": 0,
                "gold": len(consumed_ids),
                "pool_remaining": len(pool),
                "human_oracle_used": human_oracle_used,
                "llm_calls": 0,
            }
        )
        logger.info("Baseline AL round %s", round_logs[-1])

    human_labels_used = int(len(bundle.initial_labeled) + human_oracle_used)
    gold_total = int((labeled["label_source"] == "gold").sum())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "labeled": labeled,
        "pool_remaining": pool,
        "human_labels_used": human_labels_used,
        "human_oracle_used": human_oracle_used,
        "gold_labels_total": gold_total,
        "llm_labels_accepted": 0,
        "silver_labels_total": 0,
        "llm_calls": 0,
        "llm_mode": "none",
        "rounds": round_logs,
        "n_train_final": len(labeled),
    }
