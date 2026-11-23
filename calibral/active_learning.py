"""Uncertainty-based active learning with LLM silver labels and human oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from calibral.data import LABEL_NEGATIVE, LABEL_POSITIVE, DataBundle
from calibral.evaluation import predict
from calibral.llm_labeler import LLMLabelResult, create_labeler, summarize_llm_usage
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


def _usage_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    cost_cfg = config.get("cost") or {}
    return {
        "price_per_million_input": cost_cfg.get("llm_price_per_million_input"),
        "price_per_million_output": cost_cfg.get("llm_price_per_million_output"),
    }


def _llm_record(
    bundle: DataBundle,
    sample_id: str,
    result: LLMLabelResult,
    *,
    phase: str,
    accepted: bool,
    uncertainty: float | None = None,
) -> dict[str, Any]:
    """Evaluation-only log of one LLM answer (gold is never used for training)."""
    gold = int(bundle.pool_labels.loc[sample_id])
    pred = None if result.source == "error" else _LLM_TO_INT[result.label]
    return {
        "sample_id": sample_id,
        "phase": phase,
        "uncertainty": uncertainty,
        "llm_label": pred,
        "llm_confidence": float(result.confidence),
        "llm_error": result.source == "error",
        "accepted_as_silver": bool(accepted),
        "gold_label": gold,
        "llm_correct": (pred == gold) if pred is not None else False,
        "provider": result.provider,
        "served_model": result.served_model,
    }


def _silver_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [r for r in records if r["accepted_as_silver"]]
    rejected = [r for r in records if not r["accepted_as_silver"] and not r["llm_error"]]
    def _acc(rows):
        return round(sum(1 for r in rows if r["llm_correct"]) / len(rows), 6) if rows else None
    return {
        # Precision of the labels that actually entered training as silver.
        "silver_label_accuracy": _acc(accepted),
        # LLM accuracy on answers routed to humans (below threshold): shows
        # whether the confidence gate filters out the LLM's mistakes.
        "rejected_llm_accuracy": _acc(rejected),
        "llm_labeled_accuracy": _acc([r for r in records if not r["llm_error"]]),
    }


def run_proposed_active_learning(
    config: dict[str, Any],
    bundle: DataBundle,
    *,
    tokenizer,
    device: str,
    selection: str = "uncertainty",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run the hybrid AL loop and return model + accounting stats.

    ``selection="uncertainty", use_llm=True`` is the proposed method.
    Ablations reuse the same loop so only one factor changes:

    - ``selection="random", use_llm=True``  -> random_silver (is uncertainty
      sampling needed, given the same LLM silver mechanism?)
    - ``selection="uncertainty", use_llm=False`` -> uncertainty_gold (classic
      AL with humans only; is the LLM needed?)
    """
    if selection not in {"uncertainty", "random"}:
        raise ValueError(f"Unknown selection={selection!r}")
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
    if budget > 0 and use_llm:
        labeler = create_labeler(config)
        llm_mode = labeler.mode
    rng = np.random.default_rng(int(config.get("seed", 42)))
    all_llm_results: list[LLMLabelResult] = []
    llm_records: list[dict[str, Any]] = []

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
    initial_train = train_model(
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
        train_out = None
        if use_llm:
            k = min(query_batch_size, len(pool))
        else:  # humans only: never select more than the remaining budget
            k = min(query_batch_size, budget - human_oracle_used, len(pool))
        if selection == "uncertainty":
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
        else:
            chosen = rng.choice(len(pool), size=k, replace=False)
            selected = pool.iloc[np.sort(chosen)].reset_index(drop=True)
            selected["uncertainty"] = np.nan
        if len(selected) == 0:
            break

        round_silver = 0
        round_gold = 0
        acquired_rows: list[dict[str, Any]] = []
        consumed_ids: list[str] = []

        # Prefetch all LLM labels for this round concurrently (order preserved).
        if labeler is not None:
            selected_texts = [str(t) for t in selected["text"].tolist()]
            llm_results: list[LLMLabelResult | None] = list(labeler.label_many(selected_texts))
            llm_calls += len(llm_results)
            all_llm_results.extend(r for r in llm_results if r is not None)
        else:
            llm_results = [None] * len(selected)

        for (_, row), llm_result in zip(selected.iterrows(), llm_results):
            sample_id = str(row["sample_id"])
            text = str(row["text"])
            unc = row.get("uncertainty")
            unc = None if unc is None or pd.isna(unc) else float(unc)

            if llm_result is not None and llm_result.confidence >= silver_threshold:
                llm_records.append(
                    _llm_record(bundle, sample_id, llm_result, phase="al_round",
                                accepted=True, uncertainty=unc)
                )
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

            if llm_result is not None:
                llm_records.append(
                    _llm_record(bundle, sample_id, llm_result, phase="al_round",
                                accepted=False, uncertainty=unc)
                )
            gold = int(query_human_oracle(bundle, [sample_id]).iloc[0])
            human_oracle_used += 1
            acquired_rows.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "label": gold,
                    "weight": 1.0,
                    "label_source": "gold",
                    "llm_confidence": llm_result.confidence if llm_result else None,
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
            train_out = train_model(
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
        if acquired_rows and train_out is not None:
            round_logs[-1]["n_train_after"] = train_out.get("n_train")
            round_logs[-1]["best_val_macro_f1"] = train_out.get("best_val_macro_f1")
            round_logs[-1]["epoch_history"] = train_out.get("history")
        logger.info("AL round %s", round_logs[-1])

        if not acquired_rows:
            break

    # ------------------------------------------------------------------
    # Post-budget silver expansion: once the human budget is exhausted,
    # let the LLM annotate a large random slice of the remaining pool.
    # Random samples are mostly "easy" (unlike the max-uncertainty picks),
    # so LLM confidence/acceptance is high and the silver labels are clean.
    # Humans are never queried here; only silver labels are added.
    # ------------------------------------------------------------------
    post_budget_silver = int(al_cfg.get("post_budget_silver", 0))
    if post_budget_silver > 0 and labeler is not None and len(pool) > 0:
        rng = np.random.default_rng(int(config.get("seed", 42)) + 1)
        k = min(post_budget_silver, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        selected = pool.iloc[np.sort(chosen)]
        logger.info(
            "Silver expansion start | requesting %d LLM labels from remaining pool (%d)",
            k,
            len(pool),
        )
        selected_texts = [str(t) for t in selected["text"].tolist()]
        llm_results = labeler.label_many(selected_texts)
        llm_calls += len(llm_results)
        all_llm_results.extend(llm_results)

        expansion_rows: list[dict[str, Any]] = []
        consumed_ids = []
        train_out = None
        for (_, row), llm_result in zip(selected.iterrows(), llm_results):
            llm_records.append(
                _llm_record(
                    bundle, str(row["sample_id"]), llm_result,
                    phase="silver_expansion",
                    accepted=llm_result.confidence >= silver_threshold,
                )
            )
            if llm_result.confidence >= silver_threshold:
                expansion_rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "text": str(row["text"]),
                        "label": _LLM_TO_INT[llm_result.label],
                        "weight": silver_weight,
                        "label_source": "silver",
                        "llm_confidence": llm_result.confidence,
                    }
                )
                llm_labels_accepted += 1
                consumed_ids.append(str(row["sample_id"]))

        if expansion_rows:
            labeled = pd.concat(
                [labeled, pd.DataFrame(expansion_rows)], ignore_index=True
            )
            pool = pool[
                ~pool["sample_id"].astype(str).isin(consumed_ids)
            ].reset_index(drop=True)
            model = _fresh_model()
            train_out = train_model(
                model,
                tokenizer,
                labeled,
                config=config,
                device=device,
                validation_df=bundle.validation,
            )

        round_idx += 1
        round_logs.append(
            {
                "round": round_idx,
                "queried": k,
                "silver": len(expansion_rows),
                "gold": 0,
                "pool_remaining": len(pool),
                "human_oracle_used": human_oracle_used,
                "llm_calls": llm_calls,
                "phase": "silver_expansion",
            }
        )
        if expansion_rows and train_out is not None:
            round_logs[-1]["n_train_after"] = train_out.get("n_train")
            round_logs[-1]["best_val_macro_f1"] = train_out.get("best_val_macro_f1")
            round_logs[-1]["epoch_history"] = train_out.get("history")
        logger.info("Silver expansion done %s", round_logs[-1])

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
        "initial_train": initial_train,
        "n_train_final": len(labeled),
        "llm_usage": summarize_llm_usage(all_llm_results, **_usage_kwargs(config)),
        "llm_records": llm_records,
        **_silver_stats(llm_records),
    }


def run_llm_distill_training(
    config: dict[str, Any],
    bundle: DataBundle,
    *,
    tokenizer,
    device: str,
) -> dict[str, Any]:
    """LLM-only distillation baseline: no human labels in the training set.

    The LLM labels ``distill.size`` random pool samples (default: the number
    of training samples the proposed method can reach, i.e.
    initial + budget + post_budget_silver). Labels below
    ``distill.threshold`` (default: active_learning.silver_threshold; 0 keeps
    all) are dropped. The classifier is trained once on the result.
    ``distill.include_initial_gold: true`` adds the initial human seed set.
    The gold validation split is still used for early stopping, exactly as in
    every other method.
    """
    logger = get_logger("active_learning")
    al_cfg = config.get("active_learning", {})
    distill_cfg = config.get("distill") or {}
    model_cfg = config.get("model", {})
    default_size = (
        len(bundle.initial_labeled)
        + int(al_cfg.get("budget", 0))
        + int(al_cfg.get("post_budget_silver", 0))
    )
    size = int(distill_cfg.get("size") or default_size)
    threshold = float(
        distill_cfg.get("threshold", al_cfg.get("silver_threshold", 0.8))
    )
    include_gold = bool(distill_cfg.get("include_initial_gold", False))

    labeler = create_labeler(config)
    pool = bundle.pool_features.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(config.get("seed", 42)) + 2)
    k = min(size, len(pool))
    selected = pool.iloc[np.sort(rng.choice(len(pool), size=k, replace=False))]
    logger.info(
        "LLM distill | labeling %d random pool samples | threshold=%.3f | "
        "include_initial_gold=%s",
        k,
        threshold,
        include_gold,
    )
    results = labeler.label_many([str(t) for t in selected["text"].tolist()])

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for (_, row), res in zip(selected.iterrows(), results):
        accepted = res.source != "error" and res.confidence >= threshold
        records.append(
            _llm_record(bundle, str(row["sample_id"]), res, phase="distill", accepted=accepted)
        )
        if accepted:
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "text": str(row["text"]),
                    "label": _LLM_TO_INT[res.label],
                    "weight": 1.0,
                    "label_source": "silver",
                    "llm_confidence": res.confidence,
                }
            )
    labeled = pd.DataFrame(rows)
    if include_gold:
        gold = _ensure_weight_column(bundle.initial_labeled, default=1.0)
        gold["label_source"] = "gold"
        gold["weight"] = 1.0
        labeled = pd.concat([gold, labeled], ignore_index=True)
    if labeled.empty:
        raise RuntimeError("LLM distillation produced no training labels.")

    model = move_model_to_device(
        build_model(str(model_cfg.get("name")), num_labels=int(model_cfg.get("num_labels", 2))),
        device,
    )
    train_out = train_model(
        model, tokenizer, labeled, config=config, device=device,
        validation_df=bundle.validation,
    )
    n_gold = int((labeled["label_source"] == "gold").sum())
    n_silver = int((labeled["label_source"] == "silver").sum())
    return {
        "model": model,
        "tokenizer": tokenizer,
        "labeled": labeled,
        "human_labels_used": n_gold,
        "human_oracle_used": 0,
        "gold_labels_total": n_gold,
        "llm_labels_accepted": n_silver,
        "silver_labels_total": n_silver,
        "llm_calls": len(results),
        "llm_mode": labeler.mode,
        "rounds": [
            {
                "round": 1,
                "phase": "distill",
                "queried": k,
                "silver": n_silver,
                "gold": n_gold,
                "n_train_after": train_out.get("n_train"),
                "best_val_macro_f1": train_out.get("best_val_macro_f1"),
            }
        ],
        "initial_train": None,
        "n_train_final": len(labeled),
        "llm_usage": summarize_llm_usage(list(results), **_usage_kwargs(config)),
        "llm_records": records,
        **_silver_stats(records),
    }


def run_llm_zeroshot_test(config: dict[str, Any], bundle: DataBundle) -> dict[str, Any]:
    """Label the test set directly with the LLM (no classifier is trained).

    Answers that still fail after retries count as wrong. Also reports
    selective accuracy/coverage at the silver threshold, i.e. how good the
    LLM is on the subset it is confident about.
    """
    al_cfg = config.get("active_learning", {})
    threshold = float(al_cfg.get("silver_threshold", 0.8))
    labeler = create_labeler(config)
    test = bundle.test.reset_index(drop=True)
    results = labeler.label_many([str(t) for t in test["text"].tolist()])

    true = test["label"].astype(int).tolist()
    preds: list[int] = []
    probs: list[float] = []
    confident_correct = 0
    confident_n = 0
    n_errors = 0
    for gold, res in zip(true, results):
        if res.source == "error":
            n_errors += 1
            preds.append(1 - gold)  # failure counts as an error
            probs.append(0.5)
            continue
        pred = _LLM_TO_INT[res.label]
        preds.append(pred)
        conf = float(res.confidence)
        probs.append(conf if pred == LABEL_POSITIVE else 1.0 - conf)
        if conf >= threshold:
            confident_n += 1
            confident_correct += int(pred == gold)
    predictions = pd.DataFrame(
        {
            "sample_id": test["sample_id"].astype(str),
            "true_label": true,
            "pred_label": preds,
            "prob_positive": probs,
        }
    )
    return {
        "model": None,
        "tokenizer": None,
        "predictions": predictions,
        "human_labels_used": 0,
        "human_oracle_used": 0,
        "gold_labels_total": 0,
        "llm_labels_accepted": 0,
        "silver_labels_total": 0,
        "llm_calls": len(results),
        "llm_mode": labeler.mode,
        "rounds": [],
        "initial_train": None,
        "n_train_final": 0,
        "llm_usage": summarize_llm_usage(list(results), **_usage_kwargs(config)),
        "llm_records": [],
        "llm_test_errors": n_errors,
        "llm_selective_coverage": round(confident_n / max(len(results), 1), 6),
        "llm_selective_accuracy": round(confident_correct / confident_n, 6)
        if confident_n
        else None,
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
