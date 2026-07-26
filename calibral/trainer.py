"""Training loop for text classification."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from calibral.evaluation import TextClassificationDataset, compute_metrics, predict
from calibral.utils import get_logger


def train_model(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    *,
    config: dict[str, Any],
    device: str,
    validation_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train ``model`` on ``train_df``.

    Optional ``weight`` column scales per-example loss (gold=1.0, silver<=1.0).
    Validation may be monitored for logging only. Test data must not be passed
    here. Early stopping is intentionally not used in this minimal prototype.
    """
    logger = get_logger("trainer")
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    epochs = int(train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 8))
    learning_rate = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    max_length = int(model_cfg.get("max_length", 128))

    if "label" not in train_df.columns:
        raise ValueError("train_df must include gold 'label' column.")

    weights = None
    if "weight" in train_df.columns:
        weights = train_df["weight"].astype(float).tolist()

    dataset = TextClassificationDataset(
        texts=train_df["text"].astype(str).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        labels=train_df["label"].astype(int).tolist(),
        sample_ids=train_df["sample_id"].astype(str).tolist(),
        weights=weights,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = max(1, epochs * len(loader))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    try:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    except ImportError:
        scheduler = None

    model.to(device)
    model.train()
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_score: float | None = None
    best_epoch: int | None = None

    logger.info(
        "Training start | device=%s | epochs=%d | batch_size=%d | n_train=%d | "
        "lr=%s | warmup_steps=%d/%d | max_grad_norm=%s",
        device,
        epochs,
        batch_size,
        len(train_df),
        learning_rate,
        warmup_steps,
        total_steps,
        max_grad_norm,
    )

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            sample_weights = batch.pop("weights", None)
            labels = batch["labels"]

            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits
            per_sample = F.cross_entropy(logits, labels, reduction="none")
            if sample_weights is not None:
                loss = (per_sample * sample_weights).mean()
            else:
                loss = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1
            progress.set_postfix(loss=float(loss.item()))

        avg_loss = running_loss / max(n_batches, 1)
        record: dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss}

        if validation_df is not None and len(validation_df) > 0:
            # Validation is used only to keep the best checkpoint (not the test set).
            val_preds = predict(
                model,
                tokenizer,
                validation_df,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
            )
            val_metrics = compute_metrics(
                val_preds["true_label"].tolist(),
                val_preds["pred_label"].tolist(),
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            val_score = float(val_metrics["macro_f1"])
            if best_val_score is None or val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                record["best_so_far"] = True
            model.train()

        history.append(record)
        logger.info("Epoch %d done | %s", epoch, record)

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            "Restored best validation checkpoint | epoch=%s | val_macro_f1=%.4f",
            best_epoch,
            best_val_score if best_val_score is not None else float("nan"),
        )

    return {
        "history": history,
        "n_train": len(train_df),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_score,
    }
