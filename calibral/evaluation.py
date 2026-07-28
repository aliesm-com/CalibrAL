"""Evaluation metrics and prediction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from calibral.utils import get_logger


class TextClassificationDataset(Dataset):
    """Simple dataset wrapping tokenized texts and optional labels/weights."""

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int,
        labels: list[int] | None = None,
        sample_ids: list[str] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)
        self.weights = (
            None if weights is None else torch.tensor(weights, dtype=torch.float)
        )
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.weights is not None:
            item["weights"] = self.weights[idx]
        return item


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, float]:
    """Compute Macro-F1 and Accuracy from integer labels."""
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'scikit-learn'. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
    }


@torch.no_grad()
def predict(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    device: str,
    max_length: int,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run inference and return predictions with probabilities."""
    logger = get_logger("evaluation")
    model.eval()

    texts = frame["text"].astype(str).tolist()
    sample_ids = frame["sample_id"].astype(str).tolist()
    has_labels = "label" in frame.columns
    labels = frame["label"].astype(int).tolist() if has_labels else None

    dataset = TextClassificationDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        labels=labels,
        sample_ids=sample_ids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_labels: list[int] = []
    prob_positive: list[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        pred_labels.extend(preds.cpu().tolist())
        # Class 1 = positive for IMDB
        prob_positive.extend(probs[:, 1].cpu().tolist())

    out = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pred_label": pred_labels,
            "prob_positive": prob_positive,
        }
    )
    if has_labels:
        out["true_label"] = labels
        metrics = compute_metrics(labels, pred_labels)
        logger.info(
            "Eval n=%d | macro_f1=%.4f | accuracy=%.4f",
            len(out),
            metrics["macro_f1"],
            metrics["accuracy"],
        )
    return out
