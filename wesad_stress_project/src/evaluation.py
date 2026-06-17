"""Evaluation helpers shared by model notebooks and scripts."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.reshape(-1))


def collect_probabilities(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probabilities.append(logits_to_probabilities(logits).cpu().numpy())
            labels.append(batch_y.float().cpu().numpy().reshape(-1))
    return np.concatenate(probabilities), np.concatenate(labels)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
            }
        )
    table = pd.DataFrame(rows)
    best_row = table.sort_values(["macro_f1", "threshold"], ascending=[False, True]).iloc[0]
    return float(best_row["threshold"]), table


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "non_stress_precision": float(precision[0]),
        "non_stress_recall": float(recall[0]),
        "stress_precision": float(precision[1]),
        "stress_recall": float(recall[1]),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = _safe_score(roc_auc_score, y_true, probabilities)
    metrics["average_precision"] = _safe_score(average_precision_score, y_true, probabilities)
    return metrics


def per_subject_metrics(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    frame = metadata[["subject_id", "window_id"]].copy()
    frame["y_true"] = y_true.astype(int)
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)

    rows = []
    for subject_id, group in frame.groupby("subject_id"):
        rows.append(
            {
                "subject_id": subject_id,
                "n_windows": int(len(group)),
                **binary_metrics(
                    group["y_true"].to_numpy(),
                    group["probability"].to_numpy(),
                    threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def prediction_table(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": metadata["window_id"].to_numpy(),
            "subject_id": metadata["subject_id"].to_numpy(),
            "true_label": y_true.astype(int),
            "stress_probability": probabilities,
            "predicted_label": (probabilities >= threshold).astype(int),
        }
    )


def _safe_score(
    scorer: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    try:
        return float(scorer(y_true, probabilities))
    except ValueError:
        return None

