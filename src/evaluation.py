"""Evaluation helpers shared by model notebooks and scripts."""

from __future__ import annotations

import warnings
from typing import Callable

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("A supervised batch must contain inputs and labels.")
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probabilities.append(logits_to_probabilities(logits).cpu().numpy())
            labels.append(batch_y.float().cpu().numpy().reshape(-1))
    if not probabilities:
        raise ValueError("Cannot collect probabilities from an empty loader.")
    return np.concatenate(probabilities), np.concatenate(labels)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    y_true, probabilities = _validate_binary_arrays(y_true, probabilities)
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            predicted,
            labels=[0, 1],
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(y_true, predicted, average="weighted", zero_division=0)
                ),
                "stress_precision": float(precision[1]),
                "stress_recall": float(recall[1]),
                "distance_from_0_5": abs(float(round(threshold, 2)) - 0.5),
            }
        )
    table = pd.DataFrame(rows)
    best_row = table.sort_values(["macro_f1", "distance_from_0_5"], ascending=[False, True]).iloc[0]
    table = table.drop(columns=["distance_from_0_5"])
    return float(best_row["threshold"]), table


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y_true, probabilities = _validate_binary_arrays(y_true, probabilities)
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
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
    y_true, probabilities = _validate_binary_arrays(y_true, probabilities)
    if len(metadata) != len(y_true):
        raise ValueError("Metadata length must match the number of predictions.")
    if "subject_id" not in metadata or "window_id" not in metadata:
        raise ValueError("Metadata must include subject_id and window_id columns.")
    frame = metadata[["subject_id", "window_id"]].copy()
    frame["y_true"] = y_true.astype(int)
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)

    rows = []
    expected_subjects = set(metadata["subject_id"].unique())
    observed_subjects = set()
    for subject_id, group in frame.groupby("subject_id", sort=True):
        observed_subjects.add(subject_id)
        rows.append(
            {
                "subject": subject_id,
                "n_samples": int(len(group)),
                **binary_metrics(
                    group["y_true"].to_numpy(),
                    group["probability"].to_numpy(),
                    threshold,
                ),
            }
        )
    if observed_subjects != expected_subjects:
        missing = sorted(expected_subjects - observed_subjects)
        raise ValueError(f"Missing per-subject metrics for subjects: {missing}")
    result = pd.DataFrame(rows)
    validate_per_subject_columns(result)
    return result


def validate_per_subject_columns(per_subject_df: pd.DataFrame) -> None:
    """Validate the canonical schema shared by all final-model artifacts."""
    required_columns = {
        "subject",
        "n_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "stress_precision",
        "stress_recall",
        "roc_auc",
        "average_precision",
    }
    missing = required_columns.difference(per_subject_df.columns)
    if missing:
        raise ValueError(f"Missing per-subject columns: {sorted(missing)}")


def prediction_table(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true, probabilities = _validate_binary_arrays(y_true, probabilities)
    if len(metadata) != len(y_true):
        raise ValueError("Metadata length must match the number of predictions.")
    return pd.DataFrame(
        {
            "window_id": metadata["window_id"].to_numpy(),
            "subject_id": metadata["subject_id"].to_numpy(),
            "true_label": y_true.astype(int),
            "stress_probability": probabilities,
            "predicted_label": (probabilities >= threshold).astype(int),
            "threshold": threshold,
        }
    )


def _safe_score(
    scorer: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if scorer is roc_auc_score and np.unique(y_true).size < 2:
        warnings.warn(
            "roc_auc is unavailable because this subset contains only one target class; returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    try:
        return float(scorer(y_true, probabilities))
    except ValueError as error:
        warnings.warn(
            f"{getattr(scorer, '__name__', 'metric')} is unavailable for this subset: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")


def _validate_binary_arrays(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(y_true) == 0:
        raise ValueError("y_true and probabilities must not be empty.")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    return y_true, probabilities
