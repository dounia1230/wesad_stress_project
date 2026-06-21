"""Reusable, validation-selected binary-classification experiment workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE, RANDOM_SEED, WEIGHT_DECAY
from .evaluation import binary_metrics, collect_probabilities, per_subject_metrics, prediction_table, select_threshold
from .helpers import count_parameters, save_json, set_seed
from .training import pos_weight_from_labels, save_model_artifacts, train_with_early_stopping


def make_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible loaders; only the training order is shuffled."""
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    """Extract labels without assuming TensorDataset or a concrete dataset class."""
    if hasattr(dataset, "labels"):
        return torch.as_tensor(dataset.labels).float().reshape(-1)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return torch.as_tensor(dataset.tensors[1]).float().reshape(-1)
    return torch.stack([torch.as_tensor(dataset[index][1]) for index in range(len(dataset))]).float()


def run_validation_selected_experiment(
    model_factory: Callable[[], torch.nn.Module],
    datasets: tuple[Dataset, Dataset, Dataset],
    validation_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    artifact_dir: Path,
    experiment_config: dict[str, Any],
    device: torch.device,
    compare_weighted_loss: bool = True,
    gradient_clip: float | None = None,
    record_gradient_norms: bool = False,
    record_validation_macro_f1: bool = False,
) -> dict[str, Any]:
    """Train variants, select by validation macro F1, then evaluate test once.

    Loss weighting and the classification threshold are selected from validation
    results. Test probabilities are not computed until those choices are frozen.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, validation_dataset, test_dataset = datasets
    y_train = dataset_labels(train_dataset)
    methods = ["unweighted", "weighted"] if compare_weighted_loss else ["unweighted"]
    variants: list[dict[str, Any]] = []
    for method in methods:
        set_seed(RANDOM_SEED)
        train_loader, validation_loader, _ = make_loaders(*datasets)
        model = model_factory().to(device)
        weight = pos_weight_from_labels(y_train, device) if method == "weighted" else None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model, history, summary = train_with_early_stopping(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            gradient_clip=gradient_clip,
            record_gradient_norms=record_gradient_norms,
            record_validation_macro_f1=record_validation_macro_f1,
        )
        if not summary["numerically_stable"]:
            variant_dir = artifact_dir / "variants" / method
            variant_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(variant_dir / "training_history.csv", index=False)
            save_json(variant_dir / "training_summary.json", summary)
            raise FloatingPointError(summary["numerical_failure"])
        probabilities, labels = collect_probabilities(model, validation_loader, device)
        threshold, threshold_table = select_threshold(labels, probabilities)
        metrics = binary_metrics(labels, probabilities, threshold)
        variant_dir = artifact_dir / "variants" / method
        variant_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), variant_dir / "best_model.pt")
        history.to_csv(variant_dir / "training_history.csv", index=False)
        threshold_table.to_csv(variant_dir / "validation_threshold_search.csv", index=False)
        save_json(variant_dir / "validation_metrics.json", metrics)
        save_json(variant_dir / "training_summary.json", summary)
        variants.append({
            "method": method,
            "model": model,
            "history": history,
            "summary": summary,
            "threshold": threshold,
            "validation_metrics": metrics,
            "validation_probabilities": probabilities,
            "validation_labels": labels,
        })

    selected = max(
        variants,
        key=lambda item: (item["validation_metrics"]["macro_f1"], -abs(item["threshold"] - 0.5)),
    )
    model = selected["model"]
    _, _, test_loader = make_loaders(*datasets)
    inference_start = time.perf_counter()
    test_probabilities, test_labels = collect_probabilities(model, test_loader, device)
    inference_seconds = time.perf_counter() - inference_start
    test_metrics = binary_metrics(test_labels, test_probabilities, selected["threshold"])
    test_metrics["inference_time_seconds"] = inference_seconds
    subject_table = per_subject_metrics(test_metadata, test_labels, test_probabilities, selected["threshold"])
    predictions = prediction_table(test_metadata, test_labels, test_probabilities, selected["threshold"])
    validation_metrics = dict(selected["validation_metrics"])
    validation_metrics["variant_comparison"] = [
        {
            "method": item["method"],
            "threshold": item["threshold"],
            "best_epoch": item["summary"]["best_epoch"],
            **item["validation_metrics"],
        }
        for item in variants
    ]
    full_config = {
        **experiment_config,
        "seed": RANDOM_SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "gradient_clip": gradient_clip,
        "loss_weighting": selected["method"],
        "classification_threshold": selected["threshold"],
        "parameter_count": count_parameters(model),
        "artifact_paths": {
            "directory": str(artifact_dir),
            "state_dict": str(artifact_dir / "best_model.pt"),
            "training_history": str(artifact_dir / "training_history.csv"),
            "test_predictions": str(artifact_dir / "test_predictions.csv"),
        },
    }
    selected["summary"]["inference_time_seconds"] = inference_seconds
    save_model_artifacts(
        artifact_dir,
        model,
        full_config,
        selected["threshold"],
        selected["history"],
        validation_metrics,
        test_metrics,
        subject_table,
        predictions,
        selected["summary"],
    )
    save_json(artifact_dir / "experiment_config.json", full_config)
    prediction_table(
        validation_metadata,
        selected["validation_labels"],
        selected["validation_probabilities"],
        selected["threshold"],
    ).to_csv(artifact_dir / "validation_predictions.csv", index=False)
    return {
        "model": model,
        "history": selected["history"],
        "training_summary": selected["summary"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "per_subject_metrics": subject_table,
        "threshold": selected["threshold"],
        "selected_loss": selected["method"],
        "config": full_config,
    }
