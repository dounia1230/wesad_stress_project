"""Training helpers for PyTorch binary classifiers."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import torch

from .helpers import save_json


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device).reshape(-1)
        batch_size = batch_y.numel()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x).reshape(-1)
        loss = criterion(logits, batch_y)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot train on an empty loader.")
    return total_loss / total_samples


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device).reshape(-1)
            batch_size = batch_y.numel()
            logits = model(batch_x).reshape(-1)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return total_loss / total_samples


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 8,
    gradient_clip: float | None = None,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    training_start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip=gradient_clip,
        )
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    training_summary = {
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_validation_loss": float(best_validation_loss),
        "epochs_trained": len(history),
        "early_stopped": epochs_without_improvement >= patience,
        "training_time_seconds": time.perf_counter() - training_start_time,
    }
    return model, pd.DataFrame(history), training_summary


def pos_weight_from_labels(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    y = y_train.reshape(-1)
    number_negative = (y == 0).sum().item()
    number_positive = (y == 1).sum().item()
    if number_negative == 0:
        raise ValueError("Cannot compute pos_weight because the training set has no negative labels.")
    if number_positive == 0:
        raise ValueError("Cannot compute pos_weight because the training set has no positive labels.")
    return torch.tensor([number_negative / number_positive], dtype=torch.float32, device=device)


def save_model_artifacts(
    artifact_dir,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    threshold: float,
    history: pd.DataFrame,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    per_subject: pd.DataFrame,
    test_predictions: pd.DataFrame,
    training_summary: dict[str, Any] | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "best_model.pt")
    save_json(artifact_dir / "model_config.json", model_config)
    save_json(
        artifact_dir / "threshold.json",
        {
            "threshold": threshold,
            "selected_on": "validation",
            "selection_metric": "macro_f1",
            "positive_class": "stress",
        },
    )
    save_json(artifact_dir / "training_summary.json", training_summary or {})
    save_json(artifact_dir / "validation_metrics.json", validation_metrics)
    save_json(artifact_dir / "test_metrics.json", test_metrics)
    history.to_csv(artifact_dir / "training_history.csv", index=False)
    per_subject.to_csv(artifact_dir / "per_subject_metrics.csv", index=False)
    test_predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)
