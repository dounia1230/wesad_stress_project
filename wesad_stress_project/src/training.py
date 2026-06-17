"""Training helpers for PyTorch binary classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from .utils import save_json


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
) -> float:
    model.train()
    losses = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device).reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x).reshape(-1)
        loss = criterion(logits, batch_y)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device).reshape(-1)
            logits = model(batch_x).reshape(-1)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


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
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, torch.Tensor]]:
    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

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
    return model, pd.DataFrame(history), best_state or {}


def pos_weight_from_labels(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    y = y_train.reshape(-1)
    number_negative = (y == 0).sum().item()
    number_positive = (y == 1).sum().item()
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
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "best_model.pt")
    save_json(artifact_dir / "model_config.json", model_config)
    save_json(artifact_dir / "threshold.json", {"threshold": threshold})
    save_json(artifact_dir / "validation_metrics.json", validation_metrics)
    save_json(artifact_dir / "test_metrics.json", test_metrics)
    history.to_csv(artifact_dir / "training_history.csv", index=False)
    per_subject.to_csv(artifact_dir / "per_subject_metrics.csv", index=False)
    test_predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)

