"""Training helpers for PyTorch binary classifiers."""

from __future__ import annotations

import time
import math
import statistics
from collections.abc import Iterable
from typing import Any

import torch
import pandas as pd

from .helpers import save_json


def compute_global_gradient_norm(
    parameters: Iterable[torch.nn.Parameter],
    norm_type: float = 2.0,
) -> float:
    """Return the global norm of existing gradients as a detached Python float."""
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    if norm_type == float("inf"):
        return max(float(gradient.abs().max().cpu()) for gradient in gradients)
    if norm_type <= 0:
        raise ValueError("norm_type must be positive.")
    powered = sum(float(torch.linalg.vector_norm(g, ord=norm_type).cpu()) ** norm_type for g in gradients)
    return float(powered ** (1.0 / norm_type))


def unpack_supervised_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract inputs and labels from datasets that may also return metadata."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("A supervised batch must contain at least inputs and labels.")
    return batch[0], batch[1]


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    gradient_clip: float | None = None,
    record_gradient_norms: bool = False,
) -> float | tuple[float, dict[str, float | None]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    pre_norms: list[float] = []
    post_norms: list[float] = []
    for batch in loader:
        batch_x, batch_y = unpack_supervised_batch(batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device).reshape(-1)
        if next(model.parameters()).device != batch_x.device:
            raise RuntimeError("Model parameters and input batch are on different devices.")
        batch_size = batch_y.numel()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x).reshape(-1)
        loss = criterion(logits, batch_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became NaN or infinite.")
        loss.backward()
        pre_norm = compute_global_gradient_norm(model.parameters())
        if not math.isfinite(pre_norm):
            raise FloatingPointError("Gradient norm became NaN or infinite.")
        if record_gradient_norms:
            pre_norms.append(pre_norm)
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            if record_gradient_norms:
                post_norms.append(compute_global_gradient_norm(model.parameters()))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot train on an empty loader.")
    mean_loss = total_loss / total_samples
    if not record_gradient_norms:
        return mean_loss
    stats: dict[str, float | None] = {
        "gradient_pre_mean": float(statistics.fmean(pre_norms)),
        "gradient_pre_median": float(statistics.median(pre_norms)),
        "gradient_pre_max": float(max(pre_norms)),
        "gradient_post_mean": float(statistics.fmean(post_norms)) if post_norms else None,
        "gradient_post_max": float(max(post_norms)) if post_norms else None,
    }
    return mean_loss, stats


def evaluate_loss(model: torch.nn.Module, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y = unpack_supervised_batch(batch)
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
    record_gradient_norms: bool = False,
    record_validation_macro_f1: bool = False,
    verbose: bool = True,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    numerical_failure: str | None = None
    training_start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        try:
            epoch_result = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                gradient_clip=gradient_clip,
                record_gradient_norms=record_gradient_norms,
            )
        except FloatingPointError as error:
            numerical_failure = str(error)
            history.append({"epoch": epoch, "numerical_failure": numerical_failure})
            if verbose:
                print(f"Numerical failure at epoch {epoch}: {numerical_failure}")
            break
        if record_gradient_norms:
            train_loss, gradient_statistics = epoch_result
        else:
            train_loss = epoch_result
            gradient_statistics = {}
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **gradient_statistics,
            }
        if record_validation_macro_f1:
            from .evaluation import binary_metrics, collect_probabilities, select_threshold

            probabilities, labels = collect_probabilities(model, validation_loader, device)
            epoch_threshold, _ = select_threshold(labels, probabilities)
            row["validation_macro_f1"] = binary_metrics(labels, probabilities, epoch_threshold)["macro_f1"]
            row["validation_threshold"] = epoch_threshold
        history.append(row)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1
            improved = False

        if verbose:
            marker = " *" if improved else ""
            print(
                f"Epoch {epoch:03d}/{max_epochs:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"validation_loss={validation_loss:.4f} | "
                f"best_epoch={best_epoch}{marker}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"Early stopping after {epoch} epochs "
                    f"(no validation-loss improvement for {patience} epochs)."
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    training_summary = {
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_validation_loss": (
            float(best_validation_loss) if best_epoch is not None else None
        ),
        "epochs_trained": len(history),
        "early_stopped": epochs_without_improvement >= patience,
        "training_time_seconds": time.perf_counter() - training_start_time,
        "gradient_clip_max_norm": gradient_clip,
        "gradient_norms_recorded": record_gradient_norms,
        "numerically_stable": numerical_failure is None,
        "numerical_failure": numerical_failure,
    }
    return model, pd.DataFrame(history), training_summary


def pos_weight_from_labels(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    y = y_train.reshape(-1)
    number_negative = (y == 0).sum().item()
    number_positive = (y == 1).sum().item()
    if number_negative == 0:
        raise ValueError(
            "Cannot compute pos_weight because the training set has no negative labels."
        )
    if number_positive == 0:
        raise ValueError(
            "Cannot compute pos_weight because the training set has no positive labels."
        )
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
