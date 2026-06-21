"""Leakage-safe CWT scalogram generation and dataset access for WESAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pywt
from torch.utils.data import Dataset

from .config import (
    RANDOM_SEED,
    SCALOGRAM_ARTIFACT_DIR,
    SCALOGRAM_CHANNELS,
    SCALOGRAM_DIR,
    SCALOGRAM_SCALES,
    SCALOGRAM_SIZE,
    SCALOGRAM_WAVELET,
    SEQUENCE_CHANNELS,
    SPLIT_SUBJECTS,
    TARGET_HZ,
    WINDOW_SAMPLES,
)
from .helpers import save_json


@dataclass(frozen=True)
class ScalogramStatistics:
    """Per-channel moments fitted exclusively on training-subject scalograms."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def validate_subject_splits(splits: dict[str, Sequence[str]] = SPLIT_SUBJECTS) -> None:
    """Raise when any participant occurs in more than one split."""
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"Expected split keys {sorted(required)}, received {sorted(splits)}.")
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = set(splits[left]).intersection(splits[right])
        if overlap:
            raise ValueError(f"Subject overlap between {left} and {right}: {sorted(overlap)}")


def window_to_log_scalogram(
    window: np.ndarray | torch.Tensor,
    scales: Sequence[int] = SCALOGRAM_SCALES,
    wavelet: str = SCALOGRAM_WAVELET,
    sampling_rate: int = TARGET_HZ,
    output_size: tuple[int, int] = SCALOGRAM_SIZE,
) -> torch.Tensor:
    """Convert one standardized ``(960, 6)`` window to ``(3, 64, 64)``.

    The channels are BVP, EDA, and acceleration magnitude. CWT magnitudes are
    transformed with log1p and resized bilinearly. Dataset-level normalization is
    deliberately performed later using training-only statistics.
    """
    array = np.asarray(window, dtype=np.float32)
    expected = (WINDOW_SAMPLES, len(SEQUENCE_CHANNELS))
    if array.shape != expected:
        raise ValueError(f"Expected a standardized window shaped {expected}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("The time-domain window contains NaN or infinite values.")
    acc_magnitude = np.sqrt(np.square(array[:, 3]) + np.square(array[:, 4]) + np.square(array[:, 5]))
    signals = (array[:, 0], array[:, 1], acc_magnitude)
    maps = []
    for signal in signals:
        coefficients, _ = pywt.cwt(
            signal,
            np.asarray(scales),
            wavelet,
            sampling_period=1.0 / sampling_rate,
        )
        maps.append(np.log1p(np.abs(coefficients)).astype(np.float32))
    tensor = torch.from_numpy(np.stack(maps)).unsqueeze(0)
    resized = F.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False).squeeze(0)
    if tuple(resized.shape) != (3, *output_size) or not torch.isfinite(resized).all():
        raise RuntimeError("Scalogram conversion produced an invalid tensor.")
    return resized


def generate_log_scalograms(windows: torch.Tensor) -> torch.Tensor:
    """Generate unnormalized log-scalograms from standardized windows."""
    if windows.ndim != 3 or tuple(windows.shape[1:]) != (WINDOW_SAMPLES, 6):
        raise ValueError(f"Expected (N, {WINDOW_SAMPLES}, 6), got {tuple(windows.shape)}.")
    return torch.stack([window_to_log_scalogram(window) for window in windows], dim=0)


def fit_scalogram_statistics(train_scalograms: torch.Tensor) -> ScalogramStatistics:
    """Fit one population mean and standard deviation per channel on training only."""
    _validate_scalogram_tensor(train_scalograms)
    values = train_scalograms.to(torch.float64)
    mean = values.mean(dim=(0, 2, 3))
    std = values.std(dim=(0, 2, 3), correction=0)
    if torch.any(std <= 1e-12):
        raise ValueError("At least one training scalogram channel has near-zero variance.")
    return ScalogramStatistics(tuple(mean.tolist()), tuple(std.tolist()))


def normalize_scalograms(
    scalograms: torch.Tensor,
    statistics: ScalogramStatistics,
) -> torch.Tensor:
    """Apply frozen training-channel moments to an ``(N, 3, 64, 64)`` tensor."""
    _validate_scalogram_tensor(scalograms)
    mean = torch.tensor(statistics.mean, dtype=scalograms.dtype).view(1, 3, 1, 1)
    std = torch.tensor(statistics.std, dtype=scalograms.dtype).view(1, 3, 1, 1)
    normalized = (scalograms - mean) / std
    if not torch.isfinite(normalized).all():
        raise RuntimeError("Scalogram normalization produced NaN or infinite values.")
    return normalized


def build_scalogram_artifacts(
    project_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate, normalize, validate, and save all splits from processed windows.

    This function expects notebook 01 outputs. Statistics are fitted once on the
    training tensor and then frozen for validation and test.
    """
    validate_subject_splits()
    sequence_dir = project_root / "data" / "processed" / "sequence"
    metadata_dir = project_root / "data" / "processed" / "metadata"
    output_dir = project_root / SCALOGRAM_DIR.relative_to(SCALOGRAM_DIR.parents[2])
    artifact_dir = project_root / SCALOGRAM_ARTIFACT_DIR.relative_to(SCALOGRAM_ARTIFACT_DIR.parents[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / f"X_{split}.pt" for split in SPLIT_SUBJECTS]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Scalogram output already exists; pass overwrite=True explicitly.")

    raw_maps: dict[str, torch.Tensor] = {}
    labels: dict[str, torch.Tensor] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split in SPLIT_SUBJECTS:
        windows = torch.load(sequence_dir / f"X_{split}.pt", map_location="cpu", weights_only=True)
        labels[split] = torch.load(sequence_dir / f"y_{split}.pt", map_location="cpu", weights_only=True)
        metadata[split] = pd.read_csv(metadata_dir / f"windows_{split}.csv")
        raw_maps[split] = generate_log_scalograms(windows)
        _validate_alignment(raw_maps[split], labels[split], metadata[split], split)

    statistics = fit_scalogram_statistics(raw_maps["train"])
    for split in SPLIT_SUBJECTS:
        normalized = normalize_scalograms(raw_maps[split], statistics)
        torch.save(normalized, output_dir / f"X_{split}.pt")
        torch.save(labels[split].float(), output_dir / f"y_{split}.pt")

    metadata_payload = {
        "subject_lists": SPLIT_SUBJECTS,
        "scalogram_shape": [3, *SCALOGRAM_SIZE],
        "wavelet": SCALOGRAM_WAVELET,
        "scales": list(SCALOGRAM_SCALES),
        "sampling_rate_hz": TARGET_HZ,
        "sampling_period_seconds": 1.0 / TARGET_HZ,
        "resize_dimensions": list(SCALOGRAM_SIZE),
        "channel_order": SCALOGRAM_CHANNELS,
        "normalization_means": list(statistics.mean),
        "normalization_standard_deviations": list(statistics.std),
        "normalization_fitted_on": "train",
        "time_domain_normalization_fitted_on": "train",
        "random_seed": RANDOM_SEED,
    }
    save_json(artifact_dir / "scalogram_metadata.json", metadata_payload)
    return metadata_payload


class WESADScalogramDataset(Dataset):
    """Dataset returning ``(scalogram, scalar_label, metadata_dict)``."""

    def __init__(
        self,
        scalograms: torch.Tensor | Path,
        labels: torch.Tensor | Path,
        metadata: pd.DataFrame | Path,
    ) -> None:
        self.scalograms = _load_tensor(scalograms).float()
        self.labels = _load_tensor(labels).float().reshape(-1)
        self.metadata = pd.read_csv(metadata) if isinstance(metadata, Path) else metadata.reset_index(drop=True).copy()
        _validate_alignment(self.scalograms, self.labels, self.metadata, "dataset")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.metadata.iloc[index]
        item_metadata = {str(key): _python_scalar(value) for key, value in row.items()}
        return self.scalograms[index], self.labels[index], item_metadata


def _load_tensor(value: torch.Tensor | Path) -> torch.Tensor:
    return torch.load(value, map_location="cpu", weights_only=True) if isinstance(value, Path) else value


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_scalogram_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, *SCALOGRAM_SIZE):
        raise ValueError(f"Expected (N, 3, 64, 64), received {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Scalograms contain NaN or infinite values.")


def _validate_alignment(
    scalograms: torch.Tensor,
    labels: torch.Tensor,
    metadata: pd.DataFrame,
    split: str,
) -> None:
    _validate_scalogram_tensor(scalograms)
    if not (len(scalograms) == len(labels) == len(metadata)):
        raise ValueError(f"Tensor, label, and metadata lengths differ for {split}.")
    unique_labels = set(labels.detach().cpu().numpy().astype(int).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels in {split}: {sorted(unique_labels)}")
    if "subject_id" not in metadata:
        raise ValueError("Metadata must include subject_id.")
    if split in SPLIT_SUBJECTS:
        unexpected = set(metadata["subject_id"]) - set(SPLIT_SUBJECTS[split])
        if unexpected:
            raise ValueError(f"Unexpected subjects in {split}: {sorted(unexpected)}")
