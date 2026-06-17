"""Preprocessing and feature extraction for the WESAD stress project."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

from .config import (
    ACCEPTED_LABELS,
    BINARY_LABEL_MAP,
    LABEL_HZ,
    SEQUENCE_CHANNELS,
    SPLIT_SUBJECTS,
    STRIDE_SAMPLES,
    TARGET_HZ,
    TRAIN_SUBJECTS,
    VALIDATION_SUBJECTS,
    TEST_SUBJECTS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    STRIDE_SECONDS,
    WRIST_SAMPLE_RATES,
)


@dataclass(frozen=True)
class ProcessedSplit:
    X_sequence: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def load_subject_pickle(data_root: Path, subject_id: str) -> dict:
    path = data_root / subject_id / f"{subject_id}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def align_subject_to_32hz(subject_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = subject_data["signal"]["wrist"]
    label = np.asarray(subject_data["label"]).reshape(-1)

    durations = [
        wrist["BVP"].shape[0] / WRIST_SAMPLE_RATES["BVP"],
        wrist["EDA"].shape[0] / WRIST_SAMPLE_RATES["EDA"],
        wrist["TEMP"].shape[0] / WRIST_SAMPLE_RATES["TEMP"],
        wrist["ACC"].shape[0] / WRIST_SAMPLE_RATES["ACC"],
        label.shape[0] / LABEL_HZ,
    ]
    duration = min(durations)
    n_samples = int(np.floor(duration * TARGET_HZ))
    target_times = np.arange(n_samples, dtype=np.float64) / float(TARGET_HZ)

    bvp = _downsample_bvp_to_32hz(wrist["BVP"], n_samples)
    eda = _interp_to_target(wrist["EDA"], WRIST_SAMPLE_RATES["EDA"], target_times)
    temp = _interp_to_target(wrist["TEMP"], WRIST_SAMPLE_RATES["TEMP"], target_times)
    acc = _align_acc_to_target(wrist["ACC"], n_samples)
    labels = label[np.minimum((target_times * LABEL_HZ).astype(int), len(label) - 1)]

    signals = np.column_stack(
        [
            bvp.reshape(-1),
            eda.reshape(-1),
            temp.reshape(-1),
            acc[:, 0],
            acc[:, 1],
            acc[:, 2],
        ]
    ).astype(np.float32)
    return signals, labels.astype(int), target_times


def create_subject_windows(
    subject_id: str,
    split: str,
    subject_data: dict,
    start_window_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    signals, labels, _ = align_subject_to_32hz(subject_data)
    windows = []
    y = []
    rows = []
    window_id = start_window_id

    for segment_start, segment_end, label_value in _accepted_label_segments(labels):
        for start in range(segment_start, segment_end - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            binary_label = BINARY_LABEL_MAP[int(label_value)]
            windows.append(signals[start:end])
            y.append(binary_label)
            rows.append(
                {
                    "window_id": window_id,
                    "subject_id": subject_id,
                    "split": split,
                    "label": int(label_value),
                    "binary_label": int(binary_label),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_time_seconds": float(start / TARGET_HZ),
                    "end_time_seconds": float(end / TARGET_HZ),
                }
            )
            window_id += 1

    if not windows:
        return (
            np.empty((0, WINDOW_SAMPLES, len(SEQUENCE_CHANNELS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(rows),
        )
    return np.stack(windows).astype(np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(rows)


def run_preprocessing(project_root: Path) -> dict[str, object]:
    assert set(TRAIN_SUBJECTS).isdisjoint(VALIDATION_SUBJECTS)
    assert set(TRAIN_SUBJECTS).isdisjoint(TEST_SUBJECTS)
    assert set(VALIDATION_SUBJECTS).isdisjoint(TEST_SUBJECTS)

    data_root = project_root / "data" / "WESAD" / "WESAD"
    sequence_dir = project_root / "data" / "processed" / "sequence"
    metadata_dir = project_root / "data" / "processed" / "metadata"
    artifact_dir = project_root / "artifacts" / "preprocessing"
    for directory in [sequence_dir, metadata_dir, artifact_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_sequences: dict[str, list[np.ndarray]] = {split: [] for split in SPLIT_SUBJECTS}
    labels: dict[str, list[np.ndarray]] = {split: [] for split in SPLIT_SUBJECTS}
    metadata_frames: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLIT_SUBJECTS}
    next_window_id = 0

    for split, subjects in SPLIT_SUBJECTS.items():
        for subject_id in subjects:
            subject_data = load_subject_pickle(data_root, subject_id)
            X_subject, y_subject, meta_subject = create_subject_windows(
                subject_id,
                split,
                subject_data,
                next_window_id,
            )
            if len(meta_subject):
                next_window_id = int(meta_subject["window_id"].max()) + 1
            raw_sequences[split].append(X_subject)
            labels[split].append(y_subject)
            metadata_frames[split].append(meta_subject)

    X_raw = {
        split: np.concatenate(parts, axis=0).astype(np.float32)
        for split, parts in raw_sequences.items()
    }
    y = {
        split: np.concatenate(parts, axis=0).astype(np.float32)
        for split, parts in labels.items()
    }
    metadata = {
        split: pd.concat(parts, ignore_index=True)
        for split, parts in metadata_frames.items()
    }
    X_sequence, sequence_scaler = fit_transform_sequences(X_raw["train"], X_raw)

    assert X_sequence["train"].shape[1:] == (WINDOW_SAMPLES, len(SEQUENCE_CHANNELS))
    assert X_sequence["validation"].shape[1:] == (WINDOW_SAMPLES, len(SEQUENCE_CHANNELS))
    assert X_sequence["test"].shape[1:] == (WINDOW_SAMPLES, len(SEQUENCE_CHANNELS))
    assert np.isfinite(X_sequence["train"]).all()
    assert np.isfinite(X_sequence["validation"]).all()
    assert np.isfinite(X_sequence["test"]).all()

    for split in SPLIT_SUBJECTS:
        torch.save(torch.from_numpy(X_raw[split]), sequence_dir / f"X_{split}_raw.pt")
        torch.save(torch.from_numpy(X_sequence[split]), sequence_dir / f"X_{split}.pt")
        torch.save(torch.from_numpy(y[split]), sequence_dir / f"y_{split}.pt")
        metadata[split].to_csv(metadata_dir / f"windows_{split}.csv", index=False)

    all_metadata = pd.concat(metadata.values(), ignore_index=True)
    all_metadata.to_csv(metadata_dir / "windows_all.csv", index=False)
    all_metadata.to_csv(metadata_dir / "window_metadata.csv", index=False)

    with (artifact_dir / "sequence_channels.json").open("w", encoding="utf-8") as handle:
        json.dump(SEQUENCE_CHANNELS, handle, indent=2)
    with (artifact_dir / "split_subjects.json").open("w", encoding="utf-8") as handle:
        json.dump(SPLIT_SUBJECTS, handle, indent=2)
    with (artifact_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(SPLIT_SUBJECTS, handle, indent=2)
    with (artifact_dir / "preprocessing_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "target_hz": TARGET_HZ,
                "window_seconds": WINDOW_SECONDS,
                "window_samples": WINDOW_SAMPLES,
                "stride_seconds": STRIDE_SECONDS,
                "stride_samples": STRIDE_SAMPLES,
                "sequence_channels": SEQUENCE_CHANNELS,
                "accepted_labels": sorted(ACCEPTED_LABELS),
                "binary_label_map": BINARY_LABEL_MAP,
            },
            handle,
            indent=2,
        )
    with (artifact_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "0": "non_stress",
                "1": "stress",
                "source_labels": BINARY_LABEL_MAP,
            },
            handle,
            indent=2,
        )
    class_distribution = pd.concat(
        [
            pd.Series(y[split]).value_counts().sort_index().rename(split)
            for split in SPLIT_SUBJECTS
        ],
        axis=1,
    ).fillna(0).astype(int)
    class_distribution.index.name = "binary_label"
    class_distribution.to_csv(artifact_dir / "class_distribution.csv")
    joblib.dump(sequence_scaler, artifact_dir / "sequence_scaler.joblib")

    return {
        "sequence_shapes": {split: tuple(X_sequence[split].shape) for split in SPLIT_SUBJECTS},
        "label_counts": {
            split: pd.Series(y[split]).value_counts().sort_index().to_dict()
            for split in SPLIT_SUBJECTS
        },
        "metadata_rows": {split: int(len(metadata[split])) for split in SPLIT_SUBJECTS},
    }


def fit_transform_sequences(
    X_train: np.ndarray,
    splits: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    transformed = {}
    for split, X in splits.items():
        transformed[split] = (
            scaler.transform(X.reshape(-1, X.shape[-1]))
            .reshape(X.shape)
            .astype(np.float32)
        )
    return transformed, scaler


def extract_feature_table(X: np.ndarray) -> pd.DataFrame:
    if len(X) == 0:
        return pd.DataFrame()
    return pd.DataFrame([extract_window_features(window) for window in X])


def extract_window_features(window: np.ndarray) -> dict[str, float]:
    bvp = window[:, 0]
    eda = window[:, 1]
    temp = window[:, 2]
    acc_x = window[:, 3]
    acc_y = window[:, 4]
    acc_z = window[:, 5]
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    features: dict[str, float] = {}
    for values, prefix in [
        (bvp, "BVP"),
        (eda, "EDA"),
        (temp, "TEMP"),
        (acc_x, "ACC_x"),
        (acc_y, "ACC_y"),
        (acc_z, "ACC_z"),
        (acc_magnitude, "ACC_magnitude"),
    ]:
        features.update(_general_features(values, prefix))
    features.update(_bvp_features(bvp))
    features.update(_eda_features(eda))
    features.update(_temperature_features(temp))
    features.update(_acc_features(acc_x, acc_y, acc_z, acc_magnitude))
    return features


def _interp_to_target(values: np.ndarray, source_hz: int, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    source_times = np.arange(values.shape[0], dtype=np.float64) / float(source_hz)
    channels = [
        np.interp(target_times, source_times, values[:, channel]).astype(np.float32)
        for channel in range(values.shape[1])
    ]
    return np.column_stack(channels)


def _downsample_bvp_to_32hz(values: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    downsampled = resample_poly(values, up=1, down=2).astype(np.float32)
    if downsampled.shape[0] < n_samples:
        source_times = np.arange(downsampled.shape[0], dtype=np.float64) / float(TARGET_HZ)
        target_times = np.arange(n_samples, dtype=np.float64) / float(TARGET_HZ)
        downsampled = np.interp(target_times, source_times, downsampled).astype(np.float32)
    return downsampled[:n_samples, None]


def _align_acc_to_target(values: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] >= n_samples:
        return values[:n_samples].astype(np.float32)
    source_times = np.arange(values.shape[0], dtype=np.float64) / float(WRIST_SAMPLE_RATES["ACC"])
    target_times = np.arange(n_samples, dtype=np.float64) / float(TARGET_HZ)
    channels = [
        np.interp(target_times, source_times, values[:, channel]).astype(np.float32)
        for channel in range(values.shape[1])
    ]
    return np.column_stack(channels)


def _accepted_label_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    segments = []
    start = None
    current_label = None

    for index, label in enumerate(labels):
        label = int(label)
        if label in ACCEPTED_LABELS:
            if start is None:
                start = index
                current_label = label
            elif label != current_label:
                segments.append((start, index, current_label))
                start = index
                current_label = label
        elif start is not None:
            segments.append((start, index, current_label))
            start = None
            current_label = None

    if start is not None:
        segments.append((start, len(labels), current_label))
    return segments


def _general_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64) / TARGET_HZ
    slope = np.polyfit(t, x, 1)[0] if x.size > 1 else 0.0
    q25, q75 = np.percentile(x, [25, 75])
    signal_std = float(np.std(x))
    if signal_std < 1e-8:
        skewness_value = 0.0
        kurtosis_value = 0.0
    else:
        skewness_value = float(skew(x, bias=False, nan_policy="omit"))
        kurtosis_value = float(kurtosis(x, bias=False, nan_policy="omit"))
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": signal_std,
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(x)))),
        f"{prefix}_energy": float(np.sum(np.square(x))),
        f"{prefix}_slope": float(slope),
        f"{prefix}_skewness": skewness_value,
        f"{prefix}_kurtosis": kurtosis_value,
    }


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _bvp_features(bvp: np.ndarray) -> dict[str, float]:
    distance = max(1, int(0.35 * TARGET_HZ))
    prominence = max(float(np.std(bvp)) * 0.25, 1e-6)
    peaks, _ = find_peaks(bvp, distance=distance, prominence=prominence)
    peak_times = peaks / TARGET_HZ
    ibi = np.diff(peak_times)
    valid_ibi = ibi[(ibi >= 0.3) & (ibi <= 2.0)]
    hr = 60.0 / valid_ibi if len(valid_ibi) else np.array([])
    rmssd = np.sqrt(np.mean(np.square(np.diff(valid_ibi)))) if len(valid_ibi) > 1 else 0.0
    return {
        "BVP_peak_count": float(len(peaks)),
        "BVP_mean_peak_amplitude": _safe_mean(bvp[peaks]) if len(peaks) else 0.0,
        "BVP_mean_heart_rate": _safe_mean(hr),
        "BVP_std_heart_rate": float(np.std(hr)) if len(hr) else 0.0,
        "BVP_rmssd": float(rmssd),
    }


def _eda_features(eda: np.ndarray) -> dict[str, float]:
    diff = np.diff(eda)
    peaks, _ = find_peaks(
        eda,
        distance=max(1, TARGET_HZ),
        prominence=max(float(np.std(eda)) * 0.25, 1e-6),
    )
    return {
        "EDA_peak_count": float(len(peaks)),
        "EDA_mean_abs_derivative": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        "EDA_max_derivative": float(np.max(diff)) if len(diff) else 0.0,
        "EDA_min_derivative": float(np.min(diff)) if len(diff) else 0.0,
    }


def _temperature_features(temp: np.ndarray) -> dict[str, float]:
    return {
        "TEMP_first": float(temp[0]),
        "TEMP_last": float(temp[-1]),
        "TEMP_change": float(temp[-1] - temp[0]),
    }


def _acc_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    acc_magnitude: np.ndarray,
) -> dict[str, float]:
    return {
        "ACC_magnitude_mean": float(np.mean(acc_magnitude)),
        "ACC_magnitude_std": float(np.std(acc_magnitude)),
        "ACC_magnitude_energy": float(np.sum(np.square(acc_magnitude))),
        "ACC_axis_correlation_xy": _safe_corr(acc_x, acc_y),
        "ACC_axis_correlation_xz": _safe_corr(acc_x, acc_z),
        "ACC_axis_correlation_yz": _safe_corr(acc_y, acc_z),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
