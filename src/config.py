"""Project-wide constants for WESAD preprocessing and experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SUBJECTS = [
    "S3",
    "S4",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S13",
    "S16",
    "S17",
]
VALIDATION_SUBJECTS = ["S5", "S15"]
TEST_SUBJECTS = ["S2", "S11", "S14"]

SPLIT_SUBJECTS = {
    "train": TRAIN_SUBJECTS,
    "validation": VALIDATION_SUBJECTS,
    "test": TEST_SUBJECTS,
}

SEQUENCE_CHANNELS = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z"]
ACCEPTED_LABELS = {1, 2, 3}
BINARY_LABEL_MAP = {1: 0, 2: 1, 3: 0}

TARGET_HZ = 32
WINDOW_SECONDS = 30
STRIDE_SECONDS = 15
WINDOW_SAMPLES = TARGET_HZ * WINDOW_SECONDS
STRIDE_SAMPLES = TARGET_HZ * STRIDE_SECONDS

WRIST_SAMPLE_RATES = {
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
    "ACC": 32,
}
LABEL_HZ = 700

RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10

# Time-frequency representation. These values are the single source of truth for
# notebooks 10--12 and are deliberately shared across all three modalities.
SCALOGRAM_CHANNELS = ["BVP", "EDA", "ACC_magnitude"]
SCALOGRAM_WAVELET = "morl"
SCALOGRAM_SCALES = tuple(range(1, 65))
SCALOGRAM_SIZE = (64, 64)
SCALOGRAM_DIR = PROJECT_ROOT / "data" / "processed" / "scalograms"
SCALOGRAM_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessing" / "scalograms"
