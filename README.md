# WESAD Stress Classification

PyTorch research project for subject-independent binary stress classification from WESAD wrist signals. The final study compares exactly four model families:

- MLP on statistical features;
- CNN 2D on BVP, EDA, and acceleration-magnitude scalograms;
- simple RNN on normalized raw sequences;
- LSTM on normalized raw sequences.

The target is `0 = non-stress`, `1 = stress`. This repository is a research demonstration, not a medical diagnostic system.

## Data protocol

Signals are aligned to 32 Hz and divided into 30-second windows with a 15-second stride. Each raw sequence has shape `(960, 6)` with channels `BVP`, `EDA`, `TEMP`, `ACC_x`, `ACC_y`, and `ACC_z`.

The fixed subject split is applied before windowing, normalization, feature selection, or scalogram generation:

```text
train:      S3, S4, S6, S7, S8, S9, S10, S13, S16, S17
validation: S5, S15
test:       S2, S11, S14
```

Training subjects alone fit sequence scalers, feature filtering, feature scaling, scalogram normalization, and class weights. Validation selects initialization, loss weighting, early-stopping epoch, architecture variants, and classification thresholds. Test data is used only after all choices are frozen.

## Models

### MLP

The MLP uses global statistical, BVP, EDA, temperature, and movement features. Duplicate, constant, and highly correlated features are detected from training subjects only. The notebook compares equivalent `nn.Sequential` and custom `nn.Module` implementations and demonstrates Gaussian, constant, and Xavier initialization.

### CNN 2D

The CNN receives tensors shaped `(N, 3, 64, 64)`. Each channel is a Morlet CWT log-scalogram derived from BVP, EDA, or acceleration magnitude. Scalogram channel means and standard deviations are fitted globally on training subjects and reused unchanged for validation and test.

The CNN study includes manual cross-correlation, manual max/average pooling, PyTorch equivalence checks, padding and stride studies, pooling and filter-capacity studies, optional `1 × 1` convolution, and feature-map visualization.

### RNN and LSTM

Both recurrent models perform many-to-one classification from `(batch, 960, 6)` to one binary logit. Their fair comparison uses the same hidden size, recurrent-layer count, data, optimizer, learning rate, weight decay, batch size, patience, threshold search, and seed. The LSTM has more parameters because of its gates and memory cell.

Notebook 13 explains BPTT, vanishing/exploding gradients, global gradient norms, and a controlled clipping comparison.

## Repository layout

```text
.
|-- app.py
|-- requirements.txt
|-- src/
|   |-- config.py
|   |-- preprocessing.py
|   |-- scalograms.py
|   |-- models.py
|   |-- training.py
|   |-- experiments.py
|   |-- evaluation.py
|   |-- manual_ops.py
|   |-- visualization.py
|   `-- helpers.py
|-- notebooks/
|   |-- 00_data_exploration.ipynb
|   |-- 01_preprocessing_and_splits.ipynb
|   |-- 02_mlp.ipynb
|   |-- 04_rnn.ipynb
|   |-- 05_lstm.ipynb
|   |-- 10_wesad_scalogram_generation.ipynb
|   |-- 11_cnn2d_scalogram_experiments.ipynb
|   |-- 12_cnn2d_ablation_and_feature_maps.ipynb
|   |-- 13_rnn_bptt_gradient_clipping.ipynb
|   |-- 14_model_comparison.ipynb
|   `-- archive/                    # Historical notebooks, excluded from workflow
|-- reports/
|-- data/                           # Local and ignored
`-- artifacts/                      # Generated and ignored
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately when required by the local GPU environment.

Place WESAD under:

```text
data/WESAD/WESAD/
```

## Notebook execution order

```text
00_data_exploration.ipynb
01_preprocessing_and_splits.ipynb
02_mlp.ipynb
10_wesad_scalogram_generation.ipynb
11_cnn2d_scalogram_experiments.ipynb
12_cnn2d_ablation_and_feature_maps.ipynb
04_rnn.ipynb
05_lstm.ipynb
13_rnn_bptt_gradient_clipping.ipynb
14_model_comparison.ipynb
```

Training and ablation cells use explicit `RUN_*` switches. This prevents accidental retraining or cache replacement.

## Generated artifacts

Preprocessing outputs:

```text
data/processed/sequence/
data/processed/features/
data/processed/scalograms/
artifacts/preprocessing/
```

Each trained model writes under `artifacts/models/<model>/`:

```text
best_model.pt
model_config.json
experiment_config.json          # newer shared-runner experiments
threshold.json
training_history.csv
training_summary.json
validation_metrics.json
test_metrics.json
validation_predictions.csv      # newer shared-runner experiments
test_predictions.csv
per_subject_metrics.csv
```

The final comparison reads these saved artifacts and never retrains models.

## Metrics

Evaluation supports accuracy, macro F1, weighted F1, class precision/recall, ROC-AUC, average precision, confusion matrices, and per-subject metrics. Model selection uses validation macro F1; test macro F1 is reported only as observed generalization.

## Current generated results

The following values already exist in local generated artifacts:

| Model | Validation macro F1 | Test macro F1 | Stress precision | Stress recall |
|---|---:|---:|---:|---:|
| MLP | 0.8793 | 0.8563 | 0.9000 | 0.7031 |
| CNN 2D | 0.8626 | 0.8440 | 0.9535 | 0.6406 |
| RNN | 0.7507 | 0.4352 | 0.2264 | 0.2812 |
| LSTM | 0.7971 | 0.6215 | 0.6154 | 0.3125 |

The RNN and LSTM rows come from the controlled 32-unit comparison. The protocol winner is the MLP because it has the highest validation macro F1; test scores were not used to make that choice.

## Reproducibility and exclusions

- Python, NumPy, and PyTorch use seed 42.
- Raw WESAD files, processed tensors, scalogram caches, model weights, generated artifacts, virtual environments, and notebook checkpoints must remain untracked.
- Historical notebooks under `notebooks/archive/` are not part of the active workflow or final comparison.
- No test-set information is used for tuning.

## Limitations

Only fifteen independent participants are available. Validation contains two participants, so model rankings may be sensitive to participant-specific physiology, sensor placement, and activity. Overlapping windows increase the number of samples but not the number of independent subjects. Feature maps and attribution plots describe model responses and must not be interpreted as physiological causality.

When presenting results, report aggregate and per-subject metrics together and cite the official WESAD dataset paper.
