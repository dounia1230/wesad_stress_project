# WESAD Stress Classification

PyTorch research project for binary stress classification on WESAD wrist signals. The repository builds leakage-safe subject splits, preprocesses multimodal wearable signals, trains four neural architectures, compares validation-selected variants, and saves reproducible metrics and artifacts.

This is a research demonstration, not a medical diagnostic tool.

## What This Project Does

- Uses WESAD wrist modalities: `BVP`, `EDA`, `TEMP`, `ACC_x`, `ACC_y`, `ACC_z`
- Converts WESAD labels into a binary task:
  - `stress`: label `2`
  - `non-stress`: labels `1` and `3`
- Splits by subject before windowing to avoid participant leakage
- Creates 30-second windows with a 15-second stride at 32 Hz
- Compares:
  - MLP on extracted statistical features
  - CNN 2D on three-channel CWT scalograms
  - RNN and LSTM
- Trains each model with and without class-weighted loss
- Selects the final variant by validation macro F1, then reports held-out test metrics
- Saves preprocessing artifacts, trained model artifacts, predictions, metrics, plots, and SHAP explanations

## Current Result Snapshot

The final comparison notebook writes `artifacts/results/all_model_metrics.csv` and `artifacts/results/best_model.json`.

Current validation-selected model:

```text
model: mlp
validation_macro_f1: 0.9061
test_macro_f1: 0.8564
stress_precision: 0.9355
stress_recall: 0.6797
selected_imbalance_method: no_correction
```

Current CNN 2D result:

```text
model: cnn2d
validation_macro_f1: 0.8626
test_macro_f1: 0.8440
stress_precision: 0.9535
stress_recall: 0.6406
selected_imbalance_method: weighted
```

Model selection remains based on validation macro F1. Test metrics are reported only after the model, loss weighting, and classification threshold are frozen.

## Repository Layout

```text
.
|-- app.py                         # Minimal Streamlit placeholder
|-- requirements.txt
|-- src/
|   |-- config.py                  # Shared constants and experiment settings
|   |-- models.py                  # Model architecture classes
|   |-- preprocessing.py           # Signal alignment, windowing, feature extraction
|   |-- training.py                # Train/eval loops and early stopping
|   |-- evaluation.py              # Metrics, probabilities, prediction tables
|   `-- helpers.py                 # Reproducibility and utility helpers
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
|   `-- 14_model_comparison.ipynb
|-- data/                          # Local raw and processed data, ignored by Git
|-- artifacts/                     # Generated scalers, models, metrics, ignored by Git
`-- reports/                       # Generated figures and tables
```

## Setup

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your local CUDA version before running the notebooks. The code also runs on CPU for smaller experiments.

## Dataset

Download WESAD from the official dataset source and place the extracted files here:

```text
data/WESAD/WESAD/
```

Expected subject folders include values such as `S2`, `S3`, `S4`, and so on. Raw WESAD data is intentionally ignored by Git.

## Experiment Configuration

Core settings live in `src/config.py`:

```text
target sample rate: 32 Hz
window length: 30 seconds
stride: 15 seconds
batch size: 64
learning rate: 1e-3
weight decay: 1e-4
max epochs: 100
early-stopping patience: 10
random seed: 42
```

Subject split:

```text
train:      S3, S4, S6, S7, S8, S9, S10, S13, S16, S17
validation: S5, S15
test:       S2, S11, S14
```

## Run Order

Run notebooks from a fresh kernel in this order:

```text
00_data_exploration.ipynb
01_preprocessing_and_splits.ipynb
02_mlp.ipynb
04_rnn.ipynb
05_lstm.ipynb
10_wesad_scalogram_generation.ipynb
11_cnn2d_scalogram_experiments.ipynb
12_cnn2d_ablation_and_feature_maps.ipynb
13_rnn_bptt_gradient_clipping.ipynb
14_model_comparison.ipynb
```

Notebook 10 derives BVP, EDA, and acceleration-magnitude Morlet scalograms from
the already standardized sequence windows. Scalogram normalization is fitted on
training subjects only. Notebooks 11--13 perform model and threshold selection
exclusively on validation data; notebook 14 loads saved artifacts without
retraining.

The model notebooks assume preprocessing has already written processed tensors and metadata under `data/processed/`.

## Preprocessing Summary

`01_preprocessing_and_splits.ipynb` performs the reproducible data pipeline:

1. Load each subject independently.
2. Keep WESAD labels `1`, `2`, and `3`.
3. Resample wrist signals to 32 Hz:
   - BVP: downsample from 64 Hz
   - EDA and TEMP: interpolate from 4 Hz
   - ACC: keep at 32 Hz and split into x/y/z channels
4. Create windows inside continuous same-label segments.
5. Map labels to binary classes.
6. Fit sequence scaling on training windows only.
7. Transform validation and test windows with the training scaler.
8. Save tensors, metadata, split definitions, scalers, and label mappings.

The MLP notebook extracts statistical features from raw windows, removes exact duplicates, constant features, and highly correlated features using training data only, then fits a feature scaler on the selected training columns.

## Outputs

Generated files are written mainly under:

```text
data/processed/
artifacts/preprocessing/
artifacts/models/<model_name>/
artifacts/results/
reports/figures/
reports/tables/
```

Common model artifacts:

```text
best_model.pt
model_config.json
threshold.json
training_history.csv
training_summary.json
validation_metrics.json
test_metrics.json
per_subject_metrics.csv
test_predictions.csv
```

Final comparison outputs:

```text
artifacts/results/all_model_metrics.csv
artifacts/results/all_model_metrics.json
artifacts/results/best_model.json
```

## Metrics

The evaluation reports:

- macro F1
- weighted F1
- stress precision
- stress recall
- ROC-AUC
- average precision
- confusion matrix
- per-subject metrics

Model selection uses validation macro F1. Test metrics are used only for final reporting.

## SHAP Explanations

`02_mlp.ipynb` includes SHAP explanations for the feature-based MLP and writes plots under:

```text
reports/figures/shap/
```

These explanations describe the fitted model's feature sensitivity. They should not be interpreted as physiological causality.

## Streamlit App

The Streamlit file is currently a lightweight placeholder:

```bash
streamlit run app.py
```

The complete research workflow is in the notebooks. The app can be extended later into a model-comparison or inference interface.

## Reproducibility Notes

- Raw WESAD data, processed tensors, scalers, trained weights, and most generated artifacts are ignored by Git.
- Re-run notebooks in order to regenerate local artifacts.
- Model classes live in `src/models.py` and are imported by the notebooks.
- Shared reusable code lives in `src/`.
- The project is designed for transparent experimentation, not production deployment.

## Limitations

WESAD is small for subject-independent deep learning. The current workflow uses 15 participants, and results may be sensitive to the chosen subject split, sensor placement, and individual physiological differences. Per-subject metrics should be reviewed alongside aggregate scores.

## Citation

If presenting or publishing results, cite the official WESAD dataset paper and follow the dataset license terms.
