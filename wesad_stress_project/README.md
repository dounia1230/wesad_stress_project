# WESAD Stress Classification Project

This project builds a fair comparison of MLP, CNN, RNN, LSTM, GRU, CNN-LSTM, and CNN-GRU models for binary stress detection using WESAD wrist signals.

## Project structure

```text
wesad_stress_project/
  app.py
  requirements.txt
  src/                  reusable preprocessing, training, evaluation, and config
  models/               reusable PyTorch model definitions
  notebooks/            executable experiment workflow
  data/                 raw and generated datasets
  artifacts/            trained models, metrics, scalers, and configs
  reports/              generated figures and tables
```

The notebooks remain the main workflow for running and presenting the experiments. Reusable code now lives in `src/` and `models/` so model definitions and helper functions are not tied to notebook cells.

The repository root README gives a short GitHub overview, while this file documents the project workflow.

Run the notebooks in order:

1. `notebooks/00_data_exploration.ipynb`
2. `notebooks/01_preprocessing_and_splits.ipynb`
3. `notebooks/02_mlp.ipynb`
4. `notebooks/03_cnn.ipynb`
5. `notebooks/04_rnn.ipynb`
6. `notebooks/05_lstm.ipynb`
7. `notebooks/06_gru.ipynb`
8. `notebooks/07_cnn_lstm.ipynb`
9. `notebooks/08_cnn_gru.ipynb`
10. `notebooks/09_model_comparison.ipynb`

The preprocessing notebook creates the shared train, validation, and test windows. All model notebooks load those same windows so the comparison uses the same participants, labels, and windows.
