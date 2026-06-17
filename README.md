# WESAD Stress Classification Project

This project builds a fair comparison of MLP, CNN, RNN, LSTM, GRU, CNN-LSTM, and CNN-GRU models for binary stress detection using WESAD wrist signals.

## Project structure

```text
app.py
requirements.txt
wesad_utils/          reusable preprocessing, training, evaluation, config, and helpers
notebooks/            executable experiment workflow
data/                 raw and generated datasets
artifacts/            trained models, metrics, scalers, and configs
reports/              generated figures and tables
```

The notebooks remain the main workflow for running and presenting the experiments. Reusable preprocessing, training, evaluation, and artifact-saving code lives in `wesad_utils/`. Model architecture classes remain inside their notebooks so they are visible during presentation.

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
