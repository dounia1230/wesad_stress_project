# WESAD Stress Classification

Deep learning project for binary stress detection using WESAD wrist-sensor signals.

The project compares statistical-feature and sequence-based neural models on the same subject-level train, validation, and test splits. It includes preprocessing, model training, evaluation, per-subject metrics, and final model comparison notebooks.

## Project Layout

```text
wesad_stress_project/
  src/          reusable preprocessing, training, evaluation, and config code
  models/       PyTorch model definitions
  notebooks/    ordered experiment workflow
  data/         raw and processed WESAD data
  artifacts/    trained models, scalers, metrics, and predictions
  reports/      generated figures and tables
```

See [`wesad_stress_project/README.md`](wesad_stress_project/README.md) for the full notebook workflow.

