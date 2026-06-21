"""Generate the final educational notebooks for the four-model WESAD study."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


SETUP = r"""
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import *
from src.evaluation import binary_metrics, collect_probabilities, per_subject_metrics, prediction_table, select_threshold
from src.helpers import count_parameters, set_seed
from src.training import pos_weight_from_labels, save_model_artifacts, train_with_early_stopping

set_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Périphérique :", device, "| CUDA :", torch.cuda.is_available())
"""


LOAD_SEQUENCES = r"""
sequence_dir = PROJECT_ROOT / "data" / "processed" / "sequence"
metadata_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
X_train = torch.load(sequence_dir / "X_train.pt", map_location="cpu", weights_only=True).float()
y_train = torch.load(sequence_dir / "y_train.pt", map_location="cpu", weights_only=True).float()
X_validation = torch.load(sequence_dir / "X_validation.pt", map_location="cpu", weights_only=True).float()
y_validation = torch.load(sequence_dir / "y_validation.pt", map_location="cpu", weights_only=True).float()
X_test = torch.load(sequence_dir / "X_test.pt", map_location="cpu", weights_only=True).float()
y_test = torch.load(sequence_dir / "y_test.pt", map_location="cpu", weights_only=True).float()
metadata_train = pd.read_csv(metadata_dir / "windows_train.csv")
metadata_validation = pd.read_csv(metadata_dir / "windows_validation.csv")
metadata_test = pd.read_csv(metadata_dir / "windows_test.csv")

train_subjects = set(metadata_train.subject_id)
val_subjects = set(metadata_validation.subject_id)
test_subjects = set(metadata_test.subject_id)
assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)
assert X_train.shape[1:] == X_validation.shape[1:] == X_test.shape[1:] == (960, 6)
assert torch.isfinite(X_train).all() and torch.isfinite(X_validation).all() and torch.isfinite(X_test).all()
print("Train/validation/test :", X_train.shape, X_validation.shape, X_test.shape)
"""


def notebook_02() -> None:
    write("02_mlp.ipynb", [
        md("""
# 02 — MLP sur caractéristiques statistiques

Le MLP reçoit des caractéristiques globales extraites de chaque fenêtre WESAD. La sélection des caractéristiques, leur normalisation, l'initialisation, la pondération de la perte et le seuil sont déterminés sans utiliser le test.
"""),
        md("## 1. Configuration, données et protection contre les fuites"),
        code(SETUP + "\n" + r"""
import joblib
from sklearn.preprocessing import StandardScaler
from src.preprocessing import extract_feature_table
from src.models import MLPClassifier, initialize_linear_layers

raw_dir = PROJECT_ROOT / "data" / "processed" / "sequence"
metadata_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
X_train_seq = torch.load(raw_dir / "X_train_raw.pt", map_location="cpu", weights_only=True).float()
X_validation_seq = torch.load(raw_dir / "X_validation_raw.pt", map_location="cpu", weights_only=True).float()
X_test_seq = torch.load(raw_dir / "X_test_raw.pt", map_location="cpu", weights_only=True).float()
y_train = torch.load(raw_dir / "y_train.pt", map_location="cpu", weights_only=True).float()
y_validation = torch.load(raw_dir / "y_validation.pt", map_location="cpu", weights_only=True).float()
y_test = torch.load(raw_dir / "y_test.pt", map_location="cpu", weights_only=True).float()
metadata_train = pd.read_csv(metadata_dir / "windows_train.csv")
metadata_validation = pd.read_csv(metadata_dir / "windows_validation.csv")
metadata_test = pd.read_csv(metadata_dir / "windows_test.csv")
assert set(metadata_train.subject_id).isdisjoint(metadata_validation.subject_id)
assert set(metadata_train.subject_id).isdisjoint(metadata_test.subject_id)
assert set(metadata_validation.subject_id).isdisjoint(metadata_test.subject_id)
"""),
        md("""
## 2. Extraction, filtrage et normalisation ajustés sur l'entraînement

Les doublons exacts, constantes et corrélations supérieures à 0,95 sont identifiés uniquement dans l'entraînement. Le `StandardScaler` est lui aussi ajusté uniquement sur ces sujets, puis appliqué sans réajustement à validation et test.
"""),
        code(r"""
feature_frames = {
    "train": extract_feature_table(X_train_seq.numpy()),
    "validation": extract_feature_table(X_validation_seq.numpy()),
    "test": extract_feature_table(X_test_seq.numpy()),
}
columns = list(feature_frames["train"].columns)
duplicates = [c for c, duplicate in zip(columns, feature_frames["train"].T.duplicated()) if duplicate]
columns = [c for c in columns if c not in duplicates]
constants = [c for c in columns if feature_frames["train"][c].nunique(dropna=False) <= 1]
columns = [c for c in columns if c not in constants]
upper = feature_frames["train"][columns].corr().abs().where(np.triu(np.ones((len(columns), len(columns))), 1).astype(bool))
correlated = [c for c in upper.columns if (upper[c] > 0.95).any()]
feature_columns = [c for c in columns if c not in correlated]

feature_scaler = StandardScaler().fit(feature_frames["train"][feature_columns])
X_train = torch.tensor(feature_scaler.transform(feature_frames["train"][feature_columns]), dtype=torch.float32)
X_validation = torch.tensor(feature_scaler.transform(feature_frames["validation"][feature_columns]), dtype=torch.float32)
X_test = torch.tensor(feature_scaler.transform(feature_frames["test"][feature_columns]), dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
def make_train_loader():
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
preprocessing_artifacts = PROJECT_ROOT / "artifacts/preprocessing"; preprocessing_artifacts.mkdir(parents=True, exist_ok=True)
(preprocessing_artifacts / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
(preprocessing_artifacts / "feature_selection.json").write_text(json.dumps({
    "fit_split": "train", "removed_exact_duplicates": duplicates, "removed_constants": constants,
    "removed_high_correlations": correlated, "correlation_threshold": 0.95,
}, indent=2), encoding="utf-8")
joblib.dump(feature_scaler, preprocessing_artifacts / "mlp_feature_scaler.joblib")
print(X_train.shape, "caractéristiques conservées :", len(feature_columns))
"""),
        md("## 3. Deux implémentations strictement équivalentes"),
        code(r"""
def build_sequential_mlp(input_dim: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
        nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
        nn.Linear(64, 1),
    )

sequential_model = build_sequential_mlp(X_train.shape[1]).to(device)
custom_model = MLPClassifier(X_train.shape[1]).to(device)
assert count_parameters(sequential_model) == count_parameters(custom_model)
assert sequential_model(X_train[:4].to(device)).shape == custom_model(X_train[:4].to(device)).shape == (4, 1)
print(sequential_model)
print(custom_model)
print("Paramètres entraînables :", count_parameters(custom_model))
"""),
        md("## 4. `named_parameters()`, `state_dict()` et initialisations"),
        code(r"""
for name, parameter in custom_model.named_parameters():
    print(name, tuple(parameter.shape), "trainable=", parameter.requires_grad)
print("Clés state_dict :", list(custom_model.state_dict()))

for method in ("gaussian", "constant", "xavier"):
    probe = MLPClassifier(X_train.shape[1])
    initialize_linear_layers(probe, method)
    first_weight = next(module.weight for module in probe.modules() if isinstance(module, nn.Linear))
    print(method, "mean=", float(first_weight.mean()), "std=", float(first_weight.std()))
"""),
        md("""
## 5. Sélection de l'initialisation sur validation

Les trois modèles utilisent la même architecture, les mêmes données et la même graine. L'initialisation maximisant le macro-F1 de validation est gelée avant toute évaluation test. Une initialisation constante est incluse à titre pédagogique ; sa symétrie peut pénaliser l'apprentissage.
"""),
        code(r"""
RUN_MLP_TRAINING = False
initialization_runs = {}
if RUN_MLP_TRAINING:
    for method in ("gaussian", "constant", "xavier"):
        set_seed(RANDOM_SEED)
        candidate = MLPClassifier(X_train.shape[1]).to(device)
        initialize_linear_layers(candidate, method)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        candidate, history, summary = train_with_early_stopping(
            candidate, make_train_loader(), validation_loader, criterion, optimizer, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        probabilities, labels = collect_probabilities(candidate, validation_loader, device)
        candidate_threshold, _ = select_threshold(labels, probabilities)
        metrics = binary_metrics(labels, probabilities, candidate_threshold)
        initialization_runs[method] = dict(model=candidate, history=history, summary=summary, threshold=candidate_threshold, metrics=metrics)
    initialization_table = pd.DataFrame([
        {"initialization": name, "validation_macro_f1": run["metrics"]["macro_f1"], "threshold": run["threshold"], "best_epoch": run["summary"]["best_epoch"]}
        for name, run in initialization_runs.items()
    ]).sort_values("validation_macro_f1", ascending=False)
    display(initialization_table)
    selected_initialization = initialization_table.iloc[0]["initialization"]
else:
    print("Non exécuté : activer RUN_MLP_TRAINING pour sélectionner l'initialisation sur validation.")
"""),
        md("## 6. Pondération, entraînement final et seuil de validation"),
        code(r"""
if RUN_MLP_TRAINING:
    variants = []
    for weighted in (False, True):
        set_seed(RANDOM_SEED)
        candidate = MLPClassifier(X_train.shape[1]).to(device)
        initialize_linear_layers(candidate, selected_initialization)
        weight = pos_weight_from_labels(y_train, device) if weighted else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        candidate, history, summary = train_with_early_stopping(candidate, make_train_loader(), validation_loader, criterion, optimizer, device, MAX_EPOCHS, PATIENCE)
        probabilities, labels = collect_probabilities(candidate, validation_loader, device)
        candidate_threshold, _ = select_threshold(labels, probabilities)
        metrics = binary_metrics(labels, probabilities, candidate_threshold)
        variants.append(dict(model=candidate, history=history, summary=summary, threshold=candidate_threshold, metrics=metrics, weighted=weighted))
    selected = max(variants, key=lambda run: run["metrics"]["macro_f1"])
    model, history, threshold = selected["model"], selected["history"], selected["threshold"]
    validation_metrics = selected["metrics"]
    test_start = time.perf_counter(); test_probabilities, test_true = collect_probabilities(model, test_loader, device); inference_time = time.perf_counter() - test_start
    test_metrics = {**binary_metrics(test_true, test_probabilities, threshold), "inference_time_seconds": inference_time}
    subject_metrics = per_subject_metrics(metadata_test, test_true, test_probabilities, threshold)
    display(pd.DataFrame({"validation": validation_metrics, "test": test_metrics}))
"""),
        md("## 7. Sauvegarde et rechargement du meilleur `state_dict`"),
        code(r"""
if RUN_MLP_TRAINING:
    artifact_dir = PROJECT_ROOT / "artifacts/models/mlp"
    save_model_artifacts(
        artifact_dir, model,
        {"model": "MLPClassifier", "input_dim": X_train.shape[1], "parameter_count": count_parameters(model), "initialization": selected_initialization,
         "selected_imbalance_method": "class_weight" if selected["weighted"] else "no_correction",
         "subject_split": SPLIT_SUBJECTS, "input_representation": "statistical_features", "feature_scaler_fit": "train", "seed": RANDOM_SEED,
         "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
         "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE, "classification_threshold": threshold,
         "artifact_paths": {"state_dict": "artifacts/models/mlp/best_model.pt", "feature_scaler": "artifacts/preprocessing/mlp_feature_scaler.joblib"}},
        threshold, history, validation_metrics, test_metrics, subject_metrics,
        prediction_table(metadata_test, test_true, test_probabilities, threshold),
        {**selected["summary"], "selected_initialization": selected_initialization,
         "selected_imbalance_method": "class_weight" if selected["weighted"] else "no_correction", "inference_time_seconds": inference_time},
    )
    before = model.eval()(X_test[:8].to(device)).detach()
    reloaded = MLPClassifier(X_train.shape[1]).to(device)
    reloaded.load_state_dict(torch.load(artifact_dir / "best_model.pt", map_location=device, weights_only=True)); reloaded.eval()
    after = reloaded(X_test[:8].to(device)).detach()
    torch.testing.assert_close(before, after)
    print("Rechargement vérifié.")
"""),
        md("""
## 8. Métriques disponibles

`binary_metrics` rapporte exactitude, précision et rappel par classe, macro-F1, F1 pondéré, ROC-AUC, average precision et matrice de confusion. Les valeurs existantes ci-dessous proviennent uniquement des artefacts réellement générés.
"""),
        code(r"""
artifact_dir = PROJECT_ROOT / "artifacts/models/mlp"
if (artifact_dir / "test_metrics.json").exists():
    with (artifact_dir / "test_metrics.json").open(encoding="utf-8") as handle: display(pd.Series(json.load(handle), name="test"))
else:
    print("Artefact MLP absent.")
"""),
    ])


def sequence_notebook(model_kind: str) -> None:
    is_lstm = model_kind == "lstm"
    class_name = "LSTMClassifier" if is_lstm else "SimpleRNNClassifier"
    title = "LSTM" if is_lstm else "RNN simple"
    recurrent_attr = "lstm" if is_lstm else "rnn"
    equation = r"""
\[
\begin{aligned}
f_t&=\sigma(W_f[x_t,h_{t-1}]+b_f),\\
i_t&=\sigma(W_i[x_t,h_{t-1}]+b_i),\\
\tilde c_t&=\tanh(W_c[x_t,h_{t-1}]+b_c),\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
o_t&=\sigma(W_o[x_t,h_{t-1}]+b_o),\quad h_t=o_t\odot\tanh(c_t).
\end{aligned}
\]

La porte d'oubli contrôle la mémoire conservée, la porte d'entrée contrôle l'écriture de la mémoire candidate, et la porte de sortie contrôle l'état caché exposé. Le chemin additif de la cellule peut faciliter la circulation d'information à long terme. Ces quatre transformations expliquent le nombre de paramètres supérieur à celui d'un RNN simple.
""" if is_lstm else r"""
\[
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h),\qquad z=W_{hy}h_{960}+b_y.
\]

Le même état caché sert à mémoriser le passé et à produire la représentation finale. BPTT dérive la perte à travers toute la chaîne des 960 états.
"""
    forward = r"""
outputs, (hidden, cell) = model.lstm(x)
print("Input:", x.shape)
print("All recurrent outputs:", outputs.shape)
print("Hidden state:", hidden.shape)
print("Cell state:", cell.shape)
print("Last layer hidden state:", hidden[-1].shape)
""" if is_lstm else r"""
outputs, hidden = model.rnn(x)
print("Input:", x.shape)
print("All recurrent outputs:", outputs.shape)
print("Final hidden states:", hidden.shape)
print("Last layer hidden state:", hidden[-1].shape)
"""
    interpretation = (
        "`outputs` contient l'état caché à chaque pas. `hidden` contient le dernier état caché de chaque couche. `cell` contient la mémoire finale de chaque couche : elle peut conserver une information différente de l'état caché exposé. `hidden[-1]` résume la séquence pour le classifieur."
        if is_lstm else
        "`outputs` contient l'état caché à chaque pas temporel. `hidden` contient l'état final de chaque couche récurrente. `hidden[-1]` est la représentation finale de la dernière couche utilisée par le classifieur."
    )
    takeaway = (
        "La cellule LSTM ajoute une mémoire additive contrôlée par les portes d'oubli, d'entrée et de sortie. Elle possède plus de paramètres, mais ce surcoût ne garantit pas une meilleure généralisation à des sujets inconnus."
        if is_lstm else
        "Le RNN met à jour son état caché à chaque mesure ; `hidden[-1]` représente la fenêtre entière pour la classification."
    )
    write(f"{'05_lstm' if is_lstm else '04_rnn'}.ipynb", [
        md(f"# {'05' if is_lstm else '04'} — {title}"),
        md("""
## 1. Définition du problème

Il s'agit d'une classification séquentielle **many-to-one** : une fenêtre physiologique produit un seul logit binaire stress/non-stress, et non une séquence de sorties.
"""),
        code(SETUP),
        md("""
## 2. Formes d'entrée et de sortie

L'entrée réelle a la forme `(batch_size, 960, 6)` et la sortie `(batch_size, 1)`. Toutes les fenêtres ont exactement 960 pas ; aucun padding ni masque artificiel n'est nécessaire.
"""),
        code(LOAD_SEQUENCES),
        md(f"## 3. Équations du {title}\n\n{equation}"),
        md("## 4. Architecture du modèle"),
        code(f"from src.models import {class_name}\nHIDDEN_SIZE = 32\nmodel = {class_name}(input_size=6, hidden_size=HIDDEN_SIZE).to(device)\nprint(model)\nprint('Paramètres :', count_parameters(model))"),
        md("## 5. Petit exemple synthétique"),
        code("batch_size, sequence_length, input_size = 2, 5, 6\nx = torch.randn(batch_size, sequence_length, input_size, device=device)\nprint(x.shape)"),
        md("## 6. Forward pass et états récurrents"),
        code(forward),
        md(f"## 7–9. Interprétation de `outputs`, des états finaux et de `hidden[-1]`\n\n{interpretation}"),
        md("## 10. Logit de classification binaire"),
        code("logits = model(x)\nprint('Logits :', logits.shape, logits.detach().cpu())\nassert logits.shape == (batch_size, 1)"),
        md("## 11. Calcul de la perte"),
        code("targets = torch.tensor([0., 1.], device=device)\ncriterion = nn.BCEWithLogitsLoss()\nloss = criterion(logits.reshape(-1), targets)\nprint('BCEWithLogitsLoss :', float(loss))"),
        md("## 12. `loss.backward()` et BPTT"),
        code("model.zero_grad(set_to_none=True)\nloss.backward()\nprint('Norme du gradient récurrent :', float(getattr(model, '" + recurrent_attr + "').weight_hh_l0.grad.norm()))"),
        md("## 13. Mise à jour par l'optimiseur"),
        code("optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\noptimizer.step()\noptimizer.zero_grad(set_to_none=True)"),
        md("## 14. Entraînement complet"),
        code(f"""
from src.experiments import run_validation_selected_experiment
RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    datasets = (TensorDataset(X_train, y_train), TensorDataset(X_validation, y_validation), TensorDataset(X_test, y_test))
    result = run_validation_selected_experiment(
        lambda: {class_name}(input_size=6, hidden_size=HIDDEN_SIZE), datasets,
        metadata_validation, metadata_test, PROJECT_ROOT / "artifacts/models/{model_kind}_fair",
        {{"model": "{class_name}", "model_class": "{class_name}", "architecture": {{"hidden_size": HIDDEN_SIZE, "layers": 1, "bidirectional": False}},
          "input_shape": [960, 6], "input_representation": "normalized_raw_sequences", "subject_split": SPLIT_SUBJECTS,
          "normalization_statistics": "artifacts/preprocessing/sequence_scaler.joblib"}},
        device, compare_weighted_loss=True,
    )
else:
    print("Non exécuté : activer RUN_FULL_TRAINING pour réentraîner le modèle.")
"""),
        md("## 15. Évaluation et métriques sauvegardées"),
        code(f"""
artifact_dir = PROJECT_ROOT / "artifacts/models/{model_kind}_fair"
if all((artifact_dir / name).exists() for name in ["validation_metrics.json", "test_metrics.json", "threshold.json"]):
    with (artifact_dir / "validation_metrics.json").open(encoding="utf-8") as f: validation_metrics = json.load(f)
    with (artifact_dir / "test_metrics.json").open(encoding="utf-8") as f: test_metrics = json.load(f)
    with (artifact_dir / "threshold.json").open(encoding="utf-8") as f: selected_threshold = json.load(f)["threshold"]
    display(pd.DataFrame({{"validation": validation_metrics, "test": test_metrics}}))
    print("Seuil sélectionné sur validation :", selected_threshold)
else:
    print("Artefacts absents : exécuter l'entraînement complet.")
"""),
        md("## 15.1 Rechargement du meilleur `state_dict`"),
        code(f"""
checkpoint = artifact_dir / "best_model.pt"
if checkpoint.exists():
    first = {class_name}(input_size=6, hidden_size=HIDDEN_SIZE).to(device)
    second = {class_name}(input_size=6, hidden_size=HIDDEN_SIZE).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    first.load_state_dict(state); second.load_state_dict(state); first.eval(); second.eval()
    batch = X_test[:8].to(device)
    with torch.no_grad():
        logits_first, logits_second = first(batch), second(batch)
    torch.testing.assert_close(logits_first, logits_second)
    print("Logits identiques après rechargement.")
"""),
        md("## 16. Matrice de confusion"),
        code("""
if 'test_metrics' in globals():
    cm = np.asarray(test_metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap='Blues')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['non-stress','stress'], yticklabels=['non-stress','stress'], xlabel='Prédit', ylabel='Réel')
    for row in range(2):
        for col in range(2): ax.text(col, row, cm[row, col], ha='center', va='center')
    plt.show()
"""),
        md("""
## 17. Interprétation

Le macro-F1 et le rappel stress doivent être examinés avec les résultats par sujet. Une forte validation ne garantit pas la généralisation : seuls quinze participants indépendants sont disponibles et les habitudes physiologiques peuvent varier fortement.
"""),
        md(f"""
## Ce que je dois retenir pour la soutenance

- {takeaway}
- BPTT propage le gradient de la perte vers les états antérieurs.
- Un gradient évanescent devient trop faible pour apprendre des dépendances lointaines ; un gradient explosif devient excessivement grand.
- L'écrêtage limite les gradients explosifs, mais ne restaure pas les gradients évanescents.
- `hidden[-1]` est utilisé parce qu'il correspond à la représentation finale de la dernière couche.
- La performance de validation peut ne pas se transférer aux participants test jamais vus.
"""),
    ])


def notebook_13() -> None:
    write("13_rnn_bptt_gradient_clipping.ipynb", [
        md("""
# 13 — BPTT et écrêtage du gradient

La fenêtre fixe est déroulée sur 960 pas :

```text
x1 -> h1 -> h2 -> ... -> h960 -> logit -> loss
      <- gradients propagés en arrière dans le temps <-
```

BPTT multiplie successivement les Jacobiennes récurrentes. Des produits contractants produisent des gradients évanescents ; des produits expansifs produisent des gradients explosifs. Le clipping contrôle les seconds mais ne répare pas les premiers.
"""),
        code(SETUP),
        md("## 1. Norme globale réutilisable et démonstration sur un lot"),
        code(r"""
from src.models import LSTMClassifier, SimpleRNNClassifier
from src.training import compute_global_gradient_norm

set_seed(RANDOM_SEED)
demo_model = SimpleRNNClassifier(input_size=6, hidden_size=32).to(device)
inputs = torch.randn(4, 960, 6, device=device)
targets = torch.tensor([0., 1., 0., 1.], device=device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(demo_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer.zero_grad()
logits = demo_model(inputs)
loss = criterion(logits.reshape(-1), targets)
loss.backward()
norm_before = compute_global_gradient_norm(demo_model.parameters())
returned_norm = torch.nn.utils.clip_grad_norm_(demo_model.parameters(), max_norm=1.0)
norm_after = compute_global_gradient_norm(demo_model.parameters())
optimizer.step()
print("avant=", norm_before, "retournée par clip_grad_norm_=", float(returned_norm), "après=", norm_after)
torch.testing.assert_close(torch.tensor(norm_before), returned_norm.detach().cpu())
assert norm_after <= 1.00001
"""),
        md("""
`clip_grad_norm_` renvoie la norme mesurée **avant** l'écrêtage. Si cette norme ne dépasse pas 1,0, l'opération ne modifie pas les gradients. Elle ne change ni la fonction de perte ni l'architecture.
"""),
        md("## 2. Données et comparaison contrôlée"),
        code(LOAD_SEQUENCES + "\nfrom src.experiments import run_validation_selected_experiment\ndatasets=(TensorDataset(X_train,y_train),TensorDataset(X_validation,y_validation),TensorDataset(X_test,y_test))"),
        code(r"""
RUN_CLIPPING_EXPERIMENT = False
if RUN_CLIPPING_EXPERIMENT:
    for name, clip in [("without_clipping", None), ("clip_norm_1", 1.0)]:
        run_validation_selected_experiment(
            lambda: SimpleRNNClassifier(input_size=6, hidden_size=32), datasets,
            metadata_validation, metadata_test, PROJECT_ROOT / "artifacts/models/rnn_clipping" / name,
            {"model": "RNN", "architecture": {"hidden_size": 32, "layers": 1}, "seed": RANDOM_SEED,
             "subject_split": SPLIT_SUBJECTS, "input_shape": [960, 6], "normalization_statistics": "train-only sequence scaler"},
            device, compare_weighted_loss=False, gradient_clip=clip,
            record_gradient_norms=True, record_validation_macro_f1=True,
        )
else:
    print("Expérience non exécutée. Les deux réglages réutilisent la même graine, architecture, initialisation, données et optimisation.")
"""),
        md("## 3. Courbes séparées et interprétations factuelles"),
        code(r"""
from IPython.display import Markdown, display
figure_dir = PROJECT_ROOT / "reports/figures/rnn_gradient_clipping"; figure_dir.mkdir(parents=True, exist_ok=True)
paths = {"sans clipping": PROJECT_ROOT / "artifacts/models/rnn_clipping/without_clipping/training_history.csv",
         "max_norm=1.0": PROJECT_ROOT / "artifacts/models/rnn_clipping/clip_norm_1/training_history.csv"}
histories = {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}

def plot_and_interpret(column, title, ylabel, filename, interpretation):
    if not histories:
        print(title, "— artefacts absents."); return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frame in histories.items():
        if column in frame: ax.plot(frame.epoch, frame[column], label=name)
    ax.set(title=title, xlabel="Époque", ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / filename, dpi=150); plt.show()
    display(Markdown("**Interprétation.** " + interpretation))

plot_and_interpret("train_loss", "Perte d'entraînement", "BCE", "training_loss.png", "Comparer la convergence et rechercher une divergence numérique ; aucune amélioration ne doit être affirmée sans écart observé.")
plot_and_interpret("validation_loss", "Perte de validation", "BCE", "validation_loss.png", "Une perte plus stable indique une optimisation plus régulière, pas nécessairement une meilleure généralisation.")
plot_and_interpret("validation_macro_f1", "Macro-F1 de validation", "Macro-F1", "validation_macro_f1.png", "Le clipping n'améliore le macro-F1 que si la courbe et le meilleur score de validation le montrent.")
plot_and_interpret("gradient_pre_max", "Norme maximale avant clipping", "Norme L2", "gradient_pre_max.png", "Des pics très élevés soutiennent l'hypothèse de gradients instables ; leur absence ne soutient pas cette hypothèse.")
plot_and_interpret("gradient_pre_median", "Norme médiane avant clipping", "Norme L2", "gradient_pre_median.png", "La médiane distingue des pics isolés d'une instabilité fréquente.")
if "max_norm=1.0" in histories:
    clipped = histories["max_norm=1.0"]
    fig, ax = plt.subplots(figsize=(7,4)); ax.plot(clipped.epoch, clipped.gradient_pre_mean, label="avant"); ax.plot(clipped.epoch, clipped.gradient_post_mean, label="après")
    ax.set(title="Normes avant/après clipping", xlabel="Époque", ylabel="Norme L2"); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "pre_vs_post.png", dpi=150); plt.show()
    activated = bool((clipped.gradient_pre_max > 1.0).any())
    activation_text = "est" if activated else "n'est pas"
    display(Markdown(f"**Interprétation.** Le clipping {activation_text} activé au moins une fois selon la norme pré-clipping. Il limite les explosions mais ne restaure pas les gradients évanescents."))
"""),
        md("## 3.1 Synthèse interprétative des courbes"),
        code(r"""
if histories:
    unclipped = histories.get("sans clipping")
    clipped = histories.get("max_norm=1.0")
    unstable = bool(unclipped is not None and (not np.isfinite(unclipped.train_loss).all() or unclipped.gradient_pre_max.max() > 10.0))
    activated = bool(clipped is not None and (clipped.gradient_pre_max > 1.0).any())
    f1_improved = bool(
        unclipped is not None and clipped is not None
        and clipped.validation_macro_f1.max() > unclipped.validation_macro_f1.max()
    )
    display(Markdown(
        f"- Gradients manifestement instables : **{'oui' if unstable else 'non selon ce critère'}**.\n"
        f"- Clipping effectivement activé : **{'oui' if activated else 'non'}**.\n"
        f"- Amélioration du meilleur macro-F1 validation : **{'oui' if f1_improved else 'non observée'}**.\n"
        "- Une baisse de norme soutient une amélioration de stabilité, mais pas automatiquement de généralisation.\n"
        "- Le clipping ne restaure jamais un gradient déjà évanescent."
    ))
else:
    print("Conclusion impossible : exécuter l'expérience contrôlée avant d'interpréter les courbes.")

def load_optional_metrics(directory):
    path = PROJECT_ROOT / "artifacts/models" / directory / "test_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

rnn_metrics = load_optional_metrics("rnn_fair")
lstm_metrics = load_optional_metrics("lstm_fair")
if rnn_metrics and lstm_metrics:
    rnn_cm = np.asarray(rnn_metrics["confusion_matrix"])
    mostly_non_stress = rnn_cm[:, 0].sum() > rnn_cm[:, 1].sum()
    recall_improved = lstm_metrics["stress_recall"] > rnn_metrics["stress_recall"]
    display(Markdown(
        f"- RNN prédit majoritairement non-stress : **{'oui' if mostly_non_stress else 'non'}**.\n"
        f"- Rappel stress amélioré par le LSTM : **{'oui' if recall_improved else 'non'}**.\n"
        "- Le surcoût paramétrique du LSTM n'est justifié que si ce gain de validation et de rappel est reproductible par sujet."
    ))
else:
    print("Comparaison RNN/LSTM équitable absente : aucune conclusion sur le rappel ou le surcoût paramétrique.")
"""),
        md("## 4. Comparaison équitable RNN–LSTM"),
        code(r"""
RUN_FAIR_RNN_LSTM = False
if RUN_FAIR_RNN_LSTM:
    fair_results = {}
    for name, factory in {"rnn_fair": lambda: SimpleRNNClassifier(6, 32), "lstm_fair": lambda: LSTMClassifier(6, 32)}.items():
        fair_results[name] = run_validation_selected_experiment(
            factory, datasets, metadata_validation, metadata_test, PROJECT_ROOT / "artifacts/models" / name,
            {"model": name, "architecture": {"hidden_size": 32, "layers": 1, "bidirectional": False}, "seed": RANDOM_SEED,
             "subject_split": SPLIT_SUBJECTS, "input_shape": [960, 6], "normalization_statistics": "train-only sequence scaler"},
            device, compare_weighted_loss=True,
        )
"""),
        code(r"""
def recurrent_artifact_row(name, directory):
    path = PROJECT_ROOT / "artifacts/models" / directory
    required = ["model_config.json", "validation_metrics.json", "test_metrics.json", "training_summary.json"]
    if not all((path / item).exists() for item in required): return {"model": name, "status": "artefact manquant"}
    data = [json.loads((path / item).read_text(encoding="utf-8")) for item in required]
    cfg, val, test, summary = data
    return {"model": name, "status": "disponible", "parameters": cfg.get("parameter_count"), "validation_macro_f1": val.get("macro_f1"),
            "test_macro_f1": test.get("macro_f1"), "weighted_f1": test.get("weighted_f1"), "stress_precision": test.get("stress_precision"),
            "stress_recall": test.get("stress_recall"), "roc_auc": test.get("roc_auc"), "average_precision": test.get("average_precision"),
            "best_epoch": summary.get("best_epoch"), "training_time": summary.get("training_time_seconds"), "inference_time": test.get("inference_time_seconds"),
            "confusion_matrix": test.get("confusion_matrix")}

fair_table = pd.DataFrame([recurrent_artifact_row("RNN", "rnn_fair"), recurrent_artifact_row("LSTM", "lstm_fair")])
display(fair_table)
if (PROJECT_ROOT / "artifacts/models/rnn_fair/per_subject_metrics.csv").exists(): display(pd.read_csv(PROJECT_ROOT / "artifacts/models/rnn_fair/per_subject_metrics.csv"))
if (PROJECT_ROOT / "artifacts/models/lstm_fair/per_subject_metrics.csv").exists(): display(pd.read_csv(PROJECT_ROOT / "artifacts/models/lstm_fair/per_subject_metrics.csv"))
"""),
        md("""
## 5. Discussion attendue

La distribution des prédictions et la matrice de confusion permettent de vérifier si le RNN prédit surtout non-stress. Le rappel stress du LSTM ne constitue une amélioration que s'il dépasse celui du RNN dans l'expérience équitable. Cette amélioration doit ensuite être mise en regard du nombre de paramètres supplémentaire. Avec quinze sujets, ces résultats restent sensibles aux participants de validation et de test.
"""),
    ])


def notebook_14() -> None:
    write("14_model_comparison.ipynb", [
        md("""
# 14 — Comparaison finale des quatre modèles

Ce notebook charge les artefacts sans réentraîner. Le **gagnant du protocole** maximise le macro-F1 de validation. La **généralisation test observée** est mesurée après gel de l'architecture, de la perte et du seuil ; elle ne modifie jamais la sélection.
"""),
        code(SETUP),
        code(r"""
MODEL_SPECS = [
    ("MLP", "statistical features", ["mlp"]),
    ("CNN 2D", "BVP/EDA/ACC-magnitude scalograms", ["cnn2d"]),
    ("RNN", "normalized raw sequences", ["rnn_fair", "rnn"]),
    ("LSTM", "normalized raw sequences", ["lstm_fair", "lstm"]),
]
ALLOWED_MODELS = {"MLP", "CNN 2D", "RNN", "LSTM"}

def load_row(model, representation, candidates):
    row = {"model": model, "input representation": representation, "status": "missing artifact"}
    for directory in candidates:
        path = PROJECT_ROOT / "artifacts/models" / directory
        required = ["model_config.json", "validation_metrics.json", "test_metrics.json", "training_summary.json", "threshold.json"]
        if not all((path / item).exists() for item in required): continue
        config, validation, test, summary, threshold = [json.loads((path / item).read_text(encoding="utf-8")) for item in required]
        architecture = config.get("architecture", {})
        hidden = architecture.get("hidden_size", config.get("hidden_size")) if isinstance(architecture, dict) else config.get("hidden_size")
        if model in {"RNN", "LSTM"} and hidden not in {None, 32}:
            row["status"] = f"incompatible artifact ({directory}: hidden_size={hidden})"; continue
        row.update({"status": "available", "artifact directory": directory, "parameter count": config.get("parameter_count"),
            "best validation epoch": summary.get("best_epoch", config.get("best_validation_epoch")), "validation macro F1": validation.get("macro_f1"),
            "test macro F1": test.get("macro_f1"), "weighted F1": test.get("weighted_f1"), "stress precision": test.get("stress_precision"),
            "stress recall": test.get("stress_recall"), "ROC-AUC": test.get("roc_auc"), "average precision": test.get("average_precision"),
            "selected threshold": threshold.get("threshold"), "training time": summary.get("training_time_seconds"),
            "inference time": test.get("inference_time_seconds")})
        break
    return row

comparison = pd.DataFrame([load_row(*spec) for spec in MODEL_SPECS])
assert set(comparison.model) == ALLOWED_MODELS and len(comparison) == 4
display(comparison)
missing = comparison[comparison.status != "available"]
if len(missing): print("Artefacts manquants ou incompatibles :", missing[["model", "status"]].to_dict("records"))
"""),
        md("## Gagnant du protocole et généralisation observée"),
        code(r"""
available = comparison[comparison.status == "available"].copy()
if len(available):
    winner = available.loc[available["validation macro F1"].idxmax()]
    print("Gagnant du protocole (validation) :", winner.model)
    print("Généralisation test observée après gel :", winner["test macro F1"])
    available.set_index("model")[["validation macro F1", "test macro F1"]].plot.bar(figsize=(8,4), rot=0)
    plt.ylabel("Macro-F1"); plt.tight_layout(); plt.show()
"""),
        md("## Résultats par sujet"),
        code(r"""
frames = []
for _, row in available.iterrows():
    path = PROJECT_ROOT / "artifacts/models" / row["artifact directory"] / "per_subject_metrics.csv"
    if path.exists():
        frame = pd.read_csv(path); frame.insert(0, "model", row.model); frames.append(frame)
per_subject_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
display(per_subject_results)
"""),
        md("""
## Synthèse

- MLP : représentation statistique globale.
- CNN 2D : motifs temps-fréquence locaux des scalogrammes.
- RNN : dépendances temporelles ordonnées via un état caché simple.
- LSTM : dépendances temporelles avec cellule mémoire et portes.

Une architecture plus complexe ne garantit pas une meilleure généralisation indépendante des sujets.
"""),
    ])


if __name__ == "__main__":
    notebook_02()
    sequence_notebook("rnn")
    sequence_notebook("lstm")
    notebook_13()
    notebook_14()
