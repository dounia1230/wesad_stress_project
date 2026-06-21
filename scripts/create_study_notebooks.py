"""Create the CNN2D notebooks 10--12 from reviewable source cells.

Notebooks 13 and 14 are finalized by ``finalize_four_model_notebooks.py``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


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

from src.config import *
from src.helpers import count_parameters, set_seed

set_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Périphérique sélectionné :", device)
print("CUDA disponible :", torch.cuda.is_available())
"""


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, NOTEBOOKS / name)


def notebook_10() -> None:
    write("10_wesad_scalogram_generation.ipynb", [
        md(r"""
# 10 — Génération des scalogrammes WESAD

Ce notebook construit une représentation temps-fréquence sans modifier les fenêtres de 30 secondes produites par le notebook 01. La séparation par sujet précède donc la normalisation temporelle, la CWT et toute normalisation des scalogrammes. Les statistiques du test ne participent à aucune décision.
"""),
        code(SETUP),
        md(r"""
## Protocole de transformation

Pour chaque fenêtre standardisée avec les statistiques des sujets d'entraînement, trois signaux sont retenus : BVP, EDA et norme de l'accélération

\[
a(t)=\sqrt{a_x(t)^2+a_y(t)^2+a_z(t)^2}.
\]

La chaîne est : fenêtre temporelle standardisée → CWT de Morlet → module des coefficients → `log1p` → interpolation bilinéaire 64 × 64 → normalisation par canal apprise uniquement sur l'entraînement.

Employer les mêmes échelles 1 à 64 pour les trois modalités est une hypothèse simplificatrice : BVP, EDA et accélération n'occupent pas les mêmes bandes fréquentielles. Ces paramètres constituent ici une référence fixée sur validation, jamais sur test.
"""),
        code(r"""
from src.scalograms import (
    build_scalogram_artifacts,
    validate_subject_splits,
    window_to_log_scalogram,
)

validate_subject_splits()
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert set(SPLIT_SUBJECTS[left]).isdisjoint(SPLIT_SUBJECTS[right])
print(SPLIT_SUBJECTS)
"""),
        md("## Contrôle synthétique de forme et de valeurs"),
        code(r"""
synthetic_window = np.random.default_rng(RANDOM_SEED).normal(size=(960, 6)).astype(np.float32)
synthetic_map = window_to_log_scalogram(synthetic_window)
assert synthetic_map.shape == (3, 64, 64)
assert torch.isfinite(synthetic_map).all()
print("Forme d'un scalogramme :", tuple(synthetic_map.shape))
"""),
        md(r"""
## Génération complète

La cellule suivante est volontairement protégée pour éviter d'écraser un cache volumineux. Mettre `RUN_SCALOGRAM_GENERATION=True` après exécution du notebook 01. Le fichier de métadonnées sauvegarde les sujets, le nom de l'ondelette, les échelles, la fréquence, l'ordre des canaux, la taille, la graine et les moments d'entraînement.
"""),
        code(r"""
RUN_SCALOGRAM_GENERATION = False
scalogram_dir = PROJECT_ROOT / "data" / "processed" / "scalograms"
cache_ready = all(
    (scalogram_dir / f"X_{split}.pt").exists()
    and (scalogram_dir / f"y_{split}.pt").exists()
    for split in SPLIT_SUBJECTS
)

if RUN_SCALOGRAM_GENERATION:
    scalogram_metadata = build_scalogram_artifacts(PROJECT_ROOT, overwrite=False)
    display(pd.Series(scalogram_metadata, name="valeur"))
elif cache_ready:
    metadata_path = PROJECT_ROOT / "artifacts" / "preprocessing" / "scalograms" / "scalogram_metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        scalogram_metadata = json.load(handle)
    print("Cache de scalogrammes déjà disponible ; aucune régénération nécessaire.")
    display(pd.Series(scalogram_metadata, name="valeur"))
else:
    print("Non exécuté : activer RUN_SCALOGRAM_GENERATION pour générer les données réelles.")
"""),
        md("## Validation du cache sauvegardé"),
        code(r"""
scalogram_dir = PROJECT_ROOT / "data" / "processed" / "scalograms"
metadata_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
if (scalogram_dir / "X_train.pt").exists():
    for split in SPLIT_SUBJECTS:
        X = torch.load(scalogram_dir / f"X_{split}.pt", map_location="cpu", weights_only=True)
        y = torch.load(scalogram_dir / f"y_{split}.pt", map_location="cpu", weights_only=True)
        meta = pd.read_csv(metadata_dir / f"windows_{split}.csv")
        assert X.shape[1:] == (3, 64, 64) and len(X) == len(y) == len(meta)
        assert torch.isfinite(X).all() and set(y.int().tolist()) <= {0, 1}
        assert set(meta.subject_id) <= set(SPLIT_SUBJECTS[split])
        print(split, tuple(X.shape))
else:
    print("Cache absent : exécuter la cellule de génération complète.")
"""),
    ])


def notebook_11() -> None:
    write("11_cnn2d_scalogram_experiments.ipynb", [
        md(r"""
# 11 — Expériences CNN 2D sur scalogrammes

L'objectif est une classification binaire plusieurs-vers-un : chaque tenseur temps-fréquence conduit à un logit brut. `BCEWithLogitsLoss` applique de façon stable la sigmoïde et l'entropie croisée. La pondération éventuelle et le seuil sont choisis sur validation uniquement.
"""),
        code(SETUP),
        md(r"""
## Corrélation croisée 2D manuelle

PyTorch nomme `Conv2d` une opération qui réalise en pratique une corrélation croisée. Pour une entrée \(H\times W\), un noyau \(K\), un padding \(P\) et un pas \(S\),

\[
H_{out}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1,\quad
W_{out}=\left\lfloor\frac{W+2P-K}{S}\right\rfloor+1.
\]

Ainsi, pour 64 × 64, noyau 3 et pas 1 : sans padding on obtient 62 × 62, avec padding 1 on conserve 64 × 64.
"""),
        code(r"""
import torch.nn.functional as F
from src.manual_ops import corr2d, manual_avg_pool2d, manual_max_pool2d

input_matrix = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
kernel = torch.tensor([[1., 0.], [0., -1.]])
manual_output = corr2d(input_matrix, kernel, stride=1, padding=0)
pytorch_output = F.conv2d(input_matrix[None, None], kernel[None, None]).squeeze()
print("Entrée :\n", input_matrix, "\nNoyau :\n", kernel, "\nSortie :\n", manual_output)
print("Dimension : floor((3 + 2*0 - 2) / 1) + 1 = 2")
torch.testing.assert_close(manual_output, pytorch_output)
"""),
        md("## Max-pooling et average-pooling manuels"),
        code(r"""
pool_input = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
manual_max = manual_max_pool2d(pool_input, kernel_size=2, stride=2)
manual_avg = manual_avg_pool2d(pool_input, kernel_size=2, stride=2)
torch.testing.assert_close(manual_max, F.max_pool2d(pool_input[None, None], 2, 2).squeeze())
torch.testing.assert_close(manual_avg, F.avg_pool2d(pool_input[None, None], 2, 2).squeeze())
print(pool_input, "\nMax :", manual_max, "\nMoyenne :", manual_avg)
assert (64 - 3) // 1 + 1 == 62
assert (64 + 2 - 3) // 1 + 1 == 64
"""),
        md("## Modèles réutilisables et nombre de paramètres"),
        code(r"""
from src.models import WESADScalogramCNN

cnn2d = WESADScalogramCNN().to(device)
toy = torch.randn(4, 3, 64, 64, device=device)
assert cnn2d(toy).shape == (4, 1)
print("CNN 2D :", count_parameters(cnn2d), "paramètres")
print("Périphérique modèle :", next(cnn2d.parameters()).device)
print("Périphérique lot :", toy.device)
assert next(cnn2d.parameters()).device == toy.device
"""),
        md(r"""
Le CNN emploie des champs récepteurs locaux et partage ses poids sur les deux dimensions temps-fréquence. Le nombre de paramètres accompagne les mesures de performance.
"""),
        md("## Chargement du jeu de scalogrammes"),
        code(r"""
from src.scalograms import WESADScalogramDataset, validate_subject_splits

scalogram_dir = PROJECT_ROOT / "data" / "processed" / "scalograms"
metadata_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
validate_subject_splits()

def load_scalogram_datasets():
    return tuple(
        WESADScalogramDataset(
            scalogram_dir / f"X_{split}.pt",
            scalogram_dir / f"y_{split}.pt",
            metadata_dir / f"windows_{split}.csv",
        )
        for split in ("train", "validation", "test")
    )

if (scalogram_dir / "X_train.pt").exists():
    datasets = load_scalogram_datasets()
    sample_x, sample_y, sample_metadata = datasets[0][0]
    print(sample_x.shape, sample_y.shape, sample_metadata["subject_id"])
else:
    datasets = None
    print("Scalogrammes absents : exécuter le notebook 10.")
"""),
        md(r"""
## Protocole d'entraînement

Chaque variante repart de la graine 42. Adam, le taux d'apprentissage, la régularisation, la taille de lot, le nombre maximal d'époques et la patience viennent de `src.config`. Si la pondération est comparée, `pos_weight` est calculé uniquement sur les étiquettes d'entraînement. La variante et le seuil (0,10 à 0,90) maximisant le macro-F1 sont déterminés sur validation. Le test n'est chargé pour l'inférence qu'après gel de ces choix.
"""),
        code(r"""
from src.experiments import run_validation_selected_experiment

RUN_CNN2D_TRAINING = False
results = {}
if RUN_CNN2D_TRAINING:
    if datasets is None:
        raise FileNotFoundError("Exécuter le notebook 10 avant l'entraînement.")
    validation_metadata = pd.read_csv(metadata_dir / "windows_validation.csv")
    test_metadata = pd.read_csv(metadata_dir / "windows_test.csv")
    common = {
        "seed": RANDOM_SEED,
        "subject_split": SPLIT_SUBJECTS,
        "input_channels": SCALOGRAM_CHANNELS,
        "input_shape": [3, 64, 64],
        "normalization_statistics": "artifacts/preprocessing/scalograms/scalogram_metadata.json",
    }
    results["cnn2d"] = run_validation_selected_experiment(
        WESADScalogramCNN,
        datasets,
        validation_metadata,
        test_metadata,
        PROJECT_ROOT / "artifacts" / "models" / "cnn2d",
        {**common, "model_class": "WESADScalogramCNN", "architecture": "16-32-64"},
        device,
        compare_weighted_loss=True,
    )
else:
    print("Non exécuté : activer RUN_CNN2D_TRAINING pour les données réelles.")
"""),
        md("## Métriques de validation et de test"),
        code(r"""
cnn2d_artifact_dir = PROJECT_ROOT / "artifacts" / "models" / "cnn2d"
metric_files = {
    "Validation": cnn2d_artifact_dir / "validation_metrics.json",
    "Test final": cnn2d_artifact_dir / "test_metrics.json",
    "Résumé entraînement": cnn2d_artifact_dir / "training_summary.json",
}
if all(path.exists() for path in metric_files.values()):
    loaded_metrics = {}
    for section, path in metric_files.items():
        with path.open(encoding="utf-8") as handle:
            loaded_metrics[section] = json.load(handle)
    metric_names = ["macro_f1", "weighted_f1", "stress_precision", "stress_recall", "roc_auc", "average_precision"]
    metrics_table = pd.DataFrame({
        section: {name: values.get(name) for name in metric_names}
        for section, values in loaded_metrics.items()
        if section != "Résumé entraînement"
    })
    display(metrics_table)
    display(pd.Series(loaded_metrics["Résumé entraînement"], name="entraînement"))
    display(pd.read_csv(cnn2d_artifact_dir / "per_subject_metrics.csv"))
else:
    print("Métriques absentes : terminer d'abord la cellule d'entraînement CNN 2D.")
"""),
        md("## Sauvegarde par `state_dict` et contrôle du rechargement"),
        code(r"""
checkpoint_path = PROJECT_ROOT / "artifacts/models/cnn2d/best_model.pt"
if datasets is not None and checkpoint_path.exists():
    batch = next(iter(torch.utils.data.DataLoader(datasets[2], batch_size=4)))[0].to(device)
    trained = WESADScalogramCNN().to(device)
    trained.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    trained.eval()
    with torch.no_grad():
        logits_before = trained(batch)
    reloaded = WESADScalogramCNN().to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(batch)
    torch.testing.assert_close(logits_before, logits_after)
    print("Logits identiques après rechargement.")
"""),
    ])


def notebook_12() -> None:
    write("12_cnn2d_ablation_and_feature_maps.ipynb", [
        md(r"""
# 12 — Ablation CNN 2D et cartes d'activation

Chaque expérience modifie un seul facteur par rapport à la référence : padding, pas de la première convolution, type de pooling, capacité ou convolution ponctuelle 1 × 1. Tous les autres hyperparamètres restent ceux de `src.config`. Le classement des architectures repose exclusivement sur le macro-F1 de validation.
"""),
        code(SETUP),
        code(r"""
from src.models import WESADScalogramCNN
from src.scalograms import WESADScalogramDataset
from src.experiments import run_validation_selected_experiment
from src.visualization import extract_feature_maps

scalogram_dir = PROJECT_ROOT / "data/processed/scalograms"
metadata_dir = PROJECT_ROOT / "data/processed/metadata"
ABlations = {
    "baseline": dict(padding=1, conv_stride=1, pooling_type="max", filters=(16, 32, 64), use_pointwise_conv=False),
    "padding_0": dict(padding=0, conv_stride=1, pooling_type="max", filters=(16, 32, 64), use_pointwise_conv=False),
    "stride_2_first_conv": dict(padding=1, conv_stride=2, pooling_type="max", filters=(16, 32, 64), use_pointwise_conv=False),
    "avg_pool": dict(padding=1, conv_stride=1, pooling_type="avg", filters=(16, 32, 64), use_pointwise_conv=False),
    "filters_8_16_32": dict(padding=1, conv_stride=1, pooling_type="max", filters=(8, 16, 32), use_pointwise_conv=False),
    "pointwise_1x1": dict(padding=1, conv_stride=1, pooling_type="max", filters=(16, 32, 64), use_pointwise_conv=True),
}
pd.DataFrame(ABlations).T
"""),
        md(r"""
Le pas 2 est appliqué uniquement à la première convolution. La référence avec pas 1, les deux valeurs de padding, les deux poolings, les deux capacités et l'ajout 1 × 1 forment des comparaisons contrôlées.
"""),
        code(r"""
RUN_ABLATIONS = False
ablation_results = {}
if RUN_ABLATIONS:
    datasets = tuple(WESADScalogramDataset(
        scalogram_dir / f"X_{split}.pt",
        scalogram_dir / f"y_{split}.pt",
        metadata_dir / f"windows_{split}.csv",
    ) for split in ("train", "validation", "test"))
    validation_meta = pd.read_csv(metadata_dir / "windows_validation.csv")
    test_meta = pd.read_csv(metadata_dir / "windows_test.csv")
    for name, architecture in ABlations.items():
        factory = lambda architecture=architecture: WESADScalogramCNN(**architecture)
        ablation_results[name] = run_validation_selected_experiment(
            factory, datasets, validation_meta, test_meta,
            PROJECT_ROOT / "artifacts/models/cnn2d_ablations" / name,
            {
                "model_class": "WESADScalogramCNN", "architecture": architecture,
                "seed": RANDOM_SEED, "subject_split": SPLIT_SUBJECTS,
                "input_channels": SCALOGRAM_CHANNELS, "input_shape": [3, 64, 64],
                "normalization_statistics": "artifacts/preprocessing/scalograms/scalogram_metadata.json",
            }, device, compare_weighted_loss=False,
        )
else:
    print("Non exécuté : activer RUN_ABLATIONS après le notebook 10.")
"""),
        md("## Tableau d'ablation CNN 2D"),
        code(r"""
def artifact_row(name, path):
    with open(path / "model_config.json", encoding="utf-8") as f: config = json.load(f)
    with open(path / "validation_metrics.json", encoding="utf-8") as f: val = json.load(f)
    with open(path / "test_metrics.json", encoding="utf-8") as f: test = json.load(f)
    with open(path / "training_summary.json", encoding="utf-8") as f: summary = json.load(f)
    arch = config.get("architecture", {})
    return {
        "model": name, "input representation": config.get("input_shape", "raw signals"),
        "padding": arch.get("padding"), "stride": arch.get("conv_stride"),
        "pooling": arch.get("pooling_type"), "filters": arch.get("filters"),
        "1x1 convolution": arch.get("use_pointwise_conv"),
        "parameter count": config.get("parameter_count"), "best epoch": summary.get("best_epoch"),
        "validation macro F1": val.get("macro_f1"), "test macro F1": test.get("macro_f1"),
        "stress precision": test.get("stress_precision"), "stress recall": test.get("stress_recall"),
        "ROC-AUC": test.get("roc_auc"), "average precision": test.get("average_precision"),
        "training time": summary.get("training_time_seconds"),
        "test inference time": test.get("inference_time_seconds"),
    }

paths = {name: PROJECT_ROOT / "artifacts/models/cnn2d_ablations" / name for name in ABlations}
rows = [artifact_row(name, path) for name, path in paths.items() if (path / "model_config.json").exists()]
ablation_table = pd.DataFrame(rows)
display(ablation_table.sort_values("validation macro F1", ascending=False) if len(ablation_table) else "Aucun artefact d'ablation.")
"""),
        md(r"""
## Cartes d'activation

Les hooks sont retirés immédiatement après l'inférence. Les images ci-dessous sont des activations apprises, et non une preuve physiologique directe. Une formulation prudente est : « cette carte répond fortement à un motif temps-fréquence localisé ». Les réponses peuvent coïncider avec des structures BVP périodiques, des variations EDA lentes, une activité d'accélération rapide ou des bandes larges/étroites, sans autoriser une interprétation causale.
"""),
        code(r"""
feature_checkpoint = PROJECT_ROOT / "artifacts/models/cnn2d/best_model.pt"
RUN_FEATURE_MAPS = feature_checkpoint.exists()
if RUN_FEATURE_MAPS:
    dataset = WESADScalogramDataset(scalogram_dir / "X_validation.pt", scalogram_dir / "y_validation.pt", metadata_dir / "windows_validation.csv")
    example, label, metadata = dataset[0]
    model = WESADScalogramCNN().to(device)
    model.load_state_dict(torch.load(feature_checkpoint, map_location=device, weights_only=True))
    maps = extract_feature_maps(model, example[None], [model.features[0], model.features[8]])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, channel in enumerate(SCALOGRAM_CHANNELS):
        axes[i].imshow(example[i], aspect="auto", origin="lower", cmap="viridis"); axes[i].set_title(channel)
    plt.show()
    for title, activation in zip(["Première convolution", "Dernier bloc convolutif"], maps):
        count = min(8, activation.shape[1]); fig, axes = plt.subplots(2, 4, figsize=(12, 6));
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < count: ax.imshow(activation[0, i], aspect="auto", origin="lower", cmap="magma"); ax.set_title(f"Carte {i+1}")
        fig.suptitle(title); plt.show()
"""),
    ])


def notebook_13() -> None:
    write("13_rnn_bptt_gradient_clipping.ipynb", [
        md(r"""
# 13 — RNN, LSTM, BPTT et écrêtage du gradient

La tâche reste une classification séquentielle plusieurs-vers-un : 960 mesures successives et 6 canaux produisent une seule classe binaire. Toutes les fenêtres possèdent une longueur fixe de 960 pas temporels. Aucun padding ni masque n'est donc nécessaire dans cette expérience.
"""),
        code(SETUP),
        md(r"""
## Rétropropagation à travers le temps

Un RNN simple calcule

\[
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h),\qquad z=W_{hy}h_{960}+b_y.
\]

Le réseau est conceptuellement déplié sur 960 pas. BPTT applique la règle de dérivation en chaîne depuis la perte vers chaque état antérieur. Une contribution au gradient contient des produits répétés de Jacobiennes récurrentes,

\[
\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}.
\]

Si leurs valeurs singulières effectives restent inférieures à un, ce produit tend vers zéro (gradient évanescent) ; si elles sont durablement supérieures à un, il peut croître brutalement (gradient explosif). Les portes d'entrée, d'oubli et de sortie du LSTM, ainsi que son chemin de cellule mémoire, peuvent faciliter la conservation d'information sur une longue portée, sans garantir la généralisation.

L'écrêtage par norme remplace un gradient trop grand par une version remise à l'échelle. Il limite les mises à jour excessives et peut stabiliser l'optimisation. Il ne restaure pas un gradient déjà évanescent et ne résout ni le déséquilibre de classes ni la variabilité inter-sujets.
"""),
        md("## Inspection et comparaison équitable"),
        code(r"""
from src.models import LSTMClassifier, SimpleRNNClassifier
from src.training import compute_global_gradient_norm

HIDDEN_SIZE = 32
rnn = SimpleRNNClassifier(input_size=6, hidden_size=HIDDEN_SIZE).to(device)
lstm = LSTMClassifier(input_size=6, hidden_size=HIDDEN_SIZE).to(device)
inspection = pd.DataFrame([
    {"model": "RNN", "hidden size": rnn.rnn.hidden_size, "layers": rnn.rnn.num_layers, "bidirectional": rnn.rnn.bidirectional, "parameters": count_parameters(rnn)},
    {"model": "LSTM", "hidden size": lstm.lstm.hidden_size, "layers": lstm.lstm.num_layers, "bidirectional": lstm.lstm.bidirectional, "parameters": count_parameters(lstm)},
])
display(inspection)
toy = torch.randn(2, 960, 6, device=device)
assert rnn(toy).shape == lstm(toy).shape == (2, 1)
assert next(rnn.parameters()).device == toy.device
print("Périphérique modèle :", next(rnn.parameters()).device, "| lot :", toy.device)
"""),
        md(r"""
Les deux cœurs utilisent ici la même taille cachée, une couche, le même sens, les mêmes fenêtres normalisées, lots, optimiseur, taux, régularisation, patience, seuil de validation et graine. Le LSTM possède davantage de paramètres à cause de ses portes. Les sorties sont des logits bruts et la perte est `BCEWithLogitsLoss`.
"""),
        code(r"""
from torch.utils.data import TensorDataset
from src.experiments import run_validation_selected_experiment

sequence_dir = PROJECT_ROOT / "data/processed/sequence"
metadata_dir = PROJECT_ROOT / "data/processed/metadata"

def load_sequence_datasets():
    return tuple(TensorDataset(
        torch.load(sequence_dir / f"X_{split}.pt", map_location="cpu", weights_only=True),
        torch.load(sequence_dir / f"y_{split}.pt", map_location="cpu", weights_only=True),
    ) for split in ("train", "validation", "test"))

datasets = load_sequence_datasets() if (sequence_dir / "X_train.pt").exists() else None
validation_meta = pd.read_csv(metadata_dir / "windows_validation.csv") if datasets else None
test_meta = pd.read_csv(metadata_dir / "windows_test.csv") if datasets else None
"""),
        md("## RNN sans écrêtage contre RNN avec norme maximale 1,0"),
        code(r"""
RUN_CLIPPING_EXPERIMENT = False
clipping_results = {}
if RUN_CLIPPING_EXPERIMENT:
    common = {
        "model_class": "SimpleRNNClassifier", "architecture": {"input_size": 6, "hidden_size": HIDDEN_SIZE, "layers": 1, "bidirectional": False},
        "seed": RANDOM_SEED, "subject_split": SPLIT_SUBJECTS, "input_channels": SEQUENCE_CHANNELS,
        "input_shape": [960, 6], "normalization_statistics": "artifacts/preprocessing/sequence_scaler.joblib",
    }
    for name, clip in [("without_clipping", None), ("clip_norm_1", 1.0)]:
        try:
            clipping_results[name] = run_validation_selected_experiment(
                lambda: SimpleRNNClassifier(input_size=6, hidden_size=HIDDEN_SIZE), datasets,
                validation_meta, test_meta, PROJECT_ROOT / "artifacts/models/rnn_clipping" / name,
                common, device, compare_weighted_loss=False, gradient_clip=clip,
                record_gradient_norms=True, record_validation_macro_f1=True,
            )
        except (FloatingPointError, RuntimeError) as error:
            failure_dir = PROJECT_ROOT / "artifacts/models/rnn_clipping" / name
            failure_dir.mkdir(parents=True, exist_ok=True)
            with open(failure_dir / "numerical_failure.json", "w", encoding="utf-8") as handle:
                json.dump({"numerically_stable": False, "error": str(error)}, handle, indent=2)
            print(name, "arrêté honnêtement :", error)
else:
    print("Non exécuté : activer RUN_CLIPPING_EXPERIMENT pour l'expérience réelle.")
"""),
        md("## Tracés obligatoires"),
        code(r"""
figure_dir = PROJECT_ROOT / "reports/figures/rnn_gradient_clipping"
figure_dir.mkdir(parents=True, exist_ok=True)
history_paths = {
    "sans écrêtage": PROJECT_ROOT / "artifacts/models/rnn_clipping/without_clipping/training_history.csv",
    "norme max = 1.0": PROJECT_ROOT / "artifacts/models/rnn_clipping/clip_norm_1/training_history.csv",
}
histories = {name: pd.read_csv(path) for name, path in history_paths.items() if path.exists()}

def plot_metric(column, title, ylabel, filename):
    if not histories: return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frame in histories.items():
        if column in frame: ax.plot(frame.epoch, frame[column], label=name)
    ax.set(title=title, xlabel="Époque", ylabel=ylabel); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=150); plt.show()

plot_metric("train_loss", "Perte d'entraînement", "BCE", "training_loss.png")
plot_metric("validation_loss", "Perte de validation", "BCE", "validation_loss.png")
plot_metric("validation_macro_f1", "Macro-F1 de validation", "Macro-F1", "validation_macro_f1.png")
plot_metric("gradient_pre_max", "Norme pré-écrêtage maximale", "Norme L2", "gradient_pre_max.png")
plot_metric("gradient_pre_median", "Norme pré-écrêtage médiane", "Norme L2", "gradient_pre_median.png")
if "norme max = 1.0" in histories:
    clipped = histories["norme max = 1.0"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(clipped.epoch, clipped.gradient_pre_mean, label="avant"); ax.plot(clipped.epoch, clipped.gradient_post_mean, label="après")
    ax.set(title="Normes moyennes avant/après écrêtage", xlabel="Époque", ylabel="Norme L2"); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "pre_vs_post.png", dpi=150); plt.show()
"""),
        md("## RNN contre LSTM et rechargement du RNN final"),
        code(r"""
RUN_FAIR_RNN_LSTM = False
fair_results = {}
if RUN_FAIR_RNN_LSTM:
    common = {"seed": RANDOM_SEED, "subject_split": SPLIT_SUBJECTS, "input_channels": SEQUENCE_CHANNELS, "input_shape": [960, 6], "normalization_statistics": "artifacts/preprocessing/sequence_scaler.joblib"}
    for name, factory in {
        "rnn_fair": lambda: SimpleRNNClassifier(input_size=6, hidden_size=HIDDEN_SIZE),
        "lstm_fair": lambda: LSTMClassifier(input_size=6, hidden_size=HIDDEN_SIZE),
    }.items():
        fair_results[name] = run_validation_selected_experiment(factory, datasets, validation_meta, test_meta,
            PROJECT_ROOT / "artifacts/models" / name,
            {**common, "model_class": name, "architecture": {"hidden_size": HIDDEN_SIZE, "layers": 1, "bidirectional": False}},
            device, compare_weighted_loss=True)
    rows = []
    for name, result in fair_results.items():
        rows.append({"model": name, "parameters": count_parameters(result["model"]),
            "validation macro F1": result["validation_metrics"]["macro_f1"], "test macro F1": result["test_metrics"]["macro_f1"],
            "weighted F1": result["test_metrics"]["weighted_f1"], "stress precision": result["test_metrics"]["stress_precision"],
            "stress recall": result["test_metrics"]["stress_recall"], "ROC-AUC": result["test_metrics"]["roc_auc"],
            "average precision": result["test_metrics"]["average_precision"], "best epoch": result["training_summary"]["best_epoch"],
            "training time": result["training_summary"]["training_time_seconds"], "inference time": result["test_metrics"]["inference_time_seconds"]})
    display(pd.DataFrame(rows))

    final_rnn = fair_results["rnn_fair"]["model"].eval(); batch = datasets[2].tensors[0][:4].to(device)
    with torch.no_grad(): before = final_rnn(batch)
    reloaded = SimpleRNNClassifier(input_size=6, hidden_size=HIDDEN_SIZE).to(device)
    reloaded.load_state_dict(torch.load(PROJECT_ROOT / "artifacts/models/rnn_fair/best_model.pt", map_location=device, weights_only=True)); reloaded.eval()
    with torch.no_grad(): after = reloaded(batch)
    torch.testing.assert_close(before, after)
"""),
        md(r"""
## Analyse d'échec fondée sur les artefacts

Une prédiction quasi constante « non-stress » doit être vérifiée par la matrice de confusion, la distribution des classes prédites, précision/rappel stress, rappel par sujet et distributions de probabilités conditionnelles. Les 960 pas récurrents, l'évanescence du gradient, le déséquilibre, la variabilité des participants, les quinze participants indépendants, l'emploi du seul état final et le surapprentissage à des motifs propres aux sujets sont des hypothèses, pas des causes établies sans expérience dédiée.
"""),
        code(r"""
rnn_dir = PROJECT_ROOT / "artifacts/models/rnn_fair"
if not (rnn_dir / "test_predictions.csv").exists(): rnn_dir = PROJECT_ROOT / "artifacts/models/rnn"
if (rnn_dir / "test_predictions.csv").exists():
    predictions = pd.read_csv(rnn_dir / "test_predictions.csv")
    print("Distribution prédite :\n", predictions.predicted_label.value_counts(normalize=True).sort_index())
    print("Matrice de confusion :\n", pd.crosstab(predictions.true_label, predictions.predicted_label))
    per_subject = pd.read_csv(rnn_dir / "per_subject_metrics.csv")
    display(per_subject[["subject_id", "stress_recall"]])
    predictions.boxplot(column="stress_probability", by="true_label"); plt.suptitle(""); plt.title("Probabilité par vraie classe"); plt.show()
"""),
    ])


def notebook_14() -> None:
    write("14_model_comparison.ipynb", [
        md(r"""
# 14 — Comparaison globale mise à jour

Ce notebook ne réentraîne aucun modèle. Il charge les artefacts sauvegardés. Le « gagnant du protocole » est le modèle choisi selon le macro-F1 de validation. La « généralisation test observée » décrit seulement la performance mesurée après gel du modèle, de la perte et du seuil ; elle ne sert pas à réviser la sélection.
"""),
        code(SETUP),
        code(r"""
MODEL_DIRS = {
    "MLP statistique": "mlp",
    "CNN 2D": "cnn2d",
    "RNN": "rnn_fair" if (PROJECT_ROOT / "artifacts/models/rnn_fair").exists() else "rnn",
    "LSTM": "lstm_fair" if (PROJECT_ROOT / "artifacts/models/lstm_fair").exists() else "lstm",
}

def load_model_row(display_name, directory):
    path = PROJECT_ROOT / "artifacts/models" / directory
    required = ["model_config.json", "validation_metrics.json", "test_metrics.json", "training_summary.json"]
    if not all((path / name).exists() for name in required): return None
    values = []
    for name in required:
        with open(path / name, encoding="utf-8") as handle: values.append(json.load(handle))
    config, validation, test, summary = values
    return {"model": display_name, "parameter count": config.get("parameter_count"),
        "validation macro F1": validation.get("macro_f1"), "test macro F1": test.get("macro_f1"),
        "weighted F1": test.get("weighted_f1"), "stress precision": test.get("stress_precision"),
        "stress recall": test.get("stress_recall"), "ROC-AUC": test.get("roc_auc"),
        "average precision": test.get("average_precision"), "best epoch": summary.get("best_epoch"),
        "training time": summary.get("training_time_seconds"), "inference time": test.get("inference_time_seconds"),
        "artifact directory": directory}

rows = [row for name, directory in MODEL_DIRS.items() if (row := load_model_row(name, directory)) is not None]
comparison = pd.DataFrame(rows)
display(comparison)
"""),
        md("## Gagnant du protocole et généralisation test observée"),
        code(r"""
if len(comparison):
    protocol_winner = comparison.loc[comparison["validation macro F1"].idxmax()]
    print("Gagnant du protocole (validation) :", protocol_winner["model"])
    print("Sa généralisation test observée :", protocol_winner["test macro F1"])
    axes = comparison.set_index("model")[["validation macro F1", "test macro F1"]].plot.bar(figsize=(9, 4), rot=20)
    axes.set_ylabel("Macro-F1"); axes.set_title("Validation contre test (test non utilisé pour sélectionner)"); plt.tight_layout(); plt.show()
"""),
        md("## Coût et métriques de la classe stress"),
        code(r"""
if len(comparison):
    display(comparison[["model", "parameter count"]].sort_values("parameter count"))
    display(comparison[["model", "training time", "inference time"]])
    display(comparison[["model", "stress precision", "stress recall", "ROC-AUC", "average precision"]])
"""),
        md("## Résultats par sujet"),
        code(r"""
subject_frames = []
for _, row in comparison.iterrows():
    path = PROJECT_ROOT / "artifacts/models" / row["artifact directory"] / "per_subject_metrics.csv"
    if path.exists():
        frame = pd.read_csv(path); frame.insert(0, "model", row["model"]); subject_frames.append(frame)
per_subject_all = pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()
display(per_subject_all)
"""),
        md(r"""
## Synthèse conceptuelle

- **MLP** : représentation statistique globale ;
- **CNN 2D** : motifs temps-fréquence locaux dans les scalogrammes ;
- **RNN/LSTM** : dépendances temporelles ordonnées.

Une complexité ou un nombre de paramètres supérieur ne garantit pas une meilleure généralisation indépendante des sujets. WESAD ne fournit ici que quinze participants indépendants : les écarts entre validation et test peuvent refléter une forte sensibilité aux personnes composant chaque sous-ensemble. L'interprétation doit donc associer macro-F1, rappel stress et résultats par sujet, tout en conservant la distinction stricte entre choix sur validation et observation finale sur test.
"""),
    ])


if __name__ == "__main__":
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    notebook_10()
    notebook_11()
    notebook_12()
