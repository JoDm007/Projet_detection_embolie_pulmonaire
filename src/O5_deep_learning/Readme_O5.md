# Module O5 — Deep Learning nnU-Net-style (PyTorch)

## Pipeline actif : `nnunet_pe.py`

Fichier unique auto-suffisant contenant le pipeline complet de segmentation 3D de l'embolie pulmonaire.

---

## Pourquoi nnU-Net-style ?

| Critère | `unet_3d_keras.py` (archivé) | `nnunet_pe.py` (actif) |
|---|---|---|
| Framework | TensorFlow / Keras | **PyTorch** |
| Normalisation | BatchNorm (instable batch=1-2) | **InstanceNorm3d** (stable à batch=1) |
| Preprocessing | Aucun (fixe) | **Adaptatif** (percentile 0.5/99.5, patch size calculé) |
| Loss | Cross-Entropy seule | **Dice + BCE combinés** |
| Inférence | Resize brutal | **Sliding window gaussien, overlap 50 %** |
| Augmentation 3D | Basique (flip, bruit) | **Complète** (élastique, gamma, mirroring, rotation) |

---

## Classes disponibles

### `NNUNetPreprocessor`
Calcule les statistiques d'intensité du dataset (percentile 0.5/99.5, moyenne, écart-type)
et détermine le patch size optimal selon la VRAM disponible.

```python
from src.O5_deep_learning.nnunet_pe import NNUNetPreprocessor

prep = NNUNetPreprocessor(processed_dir="data/processed", vram_gb=8.0)
stats = prep.analyse_dataset()          # calcule les stats
prep.save_stats("results/deep_learning/preprocessor_stats.json")

vol_norm = prep.normalize(volume_npy)   # clip percentile + z-score
```

### `NNUNetModel`
U-Net 3D PyTorch : 5 niveaux d'encodeur/décodeur, InstanceNorm3d, LeakyReLU, sortie sigmoid.

```python
from src.O5_deep_learning.nnunet_pe import NNUNetModel

model = NNUNetModel(in_channels=1, features=32, depth=5, dropout=0.2)
print(f"{model.count_parameters():,} paramètres")
```

### `NNUNetDataset`
Dataset PyTorch patch-based avec oversampling des voxels positifs et augmentation 3D complète.

```python
from src.O5_deep_learning.nnunet_pe import NNUNetDataset

ds = NNUNetDataset(
    patient_ids=["PAT001", "PAT002"],
    preprocessor=prep,
    processed_dir="data/processed",
    patch_size=(64, 128, 128),
    augment=True,
    patches_per_vol=50,
)
```

### `NNUNetTrainer`
Entraînement CV 5-folds groupée par patient (réutilise `fold_assignments.csv` d'O4).
Early stopping sur Dice val, checkpoint par fold dans `results/deep_learning/`.

```python
from src.O5_deep_learning.nnunet_pe import NNUNetTrainer

trainer = NNUNetTrainer(
    processed_dir="data/processed",
    folds_csv="results/classical_classifier/models/folds/fold_assignments.csv",
    output_dir="results/deep_learning",
    batch_size=2,       # ajuster selon la VRAM
    epochs=200,
    patience=30,
)
results = trainer.train_all_folds()
```

### `NNUNetPredictor`
Inférence sliding window gaussien. Utilise le modèle du fold dont le patient était en validation
(démo honnête).

```python
from src.O5_deep_learning.nnunet_pe import NNUNetPredictor

predictor = NNUNetPredictor(output_dir="results/deep_learning")
result = predictor.predict_patient("PAT001", processed_dir="data/processed")

print(f"Volume thrombus : {result['volume_mm3']:.1f} mm³")
print(f"Dice vs GT      : {result['dice_score']:.4f}")

# Évaluation complète sur tous les patients (out-of-fold)
df = predictor.evaluate_all_folds()
```

---

## Usage CLI

```powershell
# Smoke test CPU (quelques secondes, sans données réelles)
python -m src.O5_deep_learning.nnunet_pe test

# Entraînement complet 5-folds (nécessite data/processed/ peuplé)
python -m src.O5_deep_learning.nnunet_pe train

# Prédiction sur un patient
python -m src.O5_deep_learning.nnunet_pe predict PAT001
```

---

## Installation de PyTorch

PyTorch n'est pas dans `requirements.txt` (trop lourd, dépend du hardware).

```powershell
# CPU-only (compatible avec toutes les machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Sorties de l'entraînement (`results/deep_learning/`)

| Fichier | Contenu |
|---|---|
| `preprocessor_stats.json` | Statistiques HU du dataset (percentiles, mean, std, patch_size) |
| `fold_X_best.pth` | Checkpoint du meilleur modèle pour le fold X |
| `fold_X_history.json` | Courbes loss train/val et Dice val par époque |
| `cv_summary.json` | Résumé des 5 folds (Dice val moyen, epochs) |
| `oof_predictions_dl.csv` | Prédictions out-of-fold (Dice, volume mm³, détection) |

---

## Métriques d'évaluation

Deux métriques primaires (invariant de comparaison avec O4) :

| Métrique | Description | Cible |
|---|---|---|
| **Dice Score voxel** | Chevauchement GT ↔ prédiction | ≥ 0.30 sur les porteurs |
| **Sensibilité patient @ 4 FP/patient** | Patients détectés (même protocole O4) | ≥ 60 % (vs 45 % pour O4) |

---

## Fichiers archivés (`*_keras.py`)

Les fichiers suivants sont des archives de l'implémentation TensorFlow/Keras initiale.
Ils ne font plus partie du pipeline actif mais sont conservés pour référence.

- `unet_3d_keras.py` — U-Net 3D Keras (BatchNorm, sigmoid)
- `cnn_3d_keras.py` — CNN 3D Keras pour classification de patchs
- `cnn_2d_keras.py` — CNN 2D Keras slice-by-slice
- `data_generator_keras.py` — Générateur Keras (`tf.keras.utils.Sequence`)
- `train_keras.py` — Entraîneur Keras
- `predict_keras.py` — Prédicteur Keras

---

*Dernière mise à jour : 12 novembre 2026*
