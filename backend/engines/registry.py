# backend/engines/registry.py
# Installation et chargement des moteurs.
#
# Sécurité (plan2 § 4.7) : un .pkl/joblib peut exécuter du code au chargement. On ne charge
# donc que des fichiers copiés dans app_data/models/ par « python -m backend.cli install-models »,
# dont l'empreinte SHA-256 est inscrite dans manifest.json et revérifiée à chaque démarrage.
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.core.security import sha256_file

from .unet import UNAVAILABLE_REASON as UNET_REASON

logger = logging.getLogger("backend.engines.registry")

CLASSICAL_DIR = "classical"
MANIFEST = "manifest.json"


@dataclass
class EngineSlot:
    name: str
    label: str
    engine: Optional[object]      # instance prête, ou None si indisponible
    reason: Optional[str] = None  # pourquoi il est indisponible


# ─────────────────────────────── Installation ───────────────────────────────
def _froc_threshold(oof: pd.DataFrame, proba_col: str, budget: int) -> float:
    """Même règle que froc_table() dans train_classifier.py : seuil strict au-dessus du
    k-ième plus fort score négatif, k = budget × nb de patients."""
    y = oof["has_embolism"].to_numpy()
    neg_sorted = np.sort(oof.loc[y == 0, proba_col].to_numpy())[::-1]
    k = int(budget * oof["PatientID"].nunique())
    return float(neg_sorted[k]) if k < len(neg_sorted) else float("-inf")


def install_classical_models(source_dir: Path, models_dir: Path, model_name: str = "logistic_regression",
                             fp_budget: int = 4, feature_spacing: str = "unit") -> dict:
    """Copie le modèle O4 retenu (final + folds) dans app_data/models/classical/,
    calcule le seuil au point de fonctionnement et écrit params.json + manifest.json."""
    src_models = source_dir / "models"
    dest = models_dir / CLASSICAL_DIR
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for f in (f"{model_name}.pkl", "scaler.pkl"):
        if not (src_models / f).exists():
            raise FileNotFoundError(f"{src_models / f} introuvable : lancer train_classifier.py.")
        shutil.copy2(src_models / f, dest / f)

    params = {
        "model": model_name,
        "final_model": f"{model_name}.pkl",
        "final_scaler": "scaler.pkl",
        "fp_budget": fp_budget,
        "feature_spacing": feature_spacing,
        "fold_assignments": None,
        "fold_model_pattern": None,
    }

    # Modèles par fold (produits par train_classifier.py depuis plan2 tâche 1.3)
    folds_src = src_models / "folds"
    fold_files = sorted(folds_src.glob(f"{model_name}_fold*.pkl")) if folds_src.exists() else []
    if fold_files and (folds_src / "fold_assignments.csv").exists():
        (dest / "folds").mkdir()
        for f in fold_files:
            shutil.copy2(f, dest / "folds" / f.name)
        shutil.copy2(folds_src / "fold_assignments.csv", dest / "folds" / "fold_assignments.csv")
        params["fold_assignments"] = "folds/fold_assignments.csv"
        params["fold_model_pattern"] = f"folds/{model_name}_fold{{fold}}.pkl"
    else:
        logger.warning("Pas de modèles par fold : les cas de démo FUMPE seront refusés.")

    # Point de fonctionnement, recalculé sur les prédictions hors-fold (non arrondies)
    oof = pd.read_csv(source_dir / "oof_predictions.csv")
    params["threshold"] = _froc_threshold(oof, f"proba_{model_name}", fp_budget)
    params["n_patients"] = int(oof["PatientID"].nunique())

    froc = pd.read_csv(source_dir / "froc.csv")
    row = froc[(froc["model"] == model_name) & (froc["FP_par_patient_max"] == fp_budget)]
    if not row.empty:
        params["sens_patients_O4"] = float(row["sens_patients_O4"].iloc[0])
        if "sens_patients_globale" in row:
            params["sens_patients_globale"] = float(row["sens_patients_globale"].iloc[0])

    (dest / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    manifest = {str(p.relative_to(dest)).replace("\\", "/"): sha256_file(p)
                for p in sorted(dest.rglob("*")) if p.is_file()}
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return params


# ─────────────────────────────── Vérification ───────────────────────────────
def verify_manifest(directory: Path) -> Optional[str]:
    """None si tous les fichiers listés existent avec la bonne empreinte, sinon le problème."""
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return "modèles non installés (lancer « python -m backend.cli install-models »)"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, digest in manifest.items():
        path = directory / rel
        if not path.is_file():
            return f"fichier manquant : {rel}"
        if sha256_file(path) != digest:
            return f"empreinte SHA-256 invalide pour {rel} (fichier modifié ?)"
    return None


# ─────────────────────────────── Chargement ───────────────────────────────
def load_engines(models_dir: Path, project_root: Path) -> dict[str, EngineSlot]:
    slots: dict[str, EngineSlot] = {}

    from .classical import ClassicalEngine
    classical_dir = models_dir / CLASSICAL_DIR
    problem = verify_manifest(classical_dir)
    if problem:
        logger.warning(f"Moteur classique indisponible : {problem}")
        slots[ClassicalEngine.name] = EngineSlot(ClassicalEngine.name, ClassicalEngine.label,
                                                 None, problem)
    else:
        engine = ClassicalEngine(classical_dir, project_root)
        slots[engine.name] = EngineSlot(engine.name, engine.label, engine)
        logger.info(f"Moteur classique chargé (seuil {engine.threshold:.4f})")

    slots["unet"] = EngineSlot("unet", "U-Net 3D — segmentation voxel par voxel (O5)", None, UNET_REASON)
    return slots
