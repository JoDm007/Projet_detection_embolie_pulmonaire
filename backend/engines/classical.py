# backend/engines/classical.py
# Moteur classique : exactement le pipeline validé dans docs/planning/plan1.md.
#   O2  ClassicalDetector (mode 'bright', paramètres lus dans config.yaml)
#   O3  EmbolismFeatureExtractor, sur les HU fenêtrées [-600, 400] (comme build_ml_dataset)
#   O4  scaler + régression logistique, seuil au point de fonctionnement 4 FP/patient
#
# Honnêteté de la démo (plan2 § 1.4) : pour un cas FUMPE, on applique le modèle du fold de
# CV où ce patient était en test, jamais un modèle qui l'a vu à l'entraînement.
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from src.O2_classical_detection.detector import ClassicalDetector
from src.O3_radiomics.feature_extractor import EmbolismFeatureExtractor

from .base import AnalysisResult, EngineUnavailable, Finding, ProgressFn

logger = logging.getLogger("backend.engines.classical")

# Même fenêtre que volume.npy / build_ml_dataset.denormalize_to_hu
FEATURE_HU_WINDOW = (-600.0, 400.0)


def detector_config(project_root: Path) -> dict:
    """Traduit la section classical_detection de config.yaml en paramètres de ClassicalDetector."""
    cfg = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    cd = cfg["classical_detection"]
    return {
        "lung_hu_threshold": cd["lung_segmentation"]["hu_threshold"],
        "lung_min_area": cd["lung_segmentation"]["min_area"],
        "detection_mode": cd.get("detection_mode", "bright"),
        "vessel_min_hu": cd.get("vessel_min_hu", 100),
        "max_hu": cd.get("max_hu", 400),
        "min_candidate_volume": cd.get("min_candidate_volume", 50),
        "max_candidate_volume": cd.get("max_candidate_volume"),
    }


def candidate_features(volume_feat: np.ndarray, labeled: np.ndarray, spacing,
                       progress: Optional[ProgressFn] = None,
                       progress_span: tuple[float, float] = (0.0, 1.0)) -> tuple[list[int], list[dict]]:
    """Features O3 de chaque candidat, calculées sur une boîte englobante (+1 voxel de marge).

    Résultat identique à un calcul sur le volume entier (toutes les features ne dépendent que
    des voxels du candidat et de sa boîte englobante — vérifié par backend/tests/test_engines.py),
    mais des centaines de fois plus rapide."""
    extractor = EmbolismFeatureExtractor()
    objects = ndimage.find_objects(labeled)
    labels, rows = [], []
    n = len(objects)
    for lbl, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        crop = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, dim))
                     for s, dim in zip(sl, labeled.shape))
        mask = (labeled[crop] == lbl).astype(np.uint8)
        feats = extractor.extract_from_arrays(volume_feat[crop], mask, spacing=spacing)
        if feats:
            labels.append(lbl)
            rows.append(feats)
        if progress and (lbl % 25 == 0 or lbl == n):
            a, b = progress_span
            progress(f"O3 — features radiomiques ({lbl}/{n} candidats)", a + (b - a) * lbl / max(n, 1))
    return labels, rows


class ClassicalEngine:
    name = "classique"
    label = "Classique — O2 détection + O3 radiomique + O4 régression logistique"
    description = ("Détecte les régions vasculaires opacifiées dans les poumons, décrit chaque "
                   "candidat par 20 features radiomiques et les trie par régression logistique.")

    def __init__(self, models_dir: Path, project_root: Path):
        self.models_dir = models_dir
        self.params = json.loads((models_dir / "params.json").read_text(encoding="utf-8"))
        self.det_config = detector_config(project_root)
        self.threshold = float(self.params["threshold"])
        self.final_model = joblib.load(models_dir / self.params["final_model"])
        self.final_scaler = joblib.load(models_dir / self.params["final_scaler"])

        self.fold_of: dict[str, int] = {}
        if self.params.get("fold_assignments"):
            df = pd.read_csv(models_dir / self.params["fold_assignments"])
            self.fold_of = dict(zip(df["PatientID"], df["fold"].astype(int)))
        self._fold_cache: dict[int, dict] = {}

    # ── choix du modèle ──
    def _model_for(self, case_id: Optional[str]):
        if case_id is None:
            return (self.final_scaler, self.final_model,
                    f"Régression logistique entraînée sur les {self.params['n_patients']} patients FUMPE")
        if not self.fold_of:
            raise EngineUnavailable(
                "Modèles par fold absents : impossible d'analyser un cas FUMPE sans réutiliser un "
                "modèle qui l'a vu à l'entraînement. Relancer train_classifier.py puis "
                "« python -m backend.cli install-models ».")
        fold = self.fold_of.get(case_id)
        if fold is None:
            raise EngineUnavailable(f"Le cas {case_id} n'apparaît pas dans la CV d'O4.")
        if fold not in self._fold_cache:
            self._fold_cache[fold] = joblib.load(
                self.models_dir / self.params["fold_model_pattern"].format(fold=fold))
        bundle = self._fold_cache[fold]
        return (bundle["scaler"], bundle["model"],
                f"Régression logistique du fold {fold + 1} (où {case_id} était en test)")

    def _feature_spacing(self, spacing):
        # Le dataset O4 actuel a été construit sans meta.json → spacing (1, 1, 1) dans
        # build_ml_dataset.load_spacing. Les features doivent être calculées pareil, sinon les
        # features de forme (en mm) changent d'échelle et le modèle reçoit autre chose qu'à
        # l'entraînement. Passer feature_spacing à "dicom" si le dataset est régénéré avec meta.json.
        if self.params.get("feature_spacing", "unit") == "dicom":
            return tuple(float(s) for s in spacing)
        return (1.0, 1.0, 1.0)

    # ── analyse ──
    def analyse(self, volume_hu, spacing, progress: ProgressFn, case_id=None) -> AnalysisResult:
        scaler, model, model_desc = self._model_for(case_id)

        progress("O2 — segmentation des poumons et détection des candidats", 0.10)
        det = ClassicalDetector(self.det_config).detect(volume_hu, spacing=tuple(spacing))
        labeled = det["candidates_labeled"]

        volume_feat = np.clip(volume_hu, *FEATURE_HU_WINDOW).astype(np.float32)
        labels, rows = candidate_features(volume_feat, labeled, self._feature_spacing(spacing),
                                          progress, (0.45, 0.85))

        progress("O4 — classification des candidats", 0.88)
        findings: list[Finding] = []
        overlay = np.zeros(labeled.shape, dtype=np.uint16)
        if rows:
            cols = list(scaler.feature_names_in_)
            X = (pd.DataFrame(rows).reindex(columns=cols)
                 .replace([np.inf, -np.inf], np.nan).fillna(0))
            Xs = pd.DataFrame(scaler.transform(X), columns=cols)
            proba = model.predict_proba(Xs)[:, 1]

            kept = sorted(((float(p), lbl) for p, lbl in zip(proba, labels) if p > self.threshold),
                          reverse=True)
            objects = ndimage.find_objects(labeled)
            voxel_mm3 = float(np.prod(spacing))
            width = labeled.shape[2]
            for idx, (score, lbl) in enumerate(kept, start=1):
                sl = objects[lbl - 1]
                region = labeled[sl] == lbl
                overlay[sl][region] = idx
                zyx = np.argwhere(region).mean(axis=0) + [s.start for s in sl]
                z, y, x = (int(round(c)) for c in zyx)
                n_vox = int(region.sum())
                findings.append(Finding(
                    id=idx, score=round(score, 4),
                    slice_range=(int(sl[0].start), int(sl[0].stop - 1)),
                    centroid_vox=(x, y, z),
                    centroid_mm=(round(x * spacing[2], 1), round(y * spacing[1], 1),
                                 round(z * spacing[0], 1)),
                    # DICOM : l'axe X des colonnes va vers la gauche du patient
                    side="droit" if x < width / 2 else "gauche",
                    n_voxels=n_vox, volume_mm3=round(n_vox * voxel_mm3, 1)))

        warnings = []
        if case_id is None:
            warnings.append(f"Scanner externe : le modèle n'a été validé que sur FUMPE "
                            f"({self.params['n_patients']} patients). Les performances sur "
                            f"d'autres scanners ne sont pas mesurées.")
        return AnalysisResult(
            engine=self.name, findings=findings, overlay=overlay,
            n_candidates=int(det["n_candidates"]),
            operating_point={k: self.params[k] for k in
                             ("model", "fp_budget", "threshold", "sens_patients_globale",
                              "sens_patients_O4") if k in self.params},
            model_used=model_desc, warnings=warnings)
