# Auteur: Joseph DATE-MASSE
# Objectif: Construire le dataset ML VP/FP pour O4 à partir des candidats O2 et du GT.
#
# Phase 2 (2026-10) :
#   - PLUS d'injection du GT comme positif : chaque ligne est un candidat O2 réel
#     (sinon le classifieur apprenait "région dessinée par le radiologue" vs "vaisseau",
#      ce qui ne correspond pas à la tâche de déploiement — cf. docs/planning/plan1.md).
#   - Étiquetage par RECOUVREMENT : un candidat est VP s'il touche le GT.
#   - Features calculées sur les HU (et non sur le volume normalisé [0,1]).
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from .feature_extractor import EmbolismFeatureExtractor
except (ImportError, ValueError):
    from feature_extractor import EmbolismFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# volume.npy = clip(HU, -600, 400) puis (x + 600) / 1000. On inverse pour retrouver les HU.
HU_MIN, HU_MAX = -600.0, 400.0


def denormalize_to_hu(volume_norm: np.ndarray) -> np.ndarray:
    """Reconvertit le volume normalisé [0,1] en HU (fenêtre [-600, 400])."""
    return volume_norm.astype(np.float32) * (HU_MAX - HU_MIN) + HU_MIN


def is_true_positive(gt_mask: np.ndarray, cand_mask: np.ndarray) -> bool:
    """
    VP si le candidat recouvre le GT (intersection > 0) — critère 'hit' standard en CAD.
    On n'exige PAS un IoU élevé : le GT FUMPE annote tout le lumen, bien plus gros que le
    candidat. On maximise ainsi la classe positive, qui reste rare (~130 VP).
    """
    return int(np.logical_and(gt_mask > 0, cand_mask > 0).sum()) > 0


def load_spacing(p_dir: Path):
    """Spacing (z, y, x) en mm depuis meta.json ; (1,1,1) par défaut si absent."""
    meta_f = p_dir / "meta.json"
    if meta_f.exists():
        try:
            m = json.loads(meta_f.read_text(encoding="utf-8"))
            ps = m.get("spacing", [1.0, 1.0])
            return (float(m.get("slice_spacing", 1.0)), float(ps[0]), float(ps[1]))
        except Exception:
            pass
    return (1.0, 1.0, 1.0)


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    output_csv = processed_dir / "ml_dataset_balanced.csv"
    extractor = EmbolismFeatureExtractor()

    if not processed_dir.exists():
        logger.error("❌ Dossier data/processed introuvable. Lance O1 et O2 d'abord.")
        return

    patient_dirs = sorted([d for d in processed_dir.iterdir()
                           if d.is_dir() and d.name.startswith("PAT")])
    logger.info(f"🚀 Dataset ML : candidats O2 étiquetés par recouvrement GT, sur "
                f"{len(patient_dirs)} patients...")

    all_rows = []
    porteurs = set()          # patients avec GT non vide
    porteurs_avec_vp = set()  # patients avec >=1 candidat VP

    for p_dir in patient_dirs:
        pid = p_dir.name
        vol_path = p_dir / "volume.npy"
        gt_path = p_dir / "mask.npy"
        cand_path = p_dir / "candidates.npz"

        if not (vol_path.exists() and gt_path.exists() and cand_path.exists()):
            logger.warning(f"⚠️ {pid} : volume/mask/candidates manquant(s), ignoré")
            continue

        volume_hu = denormalize_to_hu(np.load(vol_path))
        gt_mask = np.load(gt_path) > 0
        cand_mask = np.load(cand_path)["labels"]
        spacing = load_spacing(p_dir)
        if gt_mask.any():
            porteurs.add(pid)

        for lbl in np.unique(cand_mask[cand_mask > 0]):
            mask_cand = (cand_mask == lbl).astype(np.uint8)
            label = 1 if is_true_positive(gt_mask, mask_cand) else 0
            feats = extractor.extract_from_arrays(volume_hu, mask_cand, spacing=spacing)
            if feats:
                feats.update({'PatientID': pid,
                              'candidate_id': f"{pid}_O2_{int(lbl)}",
                              'has_embolism': label})
                all_rows.append(feats)
                if label == 1:
                    porteurs_avec_vp.add(pid)

    if not all_rows:
        logger.error("❌ Aucune feature extraite. Vérifie que O2 a généré les candidates.npz.")
        return

    df = pd.DataFrame(all_rows)
    cols = ['PatientID', 'candidate_id', 'has_embolism']
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(output_csv, index=False)

    vp = int(df['has_embolism'].sum())
    fp = len(df) - vp
    logger.info(f"🎉 DATASET GÉNÉRÉ : {output_csv}")
    logger.info(f"   📊 Candidats: {len(df)} | VP: {vp} | FP: {fp} | %VP: {vp / len(df):.2%}")
    logger.info(f"   📊 Porteurs avec >=1 VP : {len(porteurs_avec_vp)}/{len(porteurs)} "
                f"= plafond de sensibilité atteignable par O4 ({100*len(porteurs_avec_vp)/max(len(porteurs),1):.0f}%)")
    logger.warning("   ⚠️ Aucun GT injecté : les métriques O4 sont à lire avec ce plafond en tête.")


if __name__ == "__main__":
    main()
