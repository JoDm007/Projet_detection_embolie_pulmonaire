# Auteur: Joseph DATE-MASSE
# Objectif: Fusionner les Vrais Positifs (GT) et Faux Positifs (O2) pour O4
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from scipy.ndimage import center_of_mass
from .feature_extractor import EmbolismFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _find_overlap(gt_mask, cand_mask, tol_voxels=5):
    """Vérifie si un candidat O2 touche ou est proche du GT (tolérance en voxels)"""
    gt_bin = (gt_mask > 0).astype(np.uint8)
    cand_bin = (cand_mask > 0).astype(np.uint8)
    intersection = np.logical_and(gt_bin, cand_bin).sum()
    if intersection > 0:
        return True
    # Fallback: distance entre centroïdes
    try:
        gt_c = center_of_mass(gt_bin)
        cand_c = center_of_mass(cand_bin)
        dist = np.sqrt(np.sum((np.array(gt_c) - np.array(cand_c))**2))
        return dist <= tol_voxels
    except:
        return False

def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    output_csv = processed_dir / "ml_dataset_balanced.csv"
    extractor = EmbolismFeatureExtractor()
    
    if not processed_dir.exists():
        logger.error("❌ Dossier data/processed introuvable. Lance O1 et O2 d'abord.")
        return

    patient_dirs = sorted([d for d in processed_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("PAT")])
    logger.info(f"🚀 Construction dataset ML sur {len(patient_dirs)} patients...")

    all_rows = []

    for p_dir in patient_dirs:
        pid = p_dir.name
        vol_path = p_dir / "volume.npy"
        gt_path = p_dir / "mask.npy"          # GT = Vrais Positifs
        cand_path = p_dir / "candidates.npy"  # Sortie O2 = Masque étiqueté des candidats
        
        if not vol_path.exists() or not gt_path.exists():
            logger.warning(f"⚠️ Fichiers manquants pour {pid}")
            continue

        volume = np.load(vol_path)
        gt_mask = np.load(gt_path)
        
        # 1️⃣ Extraire les VP depuis le GT (1 par patient, même si plusieurs caillots, on les regroupe)
        if gt_mask.sum() > 0:
            feats = extractor.extract_from_arrays(volume, gt_mask, spacing=(1.0, 0.7, 0.7))
            if feats:
                feats.update({'PatientID': pid, 'candidate_id': f"{pid}_GT", 'has_embolism': 1})
                all_rows.append(feats)
        
        # 2️⃣ Extraire les FP depuis les candidats O2
        if cand_path.exists():
            cand_mask = np.load(cand_path)
            labels = np.unique(cand_mask[cand_mask > 0])
            for lbl in labels:
                mask_cand = (cand_mask == lbl).astype(np.uint8)
                
                # Vérifier si ce candidat overlap le GT
                is_vp = _find_overlap(gt_mask, mask_cand, tol_voxels=5)
                label = 1 if is_vp else 0
                
                feats = extractor.extract_from_arrays(volume, mask_cand, spacing=(1.0, 0.7, 0.7))
                if feats:
                    feats.update({
                        'PatientID': pid,
                        'candidate_id': f"{pid}_O2_{lbl}",
                        'has_embolism': label
                    })
                    all_rows.append(feats)

    if not all_rows:
        logger.error("❌ Aucune feature extraite. Vérifie les chemins des .npy")
        return

    df = pd.DataFrame(all_rows)
    cols_order = ['PatientID', 'candidate_id', 'has_embolism'] + \
                 [c for c in df.columns if c not in ['PatientID', 'candidate_id', 'has_embolism']]
    df = df[cols_order]
    df.to_csv(output_csv, index=False)
    
    vp_count = df['has_embolism'].sum()
    fp_count = len(df) - vp_count
    logger.info(f"🎉 DATASET GÉNÉRÉ : {output_csv}")
    logger.info(f"   📊 Total: {len(df)} | VP: {vp_count} | FP: {fp_count} | Ratio: {vp_count/len(df):.2%}")

if __name__ == "__main__":
    main()