# scripts/run_classical_pipeline.py
"""
Script de traitement batch pour la détection classique (Objectif 2).
Génère les candidats, les labellise (VP/FP) et SAUVEGARDE les masques .npy
nécessaires pour l'entraînement (Objectif 3/4/5).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Ajouter le dossier racine au path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.O2_classical_detection.detector import ClassicalDetector
from src.O2_classical_detection.connected_components import compute_overlap_with_gt

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    output_csv = processed_dir / 'radiomics_features_full.csv'
    
    # Configuration optimisée (HU Hounsfield natifs)
    config = {
        'lung_hu_threshold': -400,
        'threshold_method': 'percentile',
        'threshold_percentile': 95,
        'min_hu': 50,
        'max_hu': 400,
        'min_candidate_volume': 20,
        'max_candidate_volume': 5000
    }
    detector = ClassicalDetector(config)
    
    patient_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('PAT')])
    logger.info(f"🚀 Lancement de la détection sur {len(patient_dirs)} patients...")
    
    all_candidates = []
    
    for i, p_dir in enumerate(patient_dirs):
        patient_id = p_dir.name
        vol_path = p_dir / 'volume.npy'
        mask_path = p_dir / 'mask.npy'
        
        if not vol_path.exists() or not mask_path.exists():
            logger.warning(f"⚠️ [{i+1}/{len(patient_dirs)}] {patient_id} : fichiers .npy manquants.")
            continue
            
        try:
            volume = np.load(vol_path)
            gt_mask = np.load(mask_path)
            
            # 1. Détection
            result = detector.detect(volume)
            candidates_df = result['candidates_df']
            
            if len(candidates_df) == 0:
                logger.info(f"[{i+1}/{len(patient_dirs)}] {patient_id} : 0 candidat détecté.")
                # Sauvegarder quand même un masque vide pour la cohérence
                np.save(p_dir / 'candidates.npy', np.zeros_like(volume, dtype=np.uint8))
                continue
                
            # 2. Labellisation (VP/FP)
            candidates_df = compute_overlap_with_gt(candidates_df, gt_mask, iou_threshold=0.1)
            candidates_df['PatientID'] = patient_id
            
            n_vp = (candidates_df['is_positive'] == 1).sum()
            n_fp = (candidates_df['is_positive'] == 0).sum()
            logger.info(f"[{i+1}/{len(patient_dirs)}] ✅ {patient_id}: {len(candidates_df)} candidats ({n_vp} VP, {n_fp} FP)")
            
            # 3. SAUVEGARDE CRITIQUE DES MASQUES (pour O3 build_ml_dataset)
            np.save(p_dir / 'candidates.npy', result['candidates_labeled'].astype(np.uint16))
            np.save(p_dir / 'lung_mask.npy', result['lung_mask'].astype(np.uint8))
            
            all_candidates.append(candidates_df)
            
        except Exception as e:
            logger.error(f"❌ Erreur sur {patient_id}: {e}")
            
    # Assemblage final
    if all_candidates:
        df_final = pd.concat(all_candidates, ignore_index=True)
        df_final.rename(columns={'is_positive': 'has_embolism'}, inplace=True)
        df_final.to_csv(output_csv, index=False)
        
        logger.info("="*60)
        logger.info(f"🎉 EXTRACTION BATCH TERMINÉE !")
        logger.info(f"   Fichier : {output_csv}")
        logger.info(f"   Total candidats (FP) : {len(df_final)}")
        logger.info(f"   Note: Les VP seront injectés par src/O3_radiomics/build_ml_dataset.py")
        logger.info("="*60)
    else:
        logger.warning("🚫 Aucun candidat n'a été détecté sur l'ensemble des patients.")

if __name__ == "__main__":
    main()
