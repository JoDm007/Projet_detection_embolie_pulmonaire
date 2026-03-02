# run_preprocessing.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script batch pour prétraiter tous les patients du dataset FUMPE.

Ce script applique :
1. Chargement DICOM + MASQUE (.mat)
2. Fenêtrage Hounsfield (fenêtre pulmonaire)
3. Resampling isotrope (1×1×1 mm)
4. Sauvegarde au format .npy (rapide pour le Deep Learning)

Usage:
    python scripts/run_preprocessing.py
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.O1_preprocessing.loader import get_available_patients, load_patient_data
from src.O1_preprocessing.normalisation import apply_windowing, HU_WINDOWS
from src.O1_preprocessing.resampling import standardize_spacing

# ═══════════════
# CONFIGURATION
# ═══════════════

# Configuration du logging (fichier + console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/preprocessing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Paramètres de prétraitement
CONFIG = {
    'hu_window': HU_WINDOWS['lung'],  # (-600, 600) pour embolies
    'target_spacing': (1.0, 1.0, 1.0),  # Isotropique 1mm³
    'normalize': True,  # Normaliser vers [0, 1]
    'data_dir': 'data',
    'output_dir': 'data/processed',
    'splits_dir': 'data/splits'
}


# ══════════════════════
# FONCTIONS PRINCIPALES
# ══════════════════════

def preprocess_patient(patient_id: str, 
                       inventory: Dict,
                       config: Dict) -> Tuple[bool, Optional[Dict]]:
    """
    Prétraite un patient unique.
    
    Returns:
        success: True si le traitement a réussi
        metadata: Dictionnaire avec les métadonnées du traitement
    """
    logger.info(f"🔄 Traitement de {patient_id}...")
    
    metadata = {
        'patient_id': patient_id,
        'status': 'pending',
        'error': None,
        'original_shape': None,
        'processed_shape': None,
        'original_spacing': None,
        'processed_spacing': None,
        'hu_window': config['hu_window'],
        'target_spacing': config['target_spacing']
    }
    
    try:
        # 1. Chargement des données brutes
        volume, mask, vol_meta = load_patient_data(patient_id, inventory)
        
        if volume is None or mask is None:
            raise ValueError("Volume ou masque non chargé")
        
        metadata['original_shape'] = list(volume.shape)
        metadata['original_spacing'] = {
            'xy': vol_meta['spacing'],
            'z': vol_meta['slice_spacing']
        }
        
        logger.info(f"   📊 Volume brut: {volume.shape}, HU: [{volume.min():.0f}, {volume.max():.0f}]")
        
        # 2. Fenêtrage Hounsfield
        window_min, window_max = config['hu_window']
        volume_windowed = apply_windowing(
            volume, 
            window_min, 
            window_max, 
            normalize=config['normalize']
        )
        
        logger.info(f"   🪟 Fenêtrage appliqué: [{window_min}, {window_max}] HU")
        
        # 3. Resampling isotrope
        current_spacing = (
            vol_meta['slice_spacing'],
            vol_meta['spacing'][0],
            vol_meta['spacing'][1]
        )
        
        volume_resampled, mask_resampled, new_spacing = standardize_spacing(
            volume_windowed,
            mask,
            current_spacing,
            config['target_spacing']
        )
        
        metadata['processed_shape'] = list(volume_resampled.shape)
        metadata['processed_spacing'] = list(new_spacing)
        
        logger.info(f"   📐 Resampling: {volume.shape} → {volume_resampled.shape}")
        
        # 4. Sauvegarde
        output_path = Path(config['output_dir']) / patient_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder volume et masque séparément (plus flexible)
        np.save(output_path / 'volume.npy', volume_resampled.astype(np.float32))
        np.save(output_path / 'mask.npy', mask_resampled.astype(np.uint8))
        
        # Sauvegarder les métadonnées du patient
        patient_meta = {
            'original_shape': metadata['original_shape'],
            'processed_shape': metadata['processed_shape'],
            'original_spacing': metadata['original_spacing'],
            'processed_spacing': metadata['processed_spacing'],
            'hu_window': metadata['hu_window'],
            'voxels_embolie': int(mask_resampled.sum()),
            'percentage_embolie': float(100 * mask_resampled.sum() / mask_resampled.size)
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(patient_meta, f, indent=2)
        
        logger.info(f"   ✅ Sauvegardé dans {output_path}")
        logger.info(f"   🎯 Embolie: {patient_meta['voxels_embolie']} voxels ({patient_meta['percentage_embolie']:.4f}%)")
        
        metadata['status'] = 'success'
        metadata['output_path'] = str(output_path)
        
        return True, metadata
        
    except Exception as e:
        logger.error(f"   ❌ Erreur pour {patient_id}: {e}")
        metadata['status'] = 'failed'
        metadata['error'] = str(e)
        return False, metadata


def run_preprocessing_pipeline(config: Dict) -> Dict:
    """
    Exécute le pipeline de prétraitement sur tous les patients.
    
    Returns:
        summary: Résumé statistique du traitement
    """
    logger.info("="*60)
    logger.info("🚀 DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT")
    logger.info("="*60)
    logger.info(f"📁 Dossier source: {config['data_dir']}")
    logger.info(f"📁 Dossier destination: {config['output_dir']}")
    logger.info(f"🪟 Fenêtre HU: {config['hu_window']}")
    logger.info(f"📐 Espacement cible: {config['target_spacing']}")
    logger.info("="*60)
    
    # 1. Récupérer l'inventaire des patients
    inventory = get_available_patients(config['data_dir'])
    
    if not inventory['with_both']:
        logger.error("❌ Aucun patient avec volume + masque trouvé !")
        return {'error': 'No patients found'}
    
    total_patients = len(inventory['with_both'])
    logger.info(f"📊 {total_patients} patients à traiter")
    logger.info("="*60)
    
    # 2. Créer les dossiers de sortie
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['splits_dir']).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    # 3. Traiter chaque patient
    results = []
    success_count = 0
    failed_count = 0
    failed_patients = []
    
    for i, patient_id in enumerate(inventory['with_both'], 1):
        logger.info(f"\n[{i}/{total_patients}] Patient {patient_id}")
        
        success, metadata = preprocess_patient(patient_id, inventory, config)
        results.append(metadata)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_patients.append(patient_id)
    
    # 4. Générer le résumé global
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'total_patients': total_patients,
        'success_count': success_count,
        'failed_count': failed_count,
        'success_rate': f"{100*success_count/total_patients:.1f}%",
        'failed_patients': failed_patients,
        'patient_details': results
    }
    
    # 5. Sauvegarder le résumé
    summary_path = Path(config['output_dir']) / 'preprocessing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 6. Afficher le résumé final
    logger.info("\n" + "="*60)
    logger.info("📊 RÉSUMÉ FINAL DU PRÉTRAITEMENT")
    logger.info("="*60)
    logger.info(f"✅ Succès: {success_count}/{total_patients} ({summary['success_rate']})")
    logger.info(f"❌ Échecs: {failed_count}/{total_patients}")
    
    if failed_patients:
        logger.warning(f"⚠️ Patients en échec: {failed_patients}")
    
    logger.info(f"📄 Résumé sauvegardé: {summary_path}")
    logger.info("="*60)
    
    return summary


def verify_processed_data(config: Dict) -> bool:
    """
    Vérifie l'intégrité des données prétraitées.
    """
    logger.info("\n🔍 Vérification des données prétraitées...")
    
    processed_dir = Path(config['output_dir'])
    patients_processed = [d.name for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('PAT')]
    
    if not patients_processed:
        logger.error("❌ Aucun patient trouvé dans le dossier processed !")
        return False
    
    errors = []
    
    for patient_id in patients_processed:
        patient_path = processed_dir / patient_id
        
        # Vérifier les fichiers requis
        required_files = ['volume.npy', 'mask.npy', 'metadata.json']
        for file in required_files:
            if not (patient_path / file).exists():
                errors.append(f"{patient_id}: {file} manquant")
        
        # Vérifier la cohérence des shapes
        try:
            vol = np.load(patient_path / 'volume.npy')
            mask = np.load(patient_path / 'mask.npy')
            
            if vol.shape != mask.shape:
                errors.append(f"{patient_id}: Shape mismatch vol={vol.shape}, mask={mask.shape}")
            
            if vol.dtype != np.float32:
                errors.append(f"{patient_id}: Volume dtype incorrect ({vol.dtype})")
            
            if mask.dtype != np.uint8:
                errors.append(f"{patient_id}: Mask dtype incorrect ({mask.dtype})")
                
        except Exception as e:
            errors.append(f"{patient_id}: Erreur de lecture ({e})")
    
    if errors:
        logger.error(f"❌ {len(errors)} erreurs détectées:")
        for err in errors[:10]:  # Afficher les 10 premières
            logger.error(f"   - {err}")
        return False
    else:
        logger.info(f"✅ {len(patients_processed)} patients vérifiés avec succès !")
        return True


# ═══════════════════
# POINT D'ENTRÉE
# ═══════════════════

if __name__ == "__main__":
    try:
        # 1. Lancer le pipeline
        summary = run_preprocessing_pipeline(CONFIG)
        
        # 2. Vérifier l'intégrité
        if summary.get('success_count', 0) > 0:
            is_valid = verify_processed_data(CONFIG)
            
            if is_valid:
                logger.info("\n🎉 PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS !")
                logger.info("📁 Les données sont prêtes pour l'Objectif 2 (Segmentation)")
            else:
                logger.warning("\n⚠️ PRÉTRAITEMENT TERMINÉ AVEC DES ERREURS")
                logger.warning("📋 Vérifiez le fichier preprocessing_summary.json")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        logger.error(f"\n❌ Erreur critique: {e}")
        raise