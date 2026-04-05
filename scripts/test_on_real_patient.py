# scripts/test_on_real_patient.py
#!/usr/bin/env python3
"""
Test du pipeline Objectif 2 sur un VRAI patient FUMPE
"""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from pathlib import Path
from src.O1_preprocessing.loader import get_available_patients, load_patient_data
from src.O2_classical_detection.detector import ClassicalDetector
from src.O2_classical_detection.connected_components import compute_overlap_with_gt

print("="*60)
print("🧪 TEST OBJECTIF 2 - VRAIES DONNÉES FUMPE")
print("="*60)

# 1. Charger l'inventaire
inventory = get_available_patients("data")
print(f"📊 {len(inventory['with_both'])} patients disponibles")

if not inventory['with_both']:
    print("❌ Aucun patient trouvé !")
    sys.exit(1)

# 2. Prendre le premier patient (ou un spécifique)
test_id = inventory['with_both'][0]  # Ex: PAT001
print(f"🧪 Test sur patient : {test_id}")

# 3. Charger les données RÉELLES
vol, mask_gt, meta = load_patient_data(test_id, inventory)

if vol is None or mask_gt is None:
    print("❌ Échec du chargement des données")
    sys.exit(1)

print(f"📦 Volume : {vol.shape}, HU: [{vol.min():.0f}, {vol.max():.0f}]")
print(f"🎯 Masque GT : {mask_gt.sum()} voxels d'embolie")

# 4. Lancer la détection classique
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
result = detector.detect(vol, spacing=(meta['slice_spacing'], *meta['spacing']))

# 5. ÉVALUATION : Comparer avec la vérité terrain
print("\n" + "="*60)
print("📊 ÉVALUATION DES PERFORMANCES")
print("="*60)

if len(result['candidates_df']) > 0:
    # Labelliser les candidats (positif/négatif par rapport au GT)
    result['candidates_df'] = compute_overlap_with_gt(
        result['candidates_df'],
        mask_gt,
        iou_threshold=0.5
    )
    
    n_positifs = result['candidates_df']['is_positive'].sum()
    n_negatifs = len(result['candidates_df']) - n_positifs
    
    print(f"✅ Candidats positifs (vraies embolies) : {n_positifs}")
    print(f"❌ Faux positifs : {n_negatifs}")
    print(f"📈 Précision : {100*n_positifs/len(result['candidates_df']):.1f}%")
    
    # Afficher les candidats
    print("\n📋 Détail des candidats :")
    display_cols = ['region_id', 'volume_voxels', 'intensity_mean', 'is_positive', 'iou_max']
    print(result['candidates_df'][display_cols].to_string())
else:
    print("⚠️ Aucun candidat détecté !")
    print("💡 Ajuster les seuils dans config")

print("\n✅ Test sur données RÉELLES terminé !")