# scripts/test_objectif2.py
#!/usr/bin/env python3
"""
Script de test pour l'Objectif 2 - Détection Classique
"""
import sys
import os
from pathlib import Path

# Ajouter la racine du projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🧪 TEST OBJECTIF 2 - DÉTECTION CLASSIQUE")
print("="*60)

# Tester les imports
try:
    from src.O2_classical_detection.detector import ClassicalDetector
    from src.O2_classical_detection.lung_segmentation import segment_lungs
    from src.O2_classical_detection.thresholding import adaptive_threshold
    from src.O2_classical_detection.morphology import clean_candidates
    from src.O2_classical_detection.connected_components import extract_candidates_3d
    print("✅ Tous les imports réussis !")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

# Tester le détecteur avec des données factices
import numpy as np

print("\n📦 Création de données de test...")
test_volume = np.random.randn(50, 100, 100) * 100 + 0
test_spacing = (1.0, 0.7, 0.7)

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
result = detector.detect(test_volume, spacing=test_spacing)

print(f"\n🎯 Résultats :")
print(f"   • Voxels pulmonaires : {result['lung_mask'].sum()}")
print(f"   • Candidats détectés : {result['n_candidates']}")
print(f"   • Voxels candidats : {result['total_candidate_voxels']}")

if len(result['candidates_df']) > 0:
    print(f"\n📋 Top candidats :")
    print(result['candidates_df'][['region_id', 'volume_voxels', 'intensity_mean']].head())

print("\n✅ Test Objectif 2 terminé !")