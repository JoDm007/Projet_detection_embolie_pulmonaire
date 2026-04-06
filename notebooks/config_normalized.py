"""
Configuration de détection pour données NORMALISÉES [0, 1]
À utiliser dans le notebook 04b_batch_extraction_vp_fp.ipynb
"""

# Configuration pour données normalisées (pas de HU)
config_normalized = {
    # Segmentation pulmonaire
    # Dans [0,1] avec fenêtre (-600,600): poumons ~0.0-0.2, air ~0.0
    'lung_hu_threshold': 0.15,  # Seuil pour isoler les poumons (équiv. ~-450 HU)
    'lung_min_area': 1000,
    
    # Seuillage pour détection des candidats
    # Vaisseaux contrastés : ~0.5-0.8 dans [0,1] (équiv. 50-400 HU)
    'threshold_method': 'percentile',
    'threshold_percentile': 92,  # Percentile plus haut pour données normalisées
    'min_hu': 0.35,  # Équiv. ~-150 HU → 0.35 en [0,1]
    'max_hu': 0.85,  # Équiv. ~400 HU → 0.85 en [0,1]
    
    # Filtrage morphologique
    'min_candidate_volume': 20,
    'max_candidate_volume': 5000
}

# Comment l'utiliser dans le notebook:
# config = config_normalized
# detector = ClassicalDetector(config)
