# config_hu.py

# Configuration pour le Détecteur Classique (Unités Hounsfield vraies)
# Puisque les volumes ne sont plus normalisés entre [0, 1],
# nous ajustons les seuils physiologiques (poumons, vaisseaux, embolies).

config_hu = {
    'lung_hu_threshold': -400,        # Seuil pour séparer l'air (poumons) des tissus mous. ~-400 HU
    'threshold_method': 'percentile', # Méthode de binarisation après isolation des poumons
    'threshold_percentile': 90,       # Garder les ~10% de pixels les plus brillants dans les poumons (vaisseaux + embolies potentiels)
    'min_hu': 40,                     # Valeur minimale acceptable pour une embolie/vaisseau (en HU). Un caillot est souvent entre 40 et 80 HU.
    'max_hu': 400,                    # Valeur maximale (au-delà, on est probablement sur de l'os ou un produit de contraste hyperdense, pas un caillot naturel).
    'min_candidate_volume': 10,       # Volume minimum (nombre de voxels) pour un candidat
    'max_candidate_volume': 2000,     # Volume maximum (au délà, il s'agit probablement de structures vasculaires majeures mal segmentées)
    'use_advanced_morphology': True,  # Activer le nettoyage des artefacts
    'margin_roi': (2, 2, 2)           # Marge autour des poumons pour l'extraction de la ROI
}
