"""
Configuration CORRIGÉE pour détection d'embolies pulmonaires

Le problème : les embolies sont des défauts de remplissage (zones SOMBRES)
dans les vaisseaux contrastés, pas des zones CLAIRES.

Solution : seuillage pour détecter les zones de FAIBLE contraste
entre le poumon (-600 HU) et les vaisseaux contrastés (+50 HU)
"""

# Config CORRIGÉE pour données en HU
config_embolie_hu = {
    # Segmentation pulmonaire - garder comme avant
    'lung_hu_threshold': -400,  # Seuil standard poumons
    'lung_min_area': 1000,
    
    # Seuillage pour EMBOLIES (pas vaisseaux)
    # Détecter les zones de FAIBLE intensité dans les poumons
    'threshold_method': 'fixed',  # Seuil fixe plus adapté
    'threshold_percentile': 0,    # Pas utilisé avec 'fixed'
    'min_hu': -100,   # Intensité minimale (embolie = faible contraste)
    'max_hu': 50,     # Intensité maximale (exclure vaisseaux contrastés >50 HU)
    
    # OU alternative avec percentile (détecter les 5% les plus sombres dans poumons)
    # 'threshold_method': 'percentile',
    # 'threshold_percentile': 5,  # 5% les plus sombres
    # 'min_hu': -200,
    # 'max_hu': 50,
    
    # Filtrage morphologique
    'min_candidate_volume': 10,   # Embolies = petites
    'max_candidate_volume': 500
}

# Config pour données normalisées [0,1]
# Avec fenêtre (-600, 600): 
# -600 HU → 0.0, 0 HU → 0.5, +600 HU → 1.0
config_embolie_normalized = {
    'lung_hu_threshold': 0.17,  # -400 HU → (-400+600)/1200 = 0.17
    'threshold_method': 'fixed',
    'min_hu': 0.42,   # -100 HU → 0.42
    'max_hu': 0.54,   # +50 HU → 0.54
    'min_candidate_volume': 10,
    'max_candidate_volume': 500
}

"""
EXPLICATION:
- Ancienne config : min_hu=30, max_hu=400 → détectait les vaisseaux contrastés
- Nouvelle config : min_hu=-100, max_hu=50 → détecte les EMBOLIES (faible contraste)

Les embolies apparaissent comme des zones "sombres" dans les vaisseaux,
donc intensité PLUS BASSE que le sang contrasté.
"""
