# src/2_classical_detection/thresholding.py

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
import logging
from typing import Tuple, Optional, Union
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adaptive_threshold(volume: np.ndarray,
                      lung_mask: np.ndarray,
                      method: str = 'percentile',
                      percentile: float = 95,
                      min_hu: float = 50,
                      max_hu: float = 400,
                      sigma: float = 1.0,
                      min_size: int = 5,
                      use_multi_scale: bool = False) -> np.ndarray:
    """
    Applique un seuillage adaptatif pour détecter les régions candidates.
    
    Args:
        volume: Volume 3D en HU (déjà fenêtré)
        lung_mask: Masque binaire des poumons
        method: 'percentile', 'otsu', 'fixed', 'local', 'multi_otsu'
        percentile: Percentile pour le seuil (ex: 95 = top 5%)
        min_hu: Valeur HU minimale pour une embolie candidate
        max_hu: Valeur HU maximale (pour exclure les os)
        sigma: Lissage gaussien optionnel
        min_size: Taille minimale des régions (en voxels)
        use_multi_scale: Utiliser une approche multi-échelle
    
    Returns:
        candidates: Masque binaire des régions candidates
    """
    if volume is None or lung_mask is None:
        logger.error("Volume ou masque pulmonaire manquant")
        return np.zeros_like(volume, dtype=bool)
    
    logger.info(f"🔍 Seuillage adaptatif (méthode: {method})")
    
    # Option : lissage préalable
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Optimisation : n'appliquer le filtre que sur la bounding box des poumons
        coords = np.where(lung_mask)
        if len(coords[0]) > 0:
            z0, z1 = coords[0].min(), coords[0].max() + 1
            y0, y1 = coords[1].min(), coords[1].max() + 1
            x0, x1 = coords[2].min(), coords[2].max() + 1
            
            volume_smooth = volume.copy()
            roi = volume[z0:z1, y0:y1, x0:x1]
            volume_smooth[z0:z1, y0:y1, x0:x1] = gaussian_filter(roi, sigma=sigma)
            logger.info(f"   Lissage gaussien (sigma={sigma}) appliqué sur ROI de taille {roi.shape}")
        else:
            volume_smooth = volume
    else:
        volume_smooth = volume
    
    # Travailler uniquement dans la région pulmonaire
    lung_values = volume_smooth[lung_mask]
    
    if len(lung_values) == 0:
        logger.warning("⚠️ Aucune valeur dans le masque pulmonaire")
        return np.zeros_like(volume, dtype=bool)
    
    # Déterminer le seuil selon la méthode
    threshold = _compute_threshold(lung_values, method, percentile, min_hu)
    
    if method == 'multi_otsu':
        # Seuillage multi-niveaux
        candidates = _multi_otsu_threshold(volume_smooth, lung_mask, min_hu, max_hu)
    else:
        # Seuillage simple
        candidates = _apply_single_threshold(volume_smooth, lung_mask, 
                                            threshold, min_hu, max_hu)
    
    # Si use_multi_scale, combiner plusieurs échelles
    if use_multi_scale and method != 'multi_otsu':
        candidates_multi = _multi_scale_threshold(volume, lung_mask, 
                                                 min_hu, max_hu)
        candidates = np.logical_or(candidates, candidates_multi)
        logger.info(f"   Approche multi-échelle ajoutée")
    
    # Nettoyage morphologique
    candidates = _clean_candidates(candidates, min_size)
    
    # Statistiques
    n_voxels = candidates.sum()
    if n_voxels > 0:
        # Trouver les composantes connexes
        labeled, n_regions = ndimage.label(candidates)
        logger.info(f"✅ {n_voxels} voxels candidats détectés "
                    f"({100*n_voxels/candidates.size:.3f}% du volume)")
        logger.info(f"   {n_regions} régions candidates identifiées")
        
        # Analyse des tailles
        sizes = ndimage.sum(candidates, labeled, range(1, n_regions+1))
        logger.info(f"   Tailles: min={sizes.min():.0f}, "
                    f"moy={sizes.mean():.0f}, max={sizes.max():.0f}")
    else:
        logger.warning("⚠️ Aucun candidat détecté")
    
    return candidates

def _compute_threshold(lung_values: np.ndarray,
                      method: str,
                      percentile: float,
                      min_hu: float) -> float:
    """Calcule le seuil selon la méthode choisie"""
    
    if method == 'fixed':
        threshold = min_hu
        
    elif method == 'otsu':
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(lung_values)
        logger.info(f"   Seuil Otsu: {threshold:.1f} HU")
        
    elif method == 'local':
        # Pour le local, on retourne juste min_hu car le calcul est différent
        return min_hu
        
    elif method == 'percentile':
        threshold = np.percentile(lung_values, percentile)
        logger.info(f"   Seuil percentile {percentile}: {threshold:.1f} HU")
        
    elif method == 'median_ratio':
        # Seuil basé sur un ratio de la médiane
        median_val = np.median(lung_values)
        threshold = median_val * percentile / 100
        logger.info(f"   Seuil ratio médian ({percentile}%): {threshold:.1f} HU")
        
    else:
        logger.warning(f"Méthode '{method}' inconnue, utilisation percentile")
        threshold = np.percentile(lung_values, percentile)
    
    return max(threshold, min_hu)

def _apply_single_threshold(volume: np.ndarray,
                           lung_mask: np.ndarray,
                           threshold: float,
                           min_hu: float,
                           max_hu: float) -> np.ndarray:
    """Applique un seuillage simple"""
    
    candidates = np.zeros_like(volume, dtype=bool)
    
    # Appliquer les conditions
    in_lung = lung_mask
    intensity_ok = (volume >= threshold) & (volume <= max_hu)
    
    candidates[in_lung & intensity_ok] = True
    
    logger.info(f"   Seuil appliqué: {threshold:.1f} - {max_hu} HU")
    
    return candidates

def _multi_otsu_threshold(volume: np.ndarray,
                         lung_mask: np.ndarray,
                         min_hu: float,
                         max_hu: float,
                         n_classes: int = 3) -> np.ndarray:
    """Seuillage multi-niveaux avec Otsu"""
    from skimage.filters import threshold_multiotsu
    
    try:
        # Appliquer multi-Otsu sur les valeurs pulmonaires
        lung_values = volume[lung_mask]
        thresholds = threshold_multiotsu(lung_values, classes=n_classes)
        
        logger.info(f"   Seuils multi-Otsu ({n_classes} classes): "
                    f"{thresholds}")
        
        # Prendre la classe la plus haute (embolies supposées les plus denses)
        highest_class = thresholds[-1]
        
        candidates = np.zeros_like(volume, dtype=bool)
        candidates[lung_mask] = (volume[lung_mask] >= highest_class) & \
                                 (volume[lung_mask] <= max_hu)
        
    except Exception as e:
        logger.warning(f"   Erreur multi-Otsu: {e}, fallback Otsu simple")
        candidates = _apply_single_threshold(volume, lung_mask, 
                                            min_hu, min_hu, max_hu)
    
    return candidates

def _multi_scale_threshold(volume: np.ndarray,
                          lung_mask: np.ndarray,
                          min_hu: float,
                          max_hu: float,
                          scales: list = [0.5, 1.0, 2.0]) -> np.ndarray:
    """Approche multi-échelle pour détecter des lésions de tailles variées"""
    
    from skimage.transform import resize
    
    candidates = np.zeros_like(volume, dtype=bool)
    original_shape = volume.shape
    
    for scale in scales:
        if scale == 1.0:
            # Pas de redimensionnement à l'échelle 1.0
            lung_values = volume[lung_mask]
            if len(lung_values) == 0:
                continue
            thresh = np.percentile(lung_values, 95)
            candidates_scale = (volume >= max(thresh, min_hu)) & lung_mask
        else:
            # Redimensionner volume ET lung_mask à la même taille
            new_shape = [max(1, int(s * scale)) for s in original_shape]
            vol_resized  = resize(volume,    new_shape, preserve_range=True, anti_aliasing=(scale < 1.0))
            mask_resized = resize(lung_mask.astype(float), new_shape, order=0,
                                  preserve_range=True, anti_aliasing=False) > 0.5

            lung_values = vol_resized[mask_resized]
            if len(lung_values) == 0:
                continue
            thresh = np.percentile(lung_values, 95)
            candidates_scale = (vol_resized >= max(thresh, min_hu)) & mask_resized

            # Remettre à l'échelle originale
            candidates_scale = resize(candidates_scale.astype(float),
                                      original_shape, order=0,
                                      preserve_range=True, anti_aliasing=False) > 0.5
        
        candidates = np.logical_or(candidates, candidates_scale)
    
    return candidates

def _clean_candidates(candidates: np.ndarray, min_size: int = 5) -> np.ndarray:
    """Nettoie le masque des candidats"""
    
    # 1. Opening pour séparer les régions mal connectées
    structure = ndimage.generate_binary_structure(3, 1)
    candidates = ndimage.binary_opening(candidates, structure=structure)
    
    # 2. Fermeture pour combler les petits trous
    candidates = ndimage.binary_closing(candidates, structure=structure)
    
    # 3. Enlever les petites régions (< min_size)
    candidates = remove_small_objects(candidates, min_size=min_size)
    
    # 4. Option: remplir les trous dans les régions (attention aux vaisseaux)
    # candidates = binary_fill_holes(candidates)
    
    return candidates

def threshold_3d_slicewise(volume: np.ndarray,
                          lung_mask: np.ndarray,
                          method: str = 'local',
                          block_size: int = 51,
                          offset: float = 0.01) -> np.ndarray:
    """
    Applique un seuillage 2D coupe par coupe (pour méthodes locales)
    """
    candidates = np.zeros_like(volume, dtype=bool)
    
    logger.info(f"🔍 Seuillage slice-wise (méthode: {method})")
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z]
        mask_2d = lung_mask[z]
        
        if mask_2d.sum() == 0:
            continue
        
        if method == 'local':
            # Seuillage local adaptatif
            threshold = threshold_local(slice_2d, block_size, offset=offset)
            candidates_slice = slice_2d > threshold
        else:
            # Autres méthodes
            lung_values = slice_2d[mask_2d]
            if len(lung_values) == 0:
                continue
                
            if method == 'otsu':
                thresh = threshold_otsu(lung_values)
            elif method == 'percentile':
                thresh = np.percentile(lung_values, 95)
            
            candidates_slice = slice_2d > thresh
        
        # Appliquer le masque pulmonaire
        candidates_slice = candidates_slice & mask_2d
        candidates[z] = candidates_slice
    
    n_voxels = candidates.sum()
    logger.info(f"✅ {n_voxels} voxels candidats détectés")
    
    return candidates

def get_lung_mask_from_hu(volume: np.ndarray,
                         min_hu: float = -900,
                         max_hu: float = -400) -> np.ndarray:
    """
    Génère un masque pulmonaire simple par seuillage HU
    Utile si on n'a pas de segmentation pulmonaire pré-calculée
    """
    lung_mask = (volume >= min_hu) & (volume <= max_hu)
    
    # Nettoyage morphologique
    structure = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_opening(lung_mask, structure=structure, iterations=2)
    lung_mask = ndimage.binary_closing(lung_mask, structure=structure, iterations=2)
    
    # Garder les plus grandes composantes (poumons gauche/droit)
    labeled, n_regions = ndimage.label(lung_mask)
    if n_regions > 0:
        sizes = ndimage.sum(lung_mask, labeled, range(1, n_regions+1))
        largest_labels = np.argsort(sizes)[-2:]  # 2 plus grandes régions
        lung_mask = np.isin(labeled, largest_labels + 1)
    
    logger.info(f"🫁 Masque pulmonaire généré: {lung_mask.sum()} voxels")
    
    return lung_mask

if __name__ == "__main__":
    # Test simple
    print("🧪 Test du module thresholding")
    
    # Créer des données factices
    test_vol = np.random.randn(20, 50, 50) * 100
    test_lung = np.zeros_like(test_vol, dtype=bool)
    test_lung[:, 10:40, 10:40] = True
    
    # Ajouter des "embolies" artificielles
    test_vol[5:8, 20:25, 20:25] += 200  # Région dense
    test_vol[12:15, 30:35, 30:35] += 300  # Région très dense
    
    # Tester différentes méthodes
    methods = ['percentile', 'otsu', 'fixed']
    
    for method in methods:
        print(f"\n📊 Méthode: {method}")
        candidates = adaptive_threshold(test_vol, test_lung, 
                                       method=method, percentile=90)
        print(f"   Candidats: {candidates.sum()} voxels")