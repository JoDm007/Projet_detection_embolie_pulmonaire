# src/1_preprocessing/resampling.py

import numpy as np
from scipy.ndimage import zoom
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def resample_volume(volume: np.ndarray, 
                   current_spacing: Union[list, tuple, np.ndarray], 
                   target_spacing: Union[list, tuple, np.ndarray] = [1.0, 1.0, 1.0],
                   order: int = 1,
                   preserve_range: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise la résolution spatiale du volume (Axe Z, Y, X).
    
    Args:
        volume: Array 3D (Z, Y, X)
        current_spacing: Liste [Z_thick, Y_pix, X_pix] (venant de tes métadonnées)
        target_spacing: Résolution souhaitée (par défaut 1mm partout)
        order: Ordre d'interpolation (1=linéaire, 3=cubique)
        preserve_range: Préserver la plage de valeurs après interpolation
    
    Returns:
        resampled_volume: Volume redimensionné
        new_spacing: Nouvel espacement (target_spacing réel après arrondi)
    
      """
    # Conversion en array numpy
    current_spacing = np.array(current_spacing)
    target_spacing = np.array(target_spacing)
    
    # Validation
    if len(current_spacing) != 3 or len(target_spacing) != 3:
        raise ValueError("Les espacements doivent être de longueur 3 (Z, Y, X)")
    
    if np.any(current_spacing <= 0) or np.any(target_spacing <= 0):
        raise ValueError("Les espacements doivent être positifs")
    
    # Calcul du facteur de redimensionnement
    # resize_factor = espacement_actuel / espacement_cible
    resize_factor = current_spacing / target_spacing
    
    # Calcul des nouvelles dimensions
    new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)
    
    # S'assurer que les dimensions sont au moins de 1
    new_shape = np.maximum(new_shape, 1)
    
    # Recalcul du facteur réel (pour être hyper précis avec l'arrondi)
    real_resize_factor = new_shape / np.array(volume.shape)
    
    logger.info(f"Resampling volume: {volume.shape} -> {new_shape} "
                f"(facteur: {real_resize_factor.round(3)})")
    
    # Interpolation
    resampled_volume = zoom(volume, real_resize_factor, 
                           order=order, 
                           mode='constant',
                           cval=volume.min() if preserve_range else 0)
    
    # Nouvel espacement réel après redimensionnement
    new_spacing = current_spacing / real_resize_factor
    
    return resampled_volume, new_spacing

def resample_mask(mask: np.ndarray,
                 current_spacing: Union[list, tuple, np.ndarray], 
                 target_spacing: Union[list, tuple, np.ndarray] = [1.0, 1.0, 1.0]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensionne le masque (ordre 0 pour garder des valeurs binaires).
    
    Args:
        mask: Array 3D binaire (Z, Y, X)
        current_spacing: Liste [Z_thick, Y_pix, X_pix]
        target_spacing: Résolution souhaitée
    
    Returns:
        resampled_mask: Masque redimensionné (binaire)
        new_spacing: Nouvel espacement
    """
    current_spacing = np.array(current_spacing)
    target_spacing = np.array(target_spacing)
    
    resize_factor = current_spacing / target_spacing
    new_shape = np.round(np.array(mask.shape) * resize_factor).astype(int)
    new_shape = np.maximum(new_shape, 1)
    real_resize_factor = new_shape / np.array(mask.shape)
    
    logger.info(f"Resampling masque: {mask.shape} -> {new_shape}")
    
    # order=0 est crucial pour garder un masque binaire (0 ou 1)
    resampled_mask = zoom(mask, real_resize_factor, 
                          order=0, 
                          mode='constant',
                          cval=0)
    
    # S'assurer que le masque reste binaire
    resampled_mask = (resampled_mask > 0.5).astype(np.uint8)
    
    new_spacing = current_spacing / real_resize_factor
    
    return resampled_mask, new_spacing

def resample_to_target_shape(volume: np.ndarray,
                            current_spacing: Union[list, tuple, np.ndarray],
                            target_shape: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """
    Redimensionne un volume à une forme cible spécifique.
    Utile pour avoir des entrées de taille fixe pour le deep learning.
    
    Args:
        volume: Volume 3D
        current_spacing: Espacement actuel
        target_shape: Forme souhaitée (Z, Y, X)
    
    Returns:
        Volume redimensionné à target_shape
    """
    current_spacing = np.array(current_spacing)
    target_shape = np.array(target_shape)
    
    # Facteur de redimensionnement pour atteindre target_shape
    resize_factor = target_shape / np.array(volume.shape)
    
    logger.info(f"Reshaping: {volume.shape} -> {target_shape} (facteur: {resize_factor.round(3)})")
    
    resampled = zoom(volume, resize_factor, order=1, mode='constant')
    
    return resampled

def crop_or_pad_volume(volume: np.ndarray,
                      target_shape: Union[list, tuple, np.ndarray],
                      pad_value: float = 0) -> np.ndarray:
    """
    Coupe ou ajoute du padding pour atteindre une forme cible.
    Alternative au redimensionnement quand on veut préserver la résolution.
    
    Args:
        volume: Volume 3D
        target_shape: Forme souhaitée (Z, Y, X)
        pad_value: Valeur pour le padding
    
    Returns:
        Volume de forme target_shape
    """
    target_shape = np.array(target_shape)
    current_shape = np.array(volume.shape)
    
    result = np.full(target_shape, pad_value, dtype=volume.dtype)
    
    # Calcul des indices de début/fin pour le volume original
    start = np.maximum(0, (target_shape - current_shape) // 2)
    end = start + current_shape
    
    # Ajustement si le volume original est plus grand que target_shape
    vol_start = np.maximum(0, -((target_shape - current_shape) // 2))
    vol_end = vol_start + target_shape
    
    # Indices pour le résultat
    r_start = np.maximum(0, start)
    r_end = np.minimum(target_shape, end)
    
    # Indices pour le volume original
    v_start = vol_start
    v_end = vol_start + (r_end - r_start)
    
    # Copier la partie qui tient
    slices_result = tuple(slice(s, e) for s, e in zip(r_start, r_end))
    slices_volume = tuple(slice(s, e) for s, e in zip(v_start, v_end))
    
    result[slices_result] = volume[slices_volume]
    
    logger.info(f"Crop/pad: {current_shape} -> {target_shape}")
    
    return result

def standardize_spacing(volume: np.ndarray,
                       mask: Optional[np.ndarray],
                       spacing: Tuple[float, float, float],
                       target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Pipeline complet de standardisation de l'espacement.
    
    Args:
        volume: Volume CT
        mask: Masque optionnel
        spacing: Espacement actuel (Z, Y, X)
        target_spacing: Espacement cible
    
    Returns:
        volume_std: Volume standardisé
        mask_std: Masque standardisé (ou None)
        new_spacing: Nouvel espacement
    """
    # Redimensionner le volume
    volume_std, new_spacing = resample_volume(volume, spacing, target_spacing)
    
    # Redimensionner le masque si fourni
    mask_std = None
    if mask is not None:
        mask_std, _ = resample_mask(mask, spacing, target_spacing)
        
        # Vérification rapide
        if volume_std.shape != mask_std.shape:
            logger.warning(f"Formes différentes après resampling: volume {volume_std.shape}, masque {mask_std.shape}")
            # Ajuster si nécessaire
            min_shape = np.minimum(volume_std.shape, mask_std.shape)
            volume_std = volume_std[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask_std = mask_std[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return volume_std, mask_std, new_spacing

# Test simple si exécuté directement
if __name__ == "__main__":
    print("🧪 Test du module de resampling")
    
    # Créer un volume factice
    test_volume = np.random.rand(100, 512, 512)
    test_mask = (np.random.rand(100, 512, 512) > 0.95).astype(np.uint8)
    
    current_spacing = [2.5, 0.7, 0.7]  # Z épais, XY fin
    target_spacing = [1.0, 1.0, 1.0]
    
    print(f"\nVolume original: {test_volume.shape}, spacing: {current_spacing}")
    
    # Test resampling volume
    vol_resampled, new_spacing = resample_volume(test_volume, current_spacing, target_spacing)
    print(f"Volume après resampling: {vol_resampled.shape}, spacing: {new_spacing}")
    
    # Test resampling masque
    mask_resampled, _ = resample_mask(test_mask, current_spacing, target_spacing)
    print(f"Masque après resampling: {mask_resampled.shape}, valeurs uniques: {np.unique(mask_resampled)}")
    
    # Test pipeline complet
    vol_std, mask_std, final_spacing = standardize_spacing(test_volume, test_mask, current_spacing, target_spacing)
    print(f"\nPipeline complet: volume {vol_std.shape}, masque {mask_std.shape}")
    print(f"Espacement final: {final_spacing}")