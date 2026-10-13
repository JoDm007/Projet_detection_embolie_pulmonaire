#Autor: Joseph DATE-MASSE
# src/O3_radiomics/feature_extractor.py

import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops, marching_cubes, mesh_surface_area
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis, entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbolismFeatureExtractor:
    """
    Extracteur de features radiomiques léger sans dépendance C++ externe (alternative à pyradiomics).
    Utilise NumPy, SciPy et Scikit-Image pour extraire les caractéristiques locales
    des candidats d'embolie pulmonaire (Shape, Intensité, Texture GLCM).
    """
    def __init__(self, config_path: str = None):
        """
        Initialise l'extracteur. La configuration peut définir le spacing (taille des voxels).
        """
        logger.info("✅ Extracteur léger (scikit-image) initialisé.")
        
    def extract_from_arrays(self, 
                            image_array: np.ndarray, 
                            mask_array: np.ndarray, 
                            spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
        """
        Extrait les features à partir de tableaux numpy.
        
        Args:
            image_array: Volume 3D en unités Hounsfield (HU)
            mask_array: Masque binaire 3D (1 pour l'embolie, 0 pour le fond)
            spacing: Taille du voxel en mm (z, y, x) ou (x, y, z) dépendant de l'ordre
        """
        features = {}
        
        # Vérification
        if mask_array.sum() == 0:
            logger.warning("Masque vide fourni.")
            return features
            
        # Isoler les intensités de la région d'intérêt
        roi_intensities = image_array[mask_array > 0]
        
        # === 1. FIRST ORDER FEATURES (Intensité) ===
        features['intensity_mean'] = float(np.mean(roi_intensities))
        features['intensity_median'] = float(np.median(roi_intensities))
        features['intensity_std'] = float(np.std(roi_intensities))
        features['intensity_variance'] = float(np.var(roi_intensities))
        features['intensity_min'] = float(np.min(roi_intensities))
        features['intensity_max'] = float(np.max(roi_intensities))
        features['intensity_skewness'] = float(skew(roi_intensities))
        features['intensity_kurtosis'] = float(kurtosis(roi_intensities))
        
        # Calcul de l'entropie de l'histogramme des intensités
        hist, _ = np.histogram(roi_intensities, bins=64, density=True)
        hist = hist[hist > 0]
        features['intensity_entropy'] = float(entropy(hist))
        
        # === 2. SHAPE FEATURES (Forme 3D) ===
        labels, _ = ndimage.label(mask_array > 0)
        props = regionprops(labels, intensity_image=image_array)
        
        if len(props) > 0:
            main_region = max(props, key=lambda r: r.area)
            
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            features['shape_volume_mm3'] = float(main_region.area * voxel_volume)
            
            features['shape_major_axis_length'] = float(main_region.major_axis_length * spacing[0])
            features['shape_minor_axis_length'] = float(main_region.minor_axis_length * spacing[0])
            
            if main_region.major_axis_length > 0:
                features['shape_elongation'] = float(main_region.minor_axis_length / main_region.major_axis_length)
            else:
                features['shape_elongation'] = 0.0

            try:
                padded_mask = np.pad(mask_array, pad_width=1, mode='constant', constant_values=0)
                verts, faces, _, _ = marching_cubes(padded_mask, level=0.5, spacing=spacing)
                surface_area = mesh_surface_area(verts, faces)
                features['shape_surface_area_mm2'] = float(surface_area)
                
                volume = features['shape_volume_mm3']
                if surface_area > 0:
                    sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
                    features['shape_sphericity'] = min(float(sphericity), 1.0)
            except Exception as e:
                logger.debug(f"Impossible de calculer la surface: {e}")
                features['shape_surface_area_mm2'] = 0.0
                features['shape_sphericity'] = 0.0
                
        # === 3. GLCM TEXTURE FEATURES (Approximation 2D) ===
        try:
            z_indices, y_indices, x_indices = np.where(mask_array > 0)
            if len(z_indices) > 0:
                z_center = int(np.median(z_indices))
                
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                
                slice_2d = image_array[z_center, min_y:max_y+1, min_x:max_x+1].copy()
                mask_2d = mask_array[z_center, min_y:max_y+1, min_x:max_x+1]
                
                min_hu, max_hu, bin_width = -200, 400, 25
                slice_2d = np.clip(slice_2d, min_hu, max_hu)
                slice_digitized = ((slice_2d - min_hu) / bin_width).astype(np.uint8)
                
                slice_digitized = (slice_digitized + 1) * mask_2d
                
                # GLCM: use normed=False and normalize manually to avoid double normalization
                glcm = graycomatrix(slice_digitized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=256, symmetric=True, normed=False)
                
                glcm = glcm[1:, 1:, :, :] 
                
                # Normalize manually: divide by sum to get probabilities
                sum_glcm = np.sum(glcm, axis=(0, 1))
                sum_glcm[sum_glcm == 0] = 1
                glcm_norm = glcm / sum_glcm

                features['texture_contrast'] = float(np.mean(graycoprops(glcm_norm, 'contrast')))
                features['texture_dissimilarity'] = float(np.mean(graycoprops(glcm_norm, 'dissimilarity')))
                features['texture_homogeneity'] = float(np.mean(graycoprops(glcm_norm, 'homogeneity')))
                features['texture_energy'] = float(np.mean(graycoprops(glcm_norm, 'energy')))
                features['texture_correlation'] = float(np.mean(graycoprops(glcm_norm, 'correlation')))
        except Exception as e:
            logger.debug(f"Erreur GLCM: {e}")
            
        return features

if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU FEATURE EXTRACTOR")
    print("="*60)
    
    print("Génération de données synthétiques (Volume 50x50x50)...")
    np.random.seed(42)
    test_vol = np.random.normal(loc=-800, scale=50, size=(50, 50, 50)).astype(np.float32)
    
    test_mask = np.zeros_like(test_vol, dtype=np.uint8)
    radius = 10
    center = (25, 25, 25)
    for z in range(50):
        for y in range(50):
            for x in range(50):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                    test_mask[z, y, x] = 1
                    test_vol[z, y, x] = np.random.normal(loc=100, scale=20)
                    
    print(f"Volume HU: {test_vol.shape}, Masque: {test_mask.shape} (Voxel sum: {test_mask.sum()})")
    
    extractor = EmbolismFeatureExtractor()
    features = extractor.extract_from_arrays(test_vol, test_mask)
    
    if features:
        print("\n✅ SUPER ! Les features ont pu être extraites sans pyradiomics.")
        print("\nExemples :")
        for k, v in features.items():
            print(f"  - {k:<25} : {v:.4f}")
            
        print(f"\n=> Nombre total de variables extraites : {len(features)}")
    else:
        print("\n⚠️ Echec de l'extraction.")