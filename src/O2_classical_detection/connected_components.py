# src/2_classical_detection/connected_components.py

import numpy as np
from scipy import ndimage
from skimage import measure
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
from scipy.stats import kurtosis, skew
from skimage.measure import regionprops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_candidates_3d(mask: np.ndarray,
                          volume: np.ndarray,
                          min_size: int = 10,
                          spacing: Optional[Tuple[float, float, float]] = None,
                          compute_texture: bool = False) -> pd.DataFrame:
    """
    Extrait les caractéristiques de chaque région candidate 3D.
    
    Args:
        mask: Masque binaire 3D des candidats
        volume: Volume original en HU (pour extraire les intensités)
        min_size: Taille minimale en voxels
        spacing: Espacement des voxels (Z, Y, X) en mm
        compute_texture: Calculer les features de texture (GLCM)
    
    Returns:
        df: DataFrame avec les features de chaque candidat
    """
    logger.info("🔬 Extraction des caractéristiques 3D...")
    
    if mask.sum() == 0:
        logger.info("   Aucun candidat à extraire")
        return pd.DataFrame()
    
    # Étiquetage 3D avec structure de connexité 3D (26 voisins)
    structure = ndimage.generate_binary_structure(3, 1)  # 1 = connexité 26
    labeled, num = ndimage.label(mask.astype(int), structure=structure)
    
    if num == 0:
        return pd.DataFrame()
    
    logger.info(f"   {num} composantes connexes trouvées")
    
    # Extraction des features
    features_list = []
    
    # Utiliser regionprops de skimage pour des features supplémentaires
    regions = regionprops(labeled, intensity_image=volume)
    
    for i, region in enumerate(regions, 1):
        if region.area < min_size:
            continue
        
        # Masque de la région
        region_mask = (labeled == i)
        coords = np.where(region_mask)
        
        # 1. Features géométriques de base
        z, y, x = coords
        bbox = {
            'z_min': int(z.min()), 'z_max': int(z.max()),
            'y_min': int(y.min()), 'y_max': int(y.max()),
            'x_min': int(x.min()), 'x_max': int(x.max())
        }
        
        # 2. Features de volume (voxels et mm³)
        volume_voxels = region.area
        extent_z = z.max() - z.min()
        extent_y = y.max() - y.min()
        extent_x = x.max() - x.min()
        
        features = {
            'region_id': i,
            'volume_voxels': volume_voxels,
            'extent_z': extent_z,
            'extent_y': extent_y,
            'extent_x': extent_x,
            'centroid_z': region.centroid[0],
            'centroid_y': region.centroid[1],
            'centroid_x': region.centroid[2],
        }
        
        # 3. Volume en mm³ si spacing fourni
        if spacing is not None:
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
            features['volume_mm3'] = volume_voxels * voxel_volume_mm3
            features['extent_z_mm'] = extent_z * spacing[0]
            features['extent_y_mm'] = extent_y * spacing[1]
            features['extent_x_mm'] = extent_x * spacing[2]
        
        # 4. Features de forme (100% COMPATIBLE 3D)
        features['sphericity'] = region.sphericity if hasattr(region, 'sphericity') else 0

        # Surface area et compactness (3D only)
        if hasattr(region, 'surface_area'):
            features['surface_area'] = region.surface_area
            features['compactness'] = (36 * np.pi * (region.area ** 2)) / (region.surface_area ** 3 + 1e-6)
        else:
            features['surface_area'] = 0
            features['compactness'] = 0

        # Extent et Solidity (disponibles en 3D)
        features['extent'] = region.extent if hasattr(region, 'extent') else 0
        features['solidity'] = region.solidity if hasattr(region, 'solidity') else 0

        # Equivalent diameter (disponible en 3D)
        features['equivalent_diameter'] = region.equivalent_diameter if hasattr(region, 'equivalent_diameter') else 0

        # Elongation ratio (calcul manuel pour remplacer eccentricity)
        extents = [features['extent_z'], features['extent_y'], features['extent_x']]
        if max(extents) > 0:
            features['elongation_ratio'] = min(extents) / max(extents)
        else:
            features['elongation_ratio'] = 0      
        # 5. Features d'intensité (dans le volume original)
        intensities = volume[region_mask]
        
        features.update({
            'intensity_mean': float(intensities.mean()),
            'intensity_std': float(intensities.std()),
            'intensity_min': float(intensities.min()),
            'intensity_max': float(intensities.max()),
            'intensity_median': float(np.median(intensities)),
            'intensity_p10': float(np.percentile(intensities, 10)),
            'intensity_p90': float(np.percentile(intensities, 90)),
            'intensity_iqr': float(np.percentile(intensities, 75) - np.percentile(intensities, 25)),
            'intensity_skewness': float(skew(intensities.flatten())),
            'intensity_kurtosis': float(kurtosis(intensities.flatten())),
            'intensity_energy': float((intensities**2).sum()),
            'intensity_entropy': float(-(intensities * np.log(intensities + 1e-7)).sum())
        })
        
        # 6. Features de texture (si demandé)
        if compute_texture and region.area > 100:  # Nécessite une région assez grande
            texture_features = extract_texture_features(volume, region_mask)
            features.update(texture_features)
        
        # Ajouter les coordonnées bbox
        features.update(bbox)
        
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    if len(df) > 0:
        logger.info(f"✅ {len(df)} candidats extraits avec {len(df.columns)} features")
        # Afficher les statistiques des features principales
        if 'volume_mm3' in df.columns:
            logger.info(f"   Volume moyen: {df['volume_mm3'].mean():.1f} mm³")
        logger.info(f"   Intensité moyenne: {df['intensity_mean'].mean():.1f} HU")
    
    return df


def extract_texture_features(volume: np.ndarray, 
                            region_mask: np.ndarray,
                            distances: List[int] = [1],
                            angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                            n_levels: int = 64) -> Dict:  # ← Nouveau paramètre
    """
    Extrait des features de texture GLCM
        Args:
        volume: Volume original
        region_mask: Masque de la région
        distances: Distances pour GLCM
        angles: Angles pour GLCM
        n_levels: Nombre de niveaux de gris pour la quantification (64 recommandé)
    Returns:
        dict: Features de texture
        """
    try:
        from skimage.feature import graycomatrix, graycoprops
        
        # Extraire la région avec padding
        coords = np.where(region_mask)
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        # Ajout d'un petit padding
        z_min = max(0, z_min - 2)
        z_max = min(volume.shape[0], z_max + 2)
        y_min = max(0, y_min - 2)
        y_max = min(volume.shape[1], y_max + 2)
        x_min = max(0, x_min - 2)
        x_max = min(volume.shape[2], x_max + 2)
        
        # Extraire le sous-volume
        sub_volume = volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        sub_mask = region_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        # Quantifier avec n_levels au lieu de 256 (plus rapide et stable)
        if sub_volume.max() > sub_volume.min():
            sub_vol_norm = ((sub_volume - sub_volume.min()) * (n_levels - 1) / 
                           (sub_volume.max() - sub_volume.min())).astype(np.uint8)
        else:
            sub_vol_norm = np.zeros_like(sub_volume, dtype=np.uint8)
        
        # Appliquer le masque
        sub_vol_norm[~sub_mask] = 0
        
        # Calculer GLCM en 3D (moyenne sur toutes les directions)
        glcm = graycomatrix(sub_vol_norm, distances, angles, levels=256, symmetric=True)
        
        # Features de texture
        texture_features = {
            'texture_contrast': float(graycoprops(glcm, 'contrast').mean()),
            'texture_dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
            'texture_homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
            'texture_energy': float(graycoprops(glcm, 'energy').mean()),
            'texture_correlation': float(graycoprops(glcm, 'correlation').mean()),
            'texture_ASM': float(graycoprops(glcm, 'ASM').mean())
        }
        
        return texture_features
        
    except Exception as e:
        logger.warning(f"   Erreur calcul texture: {e}")
        return {}

def filter_candidates(df: pd.DataFrame,
                     min_volume: int = 10,
                     max_volume: Optional[int] = None,
                     intensity_range: Optional[Tuple[float, float]] = None,
                     min_sphericity: float = 0.0) -> pd.DataFrame:
    """
    Filtre les candidats selon des critères
    
    Args:
        df: DataFrame des candidats
        min_volume: Volume minimum en voxels
        max_volume: Volume maximum
        intensity_range: Plage d'intensité acceptable
        min_sphericity: Sphéricité minimum
    
    Returns:
        df_filtre: DataFrame filtré
    """
    if df.empty:
        return df
    
    mask = df['volume_voxels'] >= min_volume
    
    if max_volume:
        mask &= df['volume_voxels'] <= max_volume
    
    if intensity_range:
        mask &= (df['intensity_mean'] >= intensity_range[0]) & \
                (df['intensity_mean'] <= intensity_range[1])
    
    if min_sphericity > 0:
        mask &= df['sphericity'] >= min_sphericity
    
    df_filtre = df[mask].copy()
    
    logger.info(f"Filtrage: {len(df)} -> {len(df_filtre)} candidats")
    
    return df_filtre


def calculate_elongation_ratio(extent_z, extent_y, extent_x):
    """
    Calcule un ratio d'allongement 3D (similaire à eccentricity)
    
    Returns:
        1.0 = sphère parfaite (toutes dimensions égales)
        0.0 = très allongé (une dimension >> autres)
    """
    extents = [extent_z, extent_y, extent_x]
    if max(extents) > 0:
        return min(extents) / max(extents)
    return 0

def compute_overlap_with_gt(candidates_df: pd.DataFrame,
                           gt_mask: np.ndarray,
                           iou_threshold: float = 0.5) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df
    
    gt_labeled, num_gt = ndimage.label(gt_mask)
    
    labels, iou_values = [], []  # ← Deux listes
    
    for idx, candidate in candidates_df.iterrows():
        candidate_mask = np.zeros_like(gt_mask, dtype=bool)
        z_min, z_max = int(candidate['z_min']), int(candidate['z_max'])
        y_min, y_max = int(candidate['y_min']), int(candidate['y_max'])
        x_min, x_max = int(candidate['x_min']), int(candidate['x_max'])
        
        candidate_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = True
        
        # Calculer l'IoU avec chaque région GT
        max_iou = 0
        for gt_id in range(1, num_gt + 1):
            gt_region = (gt_labeled == gt_id)
            intersection = np.logical_and(candidate_mask, gt_region).sum()
            union = np.logical_or(candidate_mask, gt_region).sum()
            
            if union > 0:
                iou = intersection / union
                max_iou = max(max_iou, iou)        
        labels.append(1 if max_iou >= iou_threshold else 0)
        iou_values.append(max_iou)  # ← Stocker chaque IoU
    
    candidates_df['is_positive'] = labels
    candidates_df['iou_max'] = iou_values  # ← Liste complète
    
    n_pos = sum(labels)
    logger.info(f"   {n_pos} candidats positifs, {len(labels)-n_pos} négatifs")
    
    return candidates_df

if __name__ == "__main__":
    # Test simple
    print("🧪 Test du module connected_components")
    
    # Créer des données factices
    test_volume = np.random.randn(50, 100, 100) * 100 + 0  # Simule des HU
    test_mask = np.zeros((50, 100, 100), dtype=bool)
    
    # Ajouter quelques régions artificielles
    test_mask[10:15, 30:40, 40:50] = True  # Région 1
    test_mask[25:28, 60:65, 70:75] = True  # Région 2
    test_mask[35:45, 10:20, 10:20] = True  # Région 3 (plus grande)
    
    # Extraire candidats
    df = extract_candidates_3d(test_mask, test_volume, min_size=5)
    print(f"\nRésultat:\n{df[['region_id', 'volume_voxels', 'intensity_mean']].head()}")
    
    # Tester le filtrage
    df_filtre = filter_candidates(df, min_volume=50)
    print(f"\nAprès filtrage: {len(df_filtre)} candidats")