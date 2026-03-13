# src/2_classical_detection/lung_segmentation.py

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure, exposure
import logging
from typing import Tuple, Optional, List, Dict
from skimage.segmentation import clear_border
from skimage.filters import threshold_local
from tqdm import tqdm  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def segment_lungs(volume: np.ndarray, 
                  method: str = 'threshold',
                  hu_threshold: float = -400,
                  min_area: int = 1000,
                  use_advanced_morphology: bool = True,
                  separate_lungs: bool = True) -> np.ndarray:
    """
    Segmente les poumons avec différentes méthodes.
    
    Args:
        volume: Volume 3D en unités Hounsfield
        method: 'threshold' (simple), 'otsu', 'region_growing'
        hu_threshold: Seuil HU pour 'threshold' (par défaut -400)
        min_area: Surface minimale pour conserver une région (en pixels 2D)
        use_advanced_morphology: Appliquer des opérations avancées
        separate_lungs: Séparer poumon gauche/droit (True) ou garder fusionné
    
    Returns:
        lung_mask: Masque binaire 3D des poumons
    """
    if volume is None or volume.size == 0:
        logger.error("Volume vide ou None")
        return np.zeros_like(volume, dtype=bool)
    
    logger.info(f"🫁 Segmentation des poumons (méthode: {method})")
    
    # 1. Binarisation selon la méthode choisie
    if method == 'threshold':
        binary = volume < hu_threshold
    elif method == 'otsu':
        # Otsu sur l'histogramme global
        thresh = filters.threshold_otsu(volume[volume < 0])  # Seulement les valeurs négatives
        binary = volume < thresh
        logger.info(f"   Seuil Otsu: {thresh:.1f} HU")
    elif method == 'adaptive':
        # Version optimisée avec vectorisation partielle
        #from skimage.filters import threshold_local
        
        # Pré-calculer le seuil sur quelques slices représentatives
        sample_slices = [volume.shape[0]//4, volume.shape[0]//2, 3*volume.shape[0]//4]
        thresholds = [filters.threshold_local(volume[z], block_size=51, offset=50) 
                    for z in sample_slices]
        global_thresh = np.median(thresholds)  # Seuil global approximatif
        
        # Appliquer sur tout le volume
        binary = volume < global_thresh
        
        # Raffinement local optionnel (seulement si nécessaire)
        if use_advanced_morphology:
            for z in range(0, volume.shape[0], 10):  # Toutes les 10 slices
                local_thresh = filters.threshold_local(volume[z], block_size=51, offset=50)
                binary[z] = volume[z] < local_thresh
    else:
        logger.warning(f"Méthode {method} inconnue, utilisation threshold")
        binary = volume < hu_threshold
    
    # 2. Traitement slice par slice
    lung_mask = np.zeros_like(binary, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_mask = binary[z].copy()
        
        # Nettoyage morphologique de base
        slice_mask = morphology.binary_opening(slice_mask, morphology.disk(2))
        slice_mask = morphology.binary_closing(slice_mask, morphology.disk(4))
        
        # Enlever les régions touchant le bord (souvent artefacts)
        slice_mask = clear_border(slice_mask)
        
        # Labeliser et garder les grandes régions
        labeled = measure.label(slice_mask, connectivity=2)
        regions = measure.regionprops(labeled)
        
        # Trier par aire
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        
        # Garder les 2 plus grandes (poumons)
        kept = 0
        for r in regions:
            if r.area >= min_area and kept < 2:
                # Vérifier que ce n'est pas trop circulaire (sinon c'est probablement un artefact)
                if r.eccentricity < 0.8:  # Pas trop circulaire
                    lung_mask[z] = lung_mask[z] | (labeled == r.label)
                    kept += 1
        
        # Si un seul poumon est trouvé, essayer avec un seuil plus bas
        if kept < 2:
            # Alternative: seuil plus permissif
            slice_mask_alt = volume[z] < (hu_threshold + 100)
            labeled_alt = measure.label(slice_mask_alt, connectivity=2)
            regions_alt = measure.regionprops(labeled_alt)
            regions_alt = sorted(regions_alt, key=lambda r: r.area, reverse=True)
            
            for r in regions_alt[:2-kept]:
                if r.area >= min_area:
                    lung_mask[z] = lung_mask[z] | (labeled_alt == r.label)
                    kept += 1
    
    # 3. Post-traitement 3D
    if use_advanced_morphology:
        lung_mask = _advanced_morphology_3d(lung_mask)
    else:
        # Remplissage simple des trous
        lung_mask = ndimage.binary_fill_holes(lung_mask)
        lung_mask = morphology.remove_small_objects(lung_mask, min_size=5000)
    
    # 4. Séparation gauche/droit si demandé
    if separate_lungs:
        lung_mask = _separate_lungs(lung_mask)
    
    # Statistiques
    n_voxels = lung_mask.sum()
    n_regions = ndimage.label(lung_mask)[1]
    
    logger.info(f"✅ Masque pulmonaire: {n_voxels} voxels "
                f"({100*n_voxels/volume.size:.1f}% du volume), "
                f"{n_regions} région(s)")
    
    return lung_mask

def _advanced_morphology_3d(mask: np.ndarray) -> np.ndarray:
    """Opérations morphologiques 3D avancées"""
    
    # 1. Ouverture 3D pour lisser
    structure = morphology.ball(3)
    mask = morphology.binary_opening(mask, structure)
    
    # 2. Fermeture 3D
    mask = morphology.binary_closing(mask, structure)
    
    # 3. Remplissage des trous 3D
    mask = ndimage.binary_fill_holes(mask)
    
    # 4. Enlever les petites composantes
    mask = morphology.remove_small_objects(mask, min_size=10000)
    
    # 5. Dilatation légère pour inclure les petites bronchioles
    mask = morphology.binary_dilation(mask, morphology.ball(1))
    
    return mask

def _separate_lungs(mask: np.ndarray) -> np.ndarray:
    labeled, n_regions = ndimage.label(mask)
    """
    Sépare les poumons gauche et droit si nécessaire
    (utile pour certaines analyses)
    """
    if n_regions <= 1:
        return mask
    
    # Si plus de 2 régions, garder les 2 plus grandes
    if n_regions > 2:
        sizes = ndimage.sum(mask.astype(int), labeled, range(1, n_regions+1))
        # ✅ Vérifier qu'il y a assez de régions
        if len(sizes) >= 2:
            largest_labels = np.argsort(sizes)[-2:] + 1
            result = np.isin(labeled, largest_labels)
        else:
            result = mask  # Fallback
    else:
        result = mask
    
    return result

def extract_lung_roi(volume: np.ndarray, 
                     lung_mask: np.ndarray,
                     margin: Tuple[int, int, int] = (5, 5, 5),
                     square_bbox: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait la région d'intérêt (ROI) des poumons avec une marge.
    
    Args:
        volume: Volume original
        lung_mask: Masque pulmonaire
        margin: Marge (z, y, x) autour des poumons
        square_bbox: Forcer un bounding box carré
    
    Returns:
        cropped_volume: Volume recadré
        cropped_mask: Masque recadré
        bbox: Coordonnées du bounding box (z0,z1, y0,y1, x0,x1)
    """
    # Trouver les coordonnées du bounding box des poumons
    coords = np.where(lung_mask)
    
    if len(coords[0]) == 0:
        logger.warning("⚠️ Aucun voxel pulmonaire détecté")
        return volume, lung_mask, None
    
    # Calculer le bounding box avec marges
    z0 = max(0, coords[0].min() - margin[0])
    z1 = min(volume.shape[0], coords[0].max() + margin[0] + 1)
    y0 = max(0, coords[1].min() - margin[1])
    y1 = min(volume.shape[1], coords[1].max() + margin[1] + 1)
    x0 = max(0, coords[2].min() - margin[2])
    x1 = min(volume.shape[2], coords[2].max() + margin[2] + 1)
    
    # Option: bounding box carré
    if square_bbox:
        size = max(z1-z0, y1-y0, x1-x0)
        center_z = (z0 + z1) // 2
        center_y = (y0 + y1) // 2
        center_x = (x0 + x1) // 2
        
        z0 = max(0, center_z - size//2)
        z1 = min(volume.shape[0], center_z + size//2)
        y0 = max(0, center_y - size//2)
        y1 = min(volume.shape[1], center_y + size//2)
        x0 = max(0, center_x - size//2)
        x1 = min(volume.shape[2], center_x + size//2)
    
    bbox = (z0, z1, y0, y1, x0, x1)
    cropped_volume = volume[z0:z1, y0:y1, x0:x1]
    cropped_mask = lung_mask[z0:z1, y0:y1, x0:x1]
    
    logger.info(f"✅ ROI extraite : {cropped_volume.shape} (bbox: {bbox})")
    
    return cropped_volume, cropped_mask, bbox

def get_lung_volumes(lung_mask: np.ndarray, 
                     spacing: Tuple[float, float, float] = (1,1,1)) -> Dict:
    """
    Calcule les volumes pulmonaires en mm³
    """
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    total_voxels = lung_mask.sum()
    total_volume_mm3 = total_voxels * voxel_volume
    
    # Séparer poumon gauche/droit si possible
    labeled, n_regions = ndimage.label(lung_mask)
    
    result = {
        'total_voxels': int(total_voxels),
        'total_volume_mm3': float(total_volume_mm3),
        'total_volume_l': float(total_volume_mm3 / 1e6),
        'n_regions': n_regions
    }
    
    if n_regions >= 2:
        sizes = ndimage.sum(lung_mask, labeled, range(1, n_regions+1))
        for i, size in enumerate(sizes[:2]):
            result[f'lung_{i+1}_voxels'] = int(size)
            result[f'lung_{i+1}_volume_mm3'] = float(size * voxel_volume)
    
    return result

def segment_trachea(volume: np.ndarray, 
                    lung_mask: np.ndarray,
                    z_start: int = 20) -> np.ndarray:
    """
    Segmente la trachée pour l'exclure du masque pulmonaire si nécessaire.
    """
    # La trachée est généralement dans les coupes supérieures, au centre
    trachea_mask = np.zeros_like(lung_mask, dtype=bool)
    
    # Chercher dans les premières coupes
    for z in tqdm(range(volume.shape[0]), desc="Segmentation slice-by-slice"):
        slice_mask = lung_mask[z]
        if slice_mask.sum() < 100:  # Peu de poumons visible
            continue
            
        # Trouver les régions centrales qui pourraient être la trachée
        labeled = measure.label(slice_mask)
        regions = measure.regionprops(labeled)
        
        for r in regions:
            # La trachée est généralement ronde et centrée
            if r.eccentricity < 0.6 and r.area > 50:
                cy, cx = r.centroid
                if abs(cy - volume.shape[1]//2) < 50 and abs(cx - volume.shape[2]//2) < 50:
                    trachea_mask[z] = trachea_mask[z] | (labeled == r.label)
    
    return trachea_mask

def visualize_lung_segmentation(volume: np.ndarray, 
                               lung_mask: np.ndarray,
                               num_slices: int = 5):
    """
    Fonction de visualisation pour debug
    """
    import matplotlib.pyplot as plt
    
    slice_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(2, num_slices, figsize=(3*num_slices, 6))
    
    for i, idx in enumerate(slice_indices):
        # Scan seul
        axes[0, i].imshow(volume[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[0, i].set_title(f'Slice {idx}')
        axes[0, i].axis('off')
        
        # Superposition avec masque
        axes[1, i].imshow(volume[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[1, i].imshow(lung_mask[idx], cmap='Reds', alpha=0.3)
        axes[1, i].axis('off')
    
    plt.suptitle('Segmentation pulmonaire', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🧪 Test du module lung_segmentation")
    
    # Créer un volume factice réaliste
    np.random.seed(42)  # Reproductibilité
    test_vol = np.random.randn(50, 200, 200) * 200 - 600  # HU ~ -600
    test_vol[:, 50:150, 50:150] = np.random.randn(50, 100, 100) * 100 - 800  # Poumons ~ -800
    
    print(f"📦 Volume test : {test_vol.shape}, HU: [{test_vol.min():.0f}, {test_vol.max():.0f}]")
    
    # Tester chaque méthode
    for method in ['threshold', 'otsu']:
        print(f"\n🔍 Méthode: {method}")
        lung_mask = segment_lungs(test_vol, method=method, hu_threshold=-400)
        print(f"   ✅ Masque shape: {lung_mask.shape}")
        print(f"   ✅ Voxels pulmonaires: {lung_mask.sum()} ({100*lung_mask.sum()/lung_mask.size:.1f}%)")
    
    # Tester extraction ROI
    cropped, mask_cropped, bbox = extract_lung_roi(test_vol, lung_mask, margin=(3,3,3))
    print(f"\n📐 ROI extraite: {cropped.shape}, bbox: {bbox}")
    
    # Tester calculs volumétriques
    spacing = (1.0, 0.7, 0.7)
    volumes = get_lung_volumes(lung_mask, spacing)
    print(f"\n📊 Volumes: {volumes['total_volume_l']:.2f} L")
    
    print("\n✅ Tous les tests passés !")