# src/2_classical_detection/morphology.py

import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import logging
from typing import Tuple, Optional, List
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_candidates(candidates: np.ndarray,
                    min_volume: int = 10,
                    max_volume: int = 10000,
                    min_sphericity: float = 0.1,
                    remove_holes: bool = True,
                    smooth_boundaries: bool = False,
                    closing_radius: int = 1) -> np.ndarray:
    """
    Nettoyage des régions candidates par opérations morphologiques avancées.
    
    Args:
        candidates: Masque binaire 3D des candidats
        min_volume: Volume minimal en voxels pour conserver une région
        max_volume: Volume maximal (exclure les très grandes régions = artères)
        min_sphericity: Sphéricité minimale (0 = très allongé, 1 = sphère parfaite)
        remove_holes: Remplir les petits trous dans les régions
        smooth_boundaries: Lisser les contours des régions
        closing_radius: Rayon pour l'opération de fermeture
    
    Returns:
        cleaned: Masque binaire nettoyé
    """
    logger.info("🧹 Nettoyage morphologique des candidats...")
    
    if candidates.sum() == 0:
        logger.info("   Aucun candidat à nettoyer")
        return candidates
    
    original_voxels = candidates.sum()
    original_regions = ndimage.label(candidates.astype(int))[1]
    
    # 1. Ouverture pour éliminer le bruit (érosion + dilatation)
    structure = ndimage.generate_binary_structure(3, 1)
    cleaned = morphology.binary_opening(candidates, footprint=structure)
    
    # 2. Fermeture optionnelle pour combler les petits trous
    if closing_radius > 0:
        closing_struct = morphology.ball(closing_radius)
        cleaned = morphology.binary_closing(cleaned, footprint=closing_struct)
    
    # 3. Remplir les trous si demandé (attention : peut prendre du temps en 3D)
    if remove_holes:
        cleaned = fill_holes_3d(cleaned, max_hole_size=min_volume)
    
    # 4. Lisser les frontières (via une petite érosion/dilatation)
    if smooth_boundaries:
        cleaned = morphology.binary_erosion(cleaned, structure)
        cleaned = morphology.binary_dilation(cleaned, structure)
    
    # 5. Étiquetage des composantes connexes
    labeled, num_regions = ndimage.label(cleaned.astype(int), structure=structure)
    logger.info(f"   {num_regions} régions connexes détectées après nettoyage")
    
    if num_regions == 0:
        return np.zeros_like(candidates, dtype=bool)
    
    # 6. Filtrage par volume et forme
    final_mask = np.zeros_like(candidates, dtype=bool)
    kept_regions = 0
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled == region_id)
        volume = region_mask.sum()
        
        # Filtre par volume
        if volume < min_volume or volume > max_volume:
            continue
        
        # Filtre par sphéricité
        sphericity = compute_sphericity(region_mask)
        if sphericity < min_sphericity:
            continue
        
        # Filtre optionnel par d'autres métriques
        if not passes_additional_filters(region_mask):
            continue
        
        # Conserver la région
        final_mask[region_mask] = True
        kept_regions += 1
    
    # Statistiques finales
    final_voxels = final_mask.sum()
    final_regions = kept_regions
    
    logger.info(f"✅ Résultat final:")
    logger.info(f"   Voxels: {original_voxels} -> {final_voxels} ({100*final_voxels/original_voxels:.1f}%)")
    logger.info(f"   Régions: {original_regions} -> {final_regions}")
    
    return final_mask

def compute_sphericity(region_mask: np.ndarray) -> float:
    coords = np.where(region_mask)
    
    if len(coords[0]) < 10:
        return 0.0
    
    # Méthode 1: Ratio des axes principaux (toujours disponible)
    extents = [c.max() - c.min() for c in coords]
    if min(extents) == 0:
        return 0.0
    sphericity_extent = min(extents) / max(extents)
    
    # Méthode 2: Volume vs surface (seulement si région assez grande)
    sphericity = sphericity_extent  # Default fallback
    if region_mask.sum() >= 50:  # Seuil minimal pour marching_cubes
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Supprimer les warnings skimage
                verts, faces, _, _ = measure.marching_cubes(
                    region_mask.astype(float), level=0.5, allow_degenerate=False
                )
                if len(faces) > 0:
                    surface_area = measure.mesh_surface_area(verts, faces)
                    volume = region_mask.sum()
                    sphericity_surface = (np.pi**(1/3) * (6*volume)**(2/3)) / (surface_area + 1e-6)
                    sphericity = (sphericity_extent + sphericity_surface) / 2
        except Exception:
            # Fallback sur la méthode simple si marching cubes échoue
            sphericity = sphericity_extent
    
    return float(sphericity)

def compute_compactness(region_mask: np.ndarray) -> float:
    """
    Calcule la compacité (inverse de l'étendue)
    Compactness = V / (A^(3/2)) pour 3D
    """
    volume = region_mask.sum()
    
    # Estimation de la surface via gradient
    from scipy.ndimage import sobel
    
    # Gradient pour estimer la surface
    grad_x = sobel(region_mask.astype(float), axis=0)
    grad_y = sobel(region_mask.astype(float), axis=1)
    grad_z = sobel(region_mask.astype(float), axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    surface = (grad_mag > 0).sum()
    
    if surface > 0:
        compactness = volume / (surface ** 1.5 + 1e-6)
        return float(compactness)
    return 0.0

def passes_additional_filters(region_mask: np.ndarray) -> bool:
    """
    Filtres supplémentaires : rejette les régions touchant les bords XY et Z.
    Marge réduite à 1 voxel : les poumons et vaisseaux périphériques peuvent
    légitimement être proches des bords du FOV scanner (bord Y ou X à 511/0).
    """
    margin = 1  # CORRECTION : 3 était trop restrictif, rejetait les vaisseaux périphériques
    shape = region_mask.shape

    # Bord Z (coupes sup/inf)
    if region_mask[:margin].any() or region_mask[-margin:].any():
        return False

    # Bord Y
    if region_mask[:, :margin].any() or region_mask[:, -margin:].any():
        return False

    # Bord X
    if region_mask[:, :, :margin].any() or region_mask[:, :, -margin:].any():
        return False

    return True

def fill_holes_3d(mask: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
    # Skip si le masque est trop grand (optimisation)
    if mask.sum() > 100000:
        logger.debug("   Skip fill_holes: région trop grande")
        return mask
    """
    Remplissage des trous dans les régions 3D
    """
    # Inverser le masque pour trouver les trous
    inverted = ~mask
    
    # Étiqueter les composantes de l'inverse
    structure = ndimage.generate_binary_structure(3, 1)
    labeled_inv, n_holes = ndimage.label(inverted, structure=structure)
    
    # Le fond (extérieur) est la plus grande composante
    sizes = ndimage.sum(inverted, labeled_inv, range(1, n_holes + 1))
    background_label = np.argmax(sizes) + 1
    
    # Remplir les trous (composantes qui ne sont pas le fond)
    result = mask.copy()
    for hole_id in range(1, n_holes + 1):
        if hole_id != background_label:
            hole_mask = (labeled_inv == hole_id)
            hole_size = hole_mask.sum()
            if hole_size <= max_hole_size:  # Ne remplir que les petits trous
                result[hole_mask] = True
    
    return result

def separate_touching_objects(mask: np.ndarray,
                             min_distance: int = 3) -> np.ndarray:
    """
    Sépare les objets qui se touchent par érosion puis dilatation conditionnelle
    """

    # Érosion forte pour séparer
    structure = morphology.ball(min_distance // 2)
    eroded = morphology.binary_erosion(mask, structure)
    
    if eroded.sum() == 0:
        return mask
    
    # Dilatation conditionnelle pour reconstruire sans reconnecter
    from skimage.morphology import reconstruction
    
    # Reconstruction géodésique
    reconstructed = reconstruction(eroded.astype(float), 
                                  mask.astype(float), 
                                  method='dilation') > 0
    
    return reconstructed

def morphological_gradient(mask: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Calcule le gradient morphologique (frontières des régions)
    """
    structure = morphology.ball(size)
    dilated = morphology.binary_dilation(mask, structure)
    eroded = morphology.binary_erosion(mask, structure)
    
    gradient = dilated & ~eroded
    return gradient

def skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    """
    Squelettisation 3D (utile pour analyser la forme des vaisseaux)
    """
    from skimage.morphology import skeletonize_3d
    return skeletonize_3d(mask.astype(np.uint8))

def analyze_regions(mask: np.ndarray) -> dict:
    """
    Analyse complète des régions pour diagnostic
    """
    labeled, n_regions = ndimage.label(mask.astype(int))
    
    if n_regions == 0:
        return {"n_regions": 0}
    
    results = {
        "n_regions": n_regions,
        "total_voxels": int(mask.sum()),
        "regions": []
    }
    
    for region_id in range(1, n_regions + 1):
        region = (labeled == region_id)
        volume = region.sum()
        
        # Caractéristiques de base
        region_info = {
            "id": region_id,
            "volume_voxels": int(volume),
            "sphericity": float(compute_sphericity(region)),
            "compactness": float(compute_compactness(region))
        }
        
        # Position
        coords = np.where(region)
        region_info["centroid"] = [float(c.mean()) for c in coords]
        region_info["bounds"] = [
            [int(c.min()), int(c.max())] for c in coords
        ]
        
        results["regions"].append(region_info)
    
    return results

def morphological_pipeline(candidates: np.ndarray,
                          operations: List[str] = ['opening', 'closing', 'remove_small'],
                          **kwargs) -> np.ndarray:
    """
    Pipeline personnalisable d'opérations morphologiques
    """
    result = candidates.copy()
    
    for op in operations:
        if op == 'opening':
            result = morphology.binary_opening(result)
            logger.info("   ✓ Opening")
            
        elif op == 'closing':
            result = morphology.binary_closing(result)
            logger.info("   ✓ Closing")
            
        elif op == 'dilation':
            radius = kwargs.get('dilation_radius', 1)
            result = morphology.binary_dilation(result, morphology.ball(radius))
            logger.info(f"   ✓ Dilation (r={radius})")
            
        elif op == 'erosion':
            radius = kwargs.get('erosion_radius', 1)
            result = morphology.binary_erosion(result, morphology.ball(radius))
            logger.info(f"   ✓ Erosion (r={radius})")
            
        elif op == 'remove_small':
            min_size = kwargs.get('min_size', 10)
            result = morphology.remove_small_objects(result, min_size=min_size)
            logger.info(f"   ✓ Remove small (<{min_size})")
            
        elif op == 'fill_holes':
            max_hole = kwargs.get('max_hole', 100)
            result = fill_holes_3d(result, max_hole_size=max_hole)
            logger.info(f"   ✓ Fill holes")
    
    return result

if __name__ == "__main__":
    print("🧪 Test du module morphology")
    
    # Créer un volume vide
    test_mask = np.zeros((50, 50, 50), dtype=bool)
    
    # Créer des formes 3D réalistes
    z, y, x = np.ogrid[0:50, 0:50, 0:50]
    
    # Sphère (simule une embolie ronde)
    sphere = (z-15)**2 + (y-15)**2 + (x-15)**2 <= 20
    test_mask[sphere] = True
    
    # Cylindre (simule un vaisseau avec embolie)
    test_mask[30:40, 25:35, 25:28] = True
    
    # Bruit (artefacts à éliminer)
    test_mask[45, 45, 45] = True
    test_mask[46, 46, 46] = True
    
    print(f"\n📊 Avant nettoyage:")
    print(f"   Voxels: {test_mask.sum()}")
    print(f"   Régions: {ndimage.label(test_mask.astype(int))[1]}")
    
    # Tester le nettoyage
    cleaned = clean_candidates(
        test_mask, 
        min_volume=10, 
        min_sphericity=0.3,
        remove_holes=True
    )
    
    print(f"\n📊 Après nettoyage:")
    print(f"   Voxels: {cleaned.sum()}")
    print(f"   Régions: {ndimage.label(cleaned.astype(int))[1]}")
    
    # Analyser
    analysis = analyze_regions(cleaned)
    print(f"\n📋 Analyse des régions conservées:")
    for region in analysis.get('regions', []):
        print(f"   • Région {region['id']}: {region['volume_voxels']} voxels, "
              f"sphéricité={region['sphericity']:.2f}")
    
    # ✅ Validation
    assert cleaned.sum() <= test_mask.sum(), "Nettoyage ne doit pas ajouter de voxels"
    assert ndimage.label(cleaned.astype(int))[1] <= 2, "Devrait garder sphère + cylindre"
    print("\n✅ Tous les tests passés !")