#loader.py

import pydicom
import numpy as np
import os
import glob
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dicom_volume(directory: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Charge les tranches DICOM, convertit en HU et récupère les métadonnées.
    
    Args:
        directory: Chemin vers le dossier contenant les fichiers .dcm
    
    Returns:
        volume: Array 3D (Z, Y, X) en unités Hounsfield
        metadata: Dictionnaire contenant:
            - spacing: (pixel_spacing_x, pixel_spacing_y)
            - slice_thickness: épaisseur de coupe
            - positions: positions des coupes
            - first_slice: premier slice (métadonnées complètes)
    """
    # Recherche des fichiers DICOM
    directory = Path(directory)
    files = sorted(glob.glob(str(directory / "*.dcm")))
    
    if not files:
        logger.warning(f"Aucun fichier DICOM trouvé dans {directory}")
        return None, None
    
    logger.info(f"Chargement de {len(files)} fichiers DICOM depuis {directory}")
    
    try:
        # Chargement de tous les slices
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception as e:
                logger.warning(f"Erreur lecture {f}: {e}")
                continue
        
        if not slices:
            logger.error("Aucun slice valide chargé")
            return None, None
        
        # Tri spatial selon la position Z (ImagePositionPatient[2])
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # Vérification de la cohérence des dimensions
        first_shape = slices[0].pixel_array.shape
        for i, s in enumerate(slices):
            if s.pixel_array.shape != first_shape:
                logger.error(f"Dimensions incohérentes: slice {i} a shape {s.pixel_array.shape}")
                return None, None
        
        # Création du volume
        volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
        
        # Conversion en unités Hounsfield
        for i, s in enumerate(slices):
            # Gestion des cas où slope/intercept ne sont pas présents
            slope = getattr(s, 'RescaleSlope', 1.0)
            intercept = getattr(s, 'RescaleIntercept', -1024)
            volume[i] = volume[i] * slope + intercept
        
        # Collecte des métadonnées importantes
        metadata = {
            'spacing': getattr(slices[0], 'PixelSpacing', [1.0, 1.0]),
            'slice_thickness': getattr(slices[0], 'SliceThickness', 1.0),
            'positions': [float(s.ImagePositionPatient[2]) for s in slices],
            'first_slice': slices[0],  # Pour accéder aux autres métadonnées si besoin
            'num_slices': len(slices),
            'modality': getattr(slices[0], 'Modality', 'Unknown'),
            'series_description': getattr(slices[0], 'SeriesDescription', 'Unknown')
        }
        
        # Calcul de l'épaisseur entre coupes si disponible
        if len(slices) > 1:
            z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
            metadata['slice_spacing'] = abs(z_positions[1] - z_positions[0])
        else:
            metadata['slice_spacing'] = metadata['slice_thickness']
        
        logger.info(f"✅ Volume chargé: shape {volume.shape}, "
                   f"spacing XY={metadata['spacing']} mm, "
                   f"spacing Z={metadata['slice_spacing']:.2f} mm")
        logger.info(f"   Plage HU: [{volume.min():.1f}, {volume.max():.1f}]")
        
        return volume, metadata
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement DICOM: {e}")
        return None, None

def load_mask(mat_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge le masque de vérité terrain (.mat) de manière robuste.
    
    Args:
        mat_path: Chemin vers le fichier .mat
    
    Returns:
        mask: Array 3D (Z, Y, X) avec valeurs binaires (0=background, 1=embolie)
        metadata: Métadonnées du masque (si disponibles)
    """
    mat_path = Path(mat_path)
    
    if not mat_path.exists():
        logger.warning(f"Masque introuvable: {mat_path}")
        return None, None
    
    try:
        data = loadmat(mat_path)
        
        # FUMPE utilise 'pixel_labels', mais on check aussi 'Label' et 'mask'
        mask = None
        possible_keys = ['pixel_labels', 'Label', 'mask', 'ground_truth', 'GT']
        
        for key in possible_keys:
            if key in data:
                mask = data[key]
                logger.info(f"Masque chargé avec clé '{key}'")
                break
        
        if mask is None:
            logger.warning(f"Clé de masque non trouvée. Clés disponibles: {list(data.keys())}")
            return None, None
        
        # S'assurer que le masque est binaire (0 ou 1)
        if mask.max() > 1:
            logger.info(f"Conversion en binaire: valeurs max={mask.max()}")
            mask = (mask > 0).astype(np.uint8)
        
        # Vérifier la dimension
        if mask.ndim == 2:
            logger.info("Masque 2D détecté, conversion en 3D")
            mask = mask[np.newaxis, ...]
        elif mask.ndim == 3:
            # S'assurer que l'ordre est (Z, Y, X)
            if mask.shape[0] > mask.shape[2]:  # Si Z est plus grand que X, probablement déjà bon
                pass
            else:
                # Sinon, on transpose (X, Y, Z) -> (Z, Y, X)
                mask = mask.transpose(2, 1, 0)
        
        # Statistiques
        n_voxels_positifs = np.sum(mask)
        pourcentage = 100 * n_voxels_positifs / mask.size
        
        logger.info(f"✅ Masque chargé: shape {mask.shape}, "
                   f"voxels positifs: {n_voxels_positifs} ({pourcentage:.3f}% du volume)")
        
        # Métadonnées additionnelles si présentes
        metadata = {}
        for key in data:
            if not key.startswith('__') and key not in possible_keys:
                if isinstance(data[key], (str, int, float, np.ndarray)):
                    metadata[key] = data[key]
        
        return mask, metadata
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du masque {mat_path}: {e}")
        return None, None

def load_patient(patient_id: str, 
                base_dir: str = "data/processed") -> Tuple[Optional[np.ndarray], 
                                                            Optional[np.ndarray], 
                                                            Optional[Dict], 
                                                            Optional[Dict]]:
    """
    Charge toutes les données pour un patient donné.
    
    Args:
        patient_id: ID du patient (ex: 'PAT001')
        base_dir: Répertoire de base des données traitées
    
    Returns:
        volume: Volume CT en HU
        mask: Masque binaire
        vol_metadata: Métadonnées du volume
        mask_metadata: Métadonnées du masque
    """
    base_dir = Path(base_dir)
    
    # Chemins
    dicom_dir = base_dir / "dicom" / patient_id
    mat_path = base_dir / "groundtruth_mat" / f"{patient_id}.mat"
    
    logger.info(f"Chargement du patient {patient_id}")
    
    # Chargement
    volume, vol_metadata = load_dicom_volume(dicom_dir)
    mask, mask_metadata = load_mask(mat_path)
    
    # Vérification de cohérence
    if volume is not None and mask is not None:
        if volume.shape != mask.shape:
            logger.warning(f"Incohérence de dimensions: volume {volume.shape}, masque {mask.shape}")
            # Tentative de correction : tronquer au plus petit
            min_z = min(volume.shape[0], mask.shape[0])
            volume = volume[:min_z]
            mask = mask[:min_z]
            logger.info(f"  -> Troncature à {min_z} slices")
    
    return volume, mask, vol_metadata, mask_metadata

def get_available_patients(base_dir: str = "data/processed") -> Dict[str, list]:
    """
    Retourne la liste des patients disponibles avec leurs données.
    
    Returns:
        dict avec clés:
            - 'with_both': patients avec volume ET masque
            - 'with_volume_only': patients avec volume seulement
            - 'with_mask_only': patients avec masque seulement
    """
    base_dir = Path(base_dir)
    
    # Patients avec volume DICOM
    dicom_dir = base_dir / "dicom"
    vol_patients = set([p.name for p in dicom_dir.iterdir() if p.is_dir()]) if dicom_dir.exists() else set()
    
    # Patients avec masque
    mat_dir = base_dir / "groundtruth_mat"
    mask_patients = set([p.stem for p in mat_dir.glob("*.mat")]) if mat_dir.exists() else set()
    
    result = {
        'with_both': sorted(vol_patients & mask_patients),
        'with_volume_only': sorted(vol_patients - mask_patients),
        'with_mask_only': sorted(mask_patients - vol_patients)
    }
    
    logger.info(f"Patients disponibles: {len(result['with_both'])} avec volume+masque, "
                f"{len(result['with_volume_only'])} volume seul, "
                f"{len(result['with_mask_only'])} masque seul")
    
    return result

if __name__ == "__main__":
    print("="*50)
    print("🧪 TEST TECHNIQUE DU LOADER (SANS GRAPHIQUES)")
    print("="*50)
    
    # On vérifie juste si les dossiers existent
    patients = get_available_patients("data/processed")
    
    if patients['with_both']:
        test_patient = patients['with_both'][0]
        print(f"\n📋 Vérification des données pour: {test_patient}")
        
        volume, mask, vol_meta, mask_meta = load_patient(test_patient, "data/processed")
        
        if volume is not None and mask is not None:
            # On valide par les chiffres, pas par l'image
            print(f"✅ Volume shape : {volume.shape}")
            print(f"✅ Masque shape : {mask.shape}")
            print(f"✅ Somme du masque (embolies) : {np.sum(mask)} voxels")
            
            # Vérification de sécurité
            assert volume.shape == mask.shape, "ERREUR: Dimensions incohérentes !"
            print("\n🚀 Test réussi : Les données sont techniquement valides.")
    else:
        print("❌ Aucun patient trouvé pour le test.")



''' 

#loader.py

import pydicom
import numpy as np
import os
import glob
from scipy.io import loadmat

def load_dicom_volume(directory):
    """Charge les tranches DICOM, convertit en HU et récupère la résolution."""
    files = glob.glob(os.path.join(directory, "*.dcm"))
    if not files: 
        print(f"⚠️ Aucun DICOM trouvé dans {directory}")
        return None, None
    
    # Chargement et tri spatial
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Création du volume en float32 pour garder la précision après calcul
    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    
    # Application de la transformation Hounsfield
    for i in range(len(slices)):
        intercept = slices[i].RescaleIntercept
        slope = slices[i].RescaleSlope
        volume[i] = volume[i] * slope + intercept
        
    # Retourne le volume ET l'espacement des pixels (dx, dy)
    return volume, slices[0].PixelSpacing

def load_mask(mat_path):
    """Charge le masque de vérité terrain (.mat) de manière robuste."""
    if not os.path.exists(mat_path):
        print(f"⚠️ Masque introuvable : {mat_path}")
        return None
        
    data = loadmat(mat_path)
    # Cherche la clé 'pixel_labels' (FUMPE standard) ou 'Label'
    mask = data.get('pixel_labels', data.get('Label'))
    
    if mask is None:
        print(f"⚠️ Clé de masque non trouvée dans {mat_path}. Clés : {list(data.keys())}")
        
    return mask

if __name__ == "__main__":
    # Test local
    test_path = r"data\raw\CT_scans_01\PAT001" # Ajuste selon tes dossiers
    vol, spacing = load_dicom_volume(test_path)
    if vol is not None:
        print(f"✅ Volume chargé : {vol.shape} | Résolution : {spacing}")
        print(f"Plage HU : [{vol.min()} , {vol.max()}]")   
'''