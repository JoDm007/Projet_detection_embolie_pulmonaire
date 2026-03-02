import pydicom
import numpy as np
import glob
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

# 1. Configuration du logging pour un suivi professionnel
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dicom_volume(directory: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Charge les fichiers DICOM d'un dossier, les convertit en Unités Hounsfield (HU)
    et extrait les métadonnées géométriques.
    """
    if directory is None: 
        return None, None
    directory = Path(directory)
    
    # Recherche récursive (trouve les .dcm même dans des sous-dossiers)
    files = sorted(glob.glob(str(directory / "**/*.dcm"), recursive=True))

    if not files:
        logger.warning(f"⚠️ Aucun DICOM trouvé dans {directory}")
        return None, None

    try:
        # Chargement des tranches
        slices = [pydicom.dcmread(f) for f in files]
        
        # TRI SPATIAL : Crucial pour reconstruire le corps humain dans le bon sens
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Création du volume 3D (Z, Y, X)
        volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)

        # CONVERSION HOUNSFIELD (HU) : Normalise les valeurs selon le scanner
        for i, s in enumerate(slices):
            slope = getattr(s, 'RescaleSlope', 1.0)
            intercept = getattr(s, 'RescaleIntercept', -1024)
            volume[i] = volume[i] * slope + intercept

        # Extraction des métadonnées pour le futur ré-échantillonnage (Resampling)
        metadata = {
            'spacing': getattr(slices[0], 'PixelSpacing', [1.0, 1.0]),
            'slice_thickness': getattr(slices[0], 'SliceThickness', 1.0),
            'slice_spacing': abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]) if len(slices) > 1 else 1.0,
            'origin': slices[0].ImagePositionPatient
        }
        
        logger.info(f"✅ Volume chargé : {volume.shape} de {directory.name}")
        return volume, metadata

    except Exception as e:
        logger.error(f"❌ Erreur DICOM sur {directory}: {e}")
        return None, None


def load_mask(mat_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge le masque (.mat), le binarise et s'assure de l'orientation (Z, Y, X).
    """
    if mat_path is None: 
        return None, None
    mat_path = Path(mat_path)

    if not mat_path.exists():
        logger.warning(f"⚠️ Masque introuvable : {mat_path}")
        return None, None

    try:
        data = loadmat(mat_path)
        # Clés possibles dans le dataset FUMPE
        mask = data.get('pixel_labels', data.get('Label', data.get('mask')))
        
        if mask is None: 
            logger.warning(f"⚠️ Clé de masque non trouvée dans {mat_path}")
            return None, None
        
        # Binarisation (0: fond, 1: embolie)
        mask = (mask > 0).astype(np.uint8)
        
        # RÉALIGNEMENT : MATLAB inverse souvent les axes (X, Y, Z) -> (Z, Y, X)
        if mask.ndim == 3 and mask.shape[0] < mask.shape[2]:
            mask = mask.transpose(2, 1, 0)
            
        logger.info(f"✅ Masque chargé : {mask.shape}")
        return mask, {}
    
    except Exception as e:
        logger.error(f"❌ Erreur Masque sur {mat_path.name}: {e}")
        return None, None


def get_available_patients(base_dir: str = "data") -> Dict[str, Any]:
    """
    Scanne les dossiers et crée un index des chemins pour chaque patient.
    Bascule automatiquement de 'processed' vers 'raw' si nécessaire.
    """
    base_path = Path(base_dir)
    
    # 1. Détermination du dossier source (Priorité au processed, sinon raw)
    processed_check = base_path / "processed"
    if processed_check.exists() and any(processed_check.iterdir()):
        source_dir = processed_check
        logger.info("📂 Mode : Données TRAITÉES (processed)")
    else:
        source_dir = base_path / "raw"
        logger.info("📂 Mode : Données BRUTES (raw) - Exploration")

    # 2. Indexation des Scans (CT_scans_01, CT_scans_02, CT_scans_03)
    vol_paths = {}
    for scan_folder in source_dir.glob("CT_scans_*"):  # ✅ underscore
        for p_dir in scan_folder.iterdir():
            if p_dir.is_dir() and p_dir.name.startswith("PAT"):
                # Normalisation : PAT001 → PAT001 (déjà 3 chiffres)
                raw_id = p_dir.name
                if raw_id.startswith("PAT"):
                    num_part = raw_id[3:]
                    normalized_id = f"PAT{int(num_part):03d}"  # Force 3 chiffres
                else:
                    normalized_id = raw_id
                vol_paths[normalized_id] = p_dir

    logger.info(f"📁 Volumes trouvés : {len(vol_paths)} patients")

    # 3. Indexation des Masques (GroundTruth SIMPLE)
    mask_paths = {}
    gt_path = source_dir / "GroundTruth"  # ✅ Pas de double dossier !
    
    if gt_path.exists():
        for m_file in gt_path.glob("*.mat"):
            name = m_file.stem  # Ex: 'PAT001'
            # Harmonisation : force le format PATxxx avec 3 chiffres
            if name.startswith("PAT"):
                num_part = name[3:]
                key = f"PAT{int(num_part):03d}"
            else:
                key = f"PAT{int(name):03d}"
            mask_paths[key] = m_file
        logger.info(f"📁 Masques trouvés dans {gt_path}: {len(mask_paths)} fichiers")
    else:
        logger.warning(f"⚠️ Dossier GroundTruth introuvable : {gt_path}")

    # 4. Intersection pour trouver les patients complets
    common_ids = sorted(set(vol_paths.keys()) & set(mask_paths.keys()))
    
    logger.info(f"📊 Inventaire : {len(common_ids)} patients avec Volume + Masque.")
    
    return {
        'with_both': common_ids,
        'vol_paths': vol_paths,
        'mask_paths': mask_paths
    }


def load_patient_data(patient_id: str, inventory: Dict):
    """
    Charge le volume et le masque en utilisant l'inventaire.
    Gère aussi les petites erreurs de dimensions (Z).
    """
    vol_path = inventory['vol_paths'].get(patient_id)
    mask_path = inventory['mask_paths'].get(patient_id)
    
    volume, v_meta = load_dicom_volume(vol_path)
    mask, _ = load_mask(mask_path)
    
    # Sécurité : recalage si une coupe manque dans l'un des deux fichiers
    if volume is not None and mask is not None:
        if volume.shape[0] != mask.shape[0]:
            logger.warning(f"⚠️ Taille Z différente pour {patient_id}. Ajustement...")
            min_z = min(volume.shape[0], mask.shape[0])
            volume, mask = volume[:min_z], mask[:min_z]
            
    return volume, mask, v_meta


if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU LOADER (SANS GRAPHIQUES)")
    print("="*60)
    
    # Test avec données brutes
    inventory = get_available_patients("data")
    
    if inventory['with_both']:
        print(f"\n✅ {len(inventory['with_both'])} patients trouvés avec Volume + Masque !")
        test_patient = inventory['with_both'][0]
        print(f"📋 Test avec patient: {test_patient}")
        
        volume, mask, meta = load_patient_data(test_patient, inventory)
        
        if volume is not None and mask is not None:
            print(f"\n✅ Test réussi !")
            print(f"   Volume shape: {volume.shape}")
            print(f"   Masque shape: {mask.shape}")
            print(f"   Voxels embolie: {np.sum(mask)}")
        else:
            print("❌ Échec du chargement des données.")
    else:
        print("❌ Aucun patient complet trouvé.")