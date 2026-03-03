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
        
        # 🔍 Liste étendue des clés possibles (FUMPE + variantes)
        # ✅ Ajout de 'Mask' avec majuscule (spécifique à FUMPE)
        possible_keys = [
            'pixel_labels', 'Label', 'mask', 'Mask',  # ← 'Mask' ajouté !
            'ground_truth', 'GT', 'seg', 'segmentation', 
            'annotation', 'truth', 'binary_mask'
        ]
        mask = None
        
        for key in possible_keys:
            if key in data:
                mask = data[key]
                logger.info(f"✅ Masque chargé avec clé '{key}'")
                break
        
        # Fallback : prendre la première clé valide (tableau numpy)
        if mask is None:
            for key in data.keys():
                if not key.startswith('__'):
                    candidate = data[key]
                    if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                        mask = candidate
                        logger.info(f"✅ Masque chargé avec clé fallback '{key}'")
                        break
        
        if mask is None: 
            # ✅ DEBUG : Afficher les clés disponibles pour aider au débogage
            logger.warning(f"⚠️ Aucune clé de masque trouvée. Clés disponibles: {list(data.keys())}")
            return None, None
        
        # Binarisation (0: fond, 1: embolie)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        # RÉALIGNEMENT ROBUSTE : MATLAB → (Z, Y, X)
        if mask.ndim == 3:
            if mask.shape[-1] < mask.shape[0]:  # Si Z est en dernière position
                mask = mask.transpose(2, 1, 0)
                logger.info(f"Transpose appliqué: → {mask.shape}")
            
        logger.info(f"✅ Masque chargé : {mask.shape}")
        return mask, {}
    
    except Exception as e:
        logger.error(f"❌ Erreur Masque sur {mat_path.name}: {e}")
        return None, None


def load_npy_volume(volume_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge un volume prétraité au format .npy
    """
    if volume_path is None:
        return None, None
    volume_path = Path(volume_path)
    
    if not volume_path.exists():
        logger.warning(f"⚠️ Fichier .npy introuvable : {volume_path}")
        return None, None
    
    try:
        volume = np.load(volume_path)
        logger.info(f"✅ Volume .npy chargé : {volume.shape}")
        return volume, {'format': 'npy', 'path': str(volume_path)}
    except Exception as e:
        logger.error(f"❌ Erreur chargement .npy {volume_path}: {e}")
        return None, None


def load_npy_mask(mask_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge un masque prétraité au format .npy
    """
    if mask_path is None:
        return None, None
    mask_path = Path(mask_path)
    
    if not mask_path.exists():
        logger.warning(f"⚠️ Fichier .npy introuvable : {mask_path}")
        return None, None
    
    try:
        mask = np.load(mask_path)
        logger.info(f"✅ Masque .npy chargé : {mask.shape}")
        return mask, {'format': 'npy', 'path': str(mask_path)}
    except Exception as e:
        logger.error(f"❌ Erreur chargement .npy {mask_path}: {e}")
        return None, None


def get_available_patients(base_dir: str = "data") -> Dict[str, Any]:
    """
    Scanne les dossiers et crée un index des chemins pour chaque patient.
    Bascule automatiquement de 'processed' vers 'raw' si nécessaire.
    """
    base_path = Path(base_dir)
    
    # 1. Détermination du dossier source
    processed_check = base_path / "processed"
    raw_check = base_path / "raw"
    
    # ✅ CORRECTION CRITIQUE : On vérifie qu'il y a VRAIMENT des patients dans processed
    use_processed = False
    
    if processed_check.exists():
        # Cherche spécifiquement des dossiers PAT* dans processed
        processed_patients = list(processed_check.glob("PAT*"))
        if len(processed_patients) > 0:
            use_processed = True
            logger.info("📂 Mode : Données TRAITÉES (processed)")
    
    if not use_processed:
        if raw_check.exists():
            source_dir = raw_check
            logger.info("📂 Mode : Données BRUTES (raw) - Exploration")
        else:
            logger.error("❌ Aucun dossier 'raw' ou 'processed' avec patients trouvé !")
            return {'with_both': [], 'vol_paths': {}, 'mask_paths': {}}
    else:
        source_dir = processed_check

    # 2. Indexation des Scans
    vol_paths = {}
    
    # Structure RAW : CT_scans_*/PAT*
    for scan_folder in source_dir.glob("CT_scans_*"):
        for p_dir in scan_folder.iterdir():
            if p_dir.is_dir() and p_dir.name.startswith("PAT"):
                raw_id = p_dir.name
                if raw_id.startswith("PAT"):
                    num_part = raw_id[3:]
                    normalized_id = f"PAT{int(num_part):03d}"
                else:
                    normalized_id = raw_id
                vol_paths[normalized_id] = p_dir
    
    # Structure PROCESSED : PAT* directement
    if use_processed:
        for p_dir in source_dir.glob("PAT*"):
            if p_dir.is_dir():
                raw_id = p_dir.name
                if raw_id.startswith("PAT"):
                    num_part = raw_id[3:]
                    normalized_id = f"PAT{int(num_part):03d}"
                else:
                    normalized_id = raw_id
                vol_paths[normalized_id] = p_dir

    logger.info(f"📁 Volumes trouvés : {len(vol_paths)} patients")

    # 3. Indexation des Masques
    mask_paths = {}
    
    if use_processed:
        # Dans processed, les masques sont déjà dans les dossiers PAT*/mask.npy
        for p_id, p_path in vol_paths.items():
            mask_npy = p_path / "mask.npy"
            if mask_npy.exists():
                mask_paths[p_id] = mask_npy
    else:
        # Dans raw, chercher dans GroundTruth
        gt_path = source_dir / "GroundTruth"
        if not gt_path.exists():
            gt_path = source_dir / "GroundTruth" / "GroundTruth"
        
        if gt_path.exists():
            for m_file in gt_path.glob("*.mat"):
                name = m_file.stem
                if name.startswith("PAT"):
                    num_part = name[3:]
                    key = f"PAT{int(num_part):03d}"
                else:
                    key = f"PAT{int(name):03d}"
                mask_paths[key] = m_file
            logger.info(f"📁 Masques trouvés dans {gt_path}: {len(mask_paths)} fichiers")
        else:
            logger.warning(f"⚠️ Dossier GroundTruth introuvable : {gt_path}")

    # 4. Intersection
    common_ids = sorted(set(vol_paths.keys()) & set(mask_paths.keys()))
    
    logger.info(f"📊 Inventaire : {len(common_ids)} patients avec Volume + Masque.")
    
    return {
        'with_both': common_ids,
        'vol_paths': vol_paths,
        'mask_paths': mask_paths,
        'use_processed': use_processed  # ← Nouveau flag important !
    }


def load_patient_data(patient_id: str, inventory: Dict):
    """
    Charge le volume et le masque en utilisant l'inventaire.
    Gère automatiquement RAW (.dcm + .mat) vs PROCESSED (.npy)
    """
    vol_path = inventory['vol_paths'].get(patient_id)
    mask_path = inventory['mask_paths'].get(patient_id)
    use_processed = inventory.get('use_processed', False)
    
    # ✅ CORRECTION CRITIQUE : Utiliser les bonnes fonctions selon le format
    if use_processed:
        # Mode PROCESSED : charger les .npy
        volume, v_meta = load_npy_volume(vol_path / "volume.npy" if isinstance(vol_path, Path) else vol_path)
        mask, m_meta = load_npy_mask(mask_path)
    else:
        # Mode RAW : charger DICOM + .mat
        volume, v_meta = load_dicom_volume(vol_path)
        mask, m_meta = load_mask_mat(mask_path)
    
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
        print(f"📁 Mode: {'PROCESSED (.npy)' if inventory['use_processed'] else 'RAW (.dcm + .mat)'}")
        
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