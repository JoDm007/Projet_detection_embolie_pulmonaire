# src/O5_deep_learning/data_generator.py
"""
Générateur de données 3D pour l'entraînement des modèles deep learning (Objectif 5)

Compatible avec :
- Les splits créés par split_data.py
- Le loader.py existant pour charger les volumes
- La normalisation déjà implémentée
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from tensorflow import keras
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import Sequence

# Import des modules du projet
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.O1_preprocessing.loader import load_patient
from src.O1_preprocessing.normalisation import apply_windowing

logger = logging.getLogger(__name__)


class VolumeDataGenerator(Sequence):
    """
    Générateur de données 3D pour TensorFlow/Keras.
    
    Charge les volumes à la volée pour éviter de saturer la RAM.
    Compatible avec les splits Train/Val/Test.
    
    Args:
        split_json: Chemin vers le fichier JSON des splits (créé par split_data.py)
        split_type: 'train', 'val', ou 'test'
        batch_size: Taille des batches
        target_size: Tuple (height, width, depth) pour le resize
        window_name: Type de fenêtrage ('pulmonary', 'mediastinal', 'soft_tissue')
        shuffle: Mélanger les échantillons à chaque époque
        augment: Appliquer la data augmentation (train uniquement)
        normalize_hu: Normaliser les HU entre 0 et 1 (avec clipping)
    """
    
    def __init__(
        self,
        split_json: str,
        split_type: str = 'train',
        batch_size: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 64),  # (height, width, depth)
        window_name: str = 'pulmonary',
        shuffle: bool = True,
        augment: bool = False,
        normalize_hu: bool = True,
        hu_min: int = -1024,
        hu_max: int = 600
    ):
        self.split_json = Path(split_json)
        self.split_type = split_type
        self.batch_size = batch_size
        self.target_size = target_size
        self.window_name = window_name
        self.shuffle = shuffle
        self.augment = augment
        self.normalize_hu = normalize_hu
        self.hu_min = hu_min
        self.hu_max = hu_max
        
        # Charger les splits
        self.patients = self._load_split()
        logger.info(f"✅ Générateur initialisé: {split_type} - {len(self.patients)} patients")
        
        self.on_epoch_end()
    
    def _load_split(self) -> List[str]:
        """Charge la liste des patients pour le split demandé"""
        with open(self.split_json, 'r') as f:
            splits = json.load(f)
        
        patients = splits.get(self.split_type, [])
        logger.info(f"📊 {self.split_type}: {len(patients)} patients")
        
        return patients
    
    def on_epoch_end(self):
        """Mélange les indices à la fin de chaque époque"""
        self.indices = np.arange(len(self.patients))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Nombre de batches par époque"""
        return int(np.ceil(len(self.patients) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Génère un batch d'indices `index`"""
        # Récupérer les indices du batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_patients = [self.patients[i] for i in batch_indices]
        
        # Charger les données du batch
        X_batch = []
        y_batch = []
        
        for patient_id in batch_patients:
            volume, mask, vol_meta, mask_meta = load_patient(patient_id)
            
            if volume is not None and mask is not None:
                # 1. Appliquer le fenêtrage
                volume_windowed = apply_windowing(volume, self.window_name)
                
                # 2. Normaliser entre 0 et 1 (optionnel)
                if self.normalize_hu:
                    volume_windowed = np.clip(volume_windowed, self.hu_min, self.hu_max)
                    volume_windowed = (volume_windowed - self.hu_min) / (self.hu_max - self.hu_min)
                
                # 3. Redimensionner (si nécessaire)
                if volume_windowed.shape != self.target_size:
                    volume_windowed = self._resize_volume(volume_windowed)
                    mask = self._resize_mask(mask)
                
                # 4. Vérifier la cohérence des dimensions
                if volume_windowed.shape != mask.shape:
                    logger.warning(f"Incohérence dimensions pour {patient_id}: "
                                  f"vol={volume_windowed.shape}, mask={mask.shape}")
                    continue
                
                X_batch.append(volume_windowed)
                y_batch.append(mask)
        
        # Convertir en numpy arrays
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Ajouter une dimension channel (pour CNN)
        # Format: (batch, height, width, depth, channels)
        X_batch = np.expand_dims(X_batch, axis=-1)  # (B, H, W, D, 1)
        y_batch = np.expand_dims(y_batch, axis=-1)
        
        # Appliquer data augmentation (si demandé)
        if self.augment:
            X_batch, y_batch = self._augment_batch(X_batch, y_batch)
        
        return X_batch, y_batch
    
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Redimensionne un volume 3D à la taille cible"""
        from scipy.ndimage import zoom
        
        current_shape = volume.shape
        zoom_factors = [
            self.target_size[0] / current_shape[0],
            self.target_size[1] / current_shape[1],
            self.target_size[2] / current_shape[2]
        ]
        
        volume_resized = zoom(volume, zoom_factors, order=1)  # Interpolation linéaire
        return volume_resized
    
    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Redimensionne un masque 3D (nearest neighbor pour garder binaire)"""
        from scipy.ndimage import zoom
        
        current_shape = mask.shape
        zoom_factors = [
            self.target_size[0] / current_shape[0],
            self.target_size[1] / current_shape[1],
            self.target_size[2] / current_shape[2]
        ]
        
        mask_resized = zoom(mask, zoom_factors, order=0)  # Nearest neighbor
        return (mask_resized > 0.5).astype(np.float32)
    
    def _augment_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applique des augmentations 3D aléatoires"""
        # Rotation aléatoire (90, 180, 270 degrés) autour de l'axe Z
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            X = np.rot90(X, k, axes=(1, 2))
            y = np.rot90(y, k, axes=(1, 2))
        
        # Flip horizontal aléatoire
        if np.random.random() > 0.5:
            X = np.flip(X, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        
        # Bruit gaussien (sur X uniquement)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.05, X.shape)
            X = X + noise
        
        return X, y


class CandidateDataGenerator(Sequence):
    """
    Générateur pour la classification patch-based (Objectif 5.2).
    Extrait des patches 3D autour des candidats d'embolie.
    
    Compatible avec la sortie de l'Objectif 2 (candidate detection).
    """
    
    def __init__(
        self,
        features_csv: str,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ):
        self.features_csv = Path(features_csv)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Charger les features (qui contiennent les coordonnées des candidats)
        self.df = pd.read_csv(features_csv)
        
        # Identifier les colonnes de coordonnées
        self.has_coords = all(c in self.df.columns for c in ['center_x', 'center_y', 'center_z'])
        
        if not self.has_coords:
            logger.warning("⚠️ Aucune coordonnée trouvée, générateur patch non fonctionnel")
        
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.df.iloc[batch_indices]
        
        X_batch = []
        y_batch = []
        
        for _, row in batch_data.iterrows():
            # TODO: Charger le volume et extraire le patch autour des coordonnées
            # À implémenter avec un loader spécifique
            pass
        
        return np.array(X_batch), np.array(y_batch)


# Fonction utilitaire pour créer les générateurs Train/Val/Test
def create_generators(
    split_json: str,
    batch_size: int = 4,
    target_size: Tuple[int, int, int] = (128, 128, 64),
    window_name: str = 'pulmonary',
    augment_train: bool = True
) -> Tuple[VolumeDataGenerator, VolumeDataGenerator, VolumeDataGenerator]:
    """
    Crée les trois générateurs (Train, Val, Test) pour l'entraînement.
    
    Returns:
        train_gen, val_gen, test_gen
    """
    train_gen = VolumeDataGenerator(
        split_json=split_json,
        split_type='train',
        batch_size=batch_size,
        target_size=target_size,
        window_name=window_name,
        shuffle=True,
        augment=augment_train
    )
    
    val_gen = VolumeDataGenerator(
        split_json=split_json,
        split_type='val',
        batch_size=batch_size,
        target_size=target_size,
        window_name=window_name,
        shuffle=False,
        augment=False
    )
    
    test_gen = VolumeDataGenerator(
        split_json=split_json,
        split_type='test',
        batch_size=batch_size,
        target_size=target_size,
        window_name=window_name,
        shuffle=False,
        augment=False
    )
    
    return train_gen, val_gen, test_gen


# Test du module
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DU DATA GENERATOR")
    print("=" * 60)
    
    # Chemins à ajuster selon ta structure
    SPLIT_JSON = "data/splits/train_val_test.json"
    
    # Vérifier si le fichier existe
    if not Path(SPLIT_JSON).exists():
        print(f"⚠️ Fichier {SPLIT_JSON} non trouvé")
        print("Crée d'abord les splits avec src/O1_preprocessing/split_data.py")
    else:
        # Créer les générateurs
        train_gen, val_gen, test_gen = create_generators(
            split_json=SPLIT_JSON,
            batch_size=2,
            target_size=(96, 96, 48)
        )
        
        print(f"\n📊 Générateurs créés:")
        print(f"   Train: {len(train_gen)} batches")
        print(f"   Val: {len(val_gen)} batches")
        print(f"   Test: {len(test_gen)} batches")
        
        # Tester un batch
        print("\n🔄 Test d'un batch...")
        X_batch, y_batch = train_gen[0]
        
        print(f"✅ Batch chargé:")
        print(f"   X shape: {X_batch.shape}")
        print(f"   y shape: {y_batch.shape}")
        print(f"   X range: [{X_batch.min():.2f}, {X_batch.max():.2f}]")