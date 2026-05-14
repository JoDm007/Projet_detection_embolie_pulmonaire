# src/O5_deep_learning/predict.py
"""
Prédiction sur nouveaux patients avec les modèles deep learning (Objectif 5)

Fonctionnalités :
- Chargement du modèle entraîné
- Prédiction sur un volume entier ou par patches
- Visualisation des résultats (superposition masque prédit)
- Sauvegarde des prédictions
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.O1_preprocessing.loader import load_patient
from src.O1_preprocessing.normalisation import apply_windowing
from src.O5_deep_learning.cnn_3d import CNN3D, SimpleCNN3D, UNet3D

logger = logging.getLogger(__name__)


class DeepLearningPredictor:
    """
    Prédicteur pour les modèles deep learning.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé (.h5)
        model_type: Type de modèle ('simple_cnn', 'cnn_3d', 'unet_3d')
        target_size: Taille cible pour le resize (height, width, depth)
        window_name: Type de fenêtrage ('pulmonary', 'mediastinal', 'soft_tissue')
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = 'cnn_3d',
        target_size: Tuple[int, int, int] = (128, 128, 64),
        window_name: str = 'pulmonary'
    ):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.target_size = target_size
        self.window_name = window_name
        
        # Charger le modèle
        self.model = self._load_model()
        
        logger.info(f"✅ Predictor initialisé: {model_path}")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Target size: {target_size}")
    
    def _load_model(self) -> tf.keras.Model:
        """Charge le modèle sauvegardé"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        # Reconstruire l'architecture
        input_shape = (*self.target_size, 1)
        
        if self.model_type == 'simple_cnn':
            model = SimpleCNN3D(input_shape=input_shape, num_classes=1)
        elif self.model_type == 'cnn_3d':
            model = CNN3D(input_shape=input_shape, num_classes=1)
        elif self.model_type == 'unet_3d':
            model = UNet3D(input_shape=input_shape, num_classes=2)
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")
        
        # Charger les poids
        model.load_weights(self.model_path)
        
        return model
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Prétraite un volume pour l'inférence.
        
        Steps:
            1. Appliquer le fenêtrage
            2. Normaliser entre 0 et 1
            3. Redimensionner à target_size
            4. Ajouter dimensions batch et channel
        """
        from scipy.ndimage import zoom
        
        # Appliquer le fenêtrage
        volume_windowed = apply_windowing(volume, self.window_name)
        
        # Normaliser entre 0 et 1
        hu_min, hu_max = -1024, 600
        volume_norm = np.clip(volume_windowed, hu_min, hu_max)
        volume_norm = (volume_norm - hu_min) / (hu_max - hu_min)
        
        # Redimensionner
        current_shape = volume_norm.shape
        zoom_factors = [
            self.target_size[0] / current_shape[0],
            self.target_size[1] / current_shape[1],
            self.target_size[2] / current_shape[2]
        ]
        volume_resized = zoom(volume_norm, zoom_factors, order=1)
        
        # Ajouter dimensions
        volume_batch = np.expand_dims(volume_resized, axis=0)  # (1, H, W, D)
        volume_batch = np.expand_dims(volume_batch, axis=-1)   # (1, H, W, D, 1)
        
        return volume_batch, volume_resized
    
    def predict_patient(
        self,
        patient_id: str,
        base_dir: str = "data/raw"
    ) -> Dict[str, Any]:
        """
        Prédit pour un patient entier.
        
        Args:
            patient_id: ID du patient (ex: 'PAT001')
            base_dir: Répertoire des données
        
        Returns:
            results: Dictionnaire avec:
                - probability: Score de probabilité (0-1)
                - prediction: Classe prédite (0 ou 1)
                - mask_pred: Masque prédit (pour U-Net)
        """
        logger.info(f"🚀 Prédiction pour {patient_id}")
        
        # Charger le volume
        volume, mask, vol_meta, mask_meta = load_patient(patient_id, base_dir)
        
        if volume is None:
            logger.error(f"Impossible de charger {patient_id}")
            return {'error': 'Patient non trouvé'}
        
        # Prétraiter
        volume_batch, volume_resized = self.preprocess_volume(volume)
        
        # Inférence
        predictions = self.model.predict(volume_batch, verbose=0)
        
        results = {
            'patient_id': patient_id,
            'volume_shape': volume.shape,
            'has_ground_truth': mask is not None
        }
        
        if self.model_type == 'unet_3d':
            # Pour U-Net, on a un masque de segmentation
            if predictions.shape[-1] == 2:
                mask_pred = np.argmax(predictions[0], axis=-1)
            else:
                mask_pred = (predictions[0] > 0.5).astype(np.uint8)
            
            # Resize back to original size
            mask_pred = self._resize_back(mask_pred, volume.shape[:3])
            
            results['mask_pred'] = mask_pred
            results['embolism_volume'] = int(mask_pred.sum())
            
            if mask is not None:
                # Calculer le Dice score si GT disponible
                intersection = np.logical_and(mask, mask_pred).sum()
                dice = 2 * intersection / (mask.sum() + mask_pred.sum() + 1e-7)
                results['dice_score'] = float(dice)
        else:
            # Pour CNN, on a une probabilité
            probability = float(predictions[0][0])
            results['probability'] = probability
            results['prediction'] = 1 if probability > 0.5 else 0
        
        logger.info(f"✅ Prédiction terminée: {results}")
        
        return results
    
    def predict_patient_patches(
        self,
        patient_id: str,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        stride: Tuple[int, int, int] = (16, 16, 16),
        base_dir: str = "data/raw"
    ) -> Dict[str, Any]:
        """
        Prédiction par patches 3D (pour CNN).
        Utile pour les volumes trop grands pour la mémoire GPU.
        
        Args:
            patient_id: ID du patient
            patch_size: Taille des patches (height, width, depth)
            stride: Pas de déplacement entre patches
            base_dir: Répertoire des données
        """
        logger.info(f"🚀 Prédiction par patches pour {patient_id} (size={patch_size})")
        
        # Charger le volume
        volume, mask, vol_meta, mask_meta = load_patient(patient_id, base_dir)
        
        if volume is None:
            return {'error': 'Patient non trouvé'}
        
        # Prétraiter
        _, volume_resized = self.preprocess_volume(volume)
        
        # Calculer les coordonnées des patches
        h, w, d = volume_resized.shape
        ph, pw, pd = patch_size
        sh, sw, sd = stride
        
        probabilities = np.zeros((h, w, d))
        counts = np.zeros((h, w, d))
        
        for z in range(0, d - pd + 1, sd):
            for y in range(0, h - ph + 1, sh):
                for x in range(0, w - pw + 1, sw):
                    patch = volume_resized[y:y+ph, x:x+pw, z:z+pd]
                    patch = np.expand_dims(patch, axis=(0, -1))
                    
                    prob = self.model.predict(patch, verbose=0)[0][0]
                    
                    probabilities[y:y+ph, x:x+pw, z:z+pd] += prob
                    counts[y:y+ph, x:x+pw, z:z+pd] += 1
        
        # Moyenne des probabilités
        probabilities = probabilities / (counts + 1e-7)
        
        # Prédiction finale (si moyenne > 0.5)
        mask_pred = (probabilities > 0.5).astype(np.uint8)
        
        # Resize back to original size
        mask_pred = self._resize_back(mask_pred, volume.shape[:3])
        
        results = {
            'patient_id': patient_id,
            'probability_map': probabilities,
            'mask_pred': mask_pred,
            'embolism_volume': int(mask_pred.sum())
        }
        
        if mask is not None:
            intersection = np.logical_and(mask, mask_pred).sum()
            dice = 2 * intersection / (mask.sum() + mask_pred.sum() + 1e-7)
            results['dice_score'] = float(dice)
        
        return results
    
    def _resize_back(self, mask: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
        """Redimensionne un masque à sa taille originale"""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            original_shape[0] / mask.shape[0],
            original_shape[1] / mask.shape[1],
            original_shape[2] / mask.shape[2]
        ]
        
        mask_resized = zoom(mask, zoom_factors, order=0)
        return (mask_resized > 0.5).astype(np.uint8)
    
    def visualize_prediction(
        self,
        results: Dict[str, Any],
        slice_idx: int = None,
        save_path: Optional[Path] = None
    ):
        """Visualise la prédiction sur une coupe"""
        patient_id = results.get('patient_id', 'unknown')
        
        # Charger le volume original pour visualisation
        volume, mask, _, _ = load_patient(patient_id)
        
        if volume is None:
            logger.error("Impossible de charger le volume pour visualisation")
            return
        
        # Sélectionner la coupe (milieu par défaut)
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Coupe CT
        axes[0].imshow(volume[slice_idx], cmap='gray')
        axes[0].set_title(f'CT Scan - Coupe {slice_idx}')
        axes[0].axis('off')
        
        # Ground truth (si disponible)
        if results.get('has_ground_truth') and mask is not None:
            axes[1].imshow(mask[slice_idx], cmap='Reds', alpha=0.5)
            axes[1].imshow(volume[slice_idx], cmap='gray', alpha=0.5)
            axes[1].set_title('Ground Truth (Rouge)')
            axes[1].axis('off')
        
        # Prédiction
        if 'mask_pred' in results:
            mask_pred_slice = results['mask_pred'][slice_idx]
            axes[2].imshow(mask_pred_slice, cmap='Greens', alpha=0.5)
            axes[2].imshow(volume[slice_idx], cmap='gray', alpha=0.5)
            axes[2].set_title(f'Prédiction (Vert) - Dice={results.get("dice_score", 0):.3f}')
        elif 'probability' in results:
            axes[2].text(0.5, 0.5, f"Probabilité: {results['probability']:.3f}\n"
                              f"Prédiction: {'Embolie' if results['prediction'] else 'Normal'}",
                              ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
            axes[2].imshow(volume[slice_idx], cmap='gray')
            axes[2].set_title('Résultat CNN')
        
        axes[2].axis('off')
        
        plt.suptitle(f'Patient: {patient_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Visualisation sauvegardée: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "results/deep_learning/models/cnn_3d_best.h5"
    
    if Path(MODEL_PATH).exists():
        predictor = DeepLearningPredictor(
            model_path=MODEL_PATH,
            model_type='cnn_3d'
        )
        
        # Tester sur un patient
        results = predictor.predict_patient('PAT001')
        print(f"\nRésultats: {results}")
        
        # Visualiser
        predictor.visualize_prediction(results)
    else:
        print(f"⚠️ Modèle non trouvé: {MODEL_PATH}")
        print("Entraîne d'abord un modèle avec train.py")