# src/O5_deep_learning/train.py
"""
Entraînement des modèles deep learning 3D pour la détection d'embolie pulmonaire (Objectif 5)

Fonctionnalités :
- Entraînement avec validation croisée (optionnelle)
- Callbacks pour early stopping et réduction du learning rate
- Sauvegarde automatique du meilleur modèle
- Logging détaillé et visualisation des courbes
- Compatible avec les générateurs de données
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import json
from datetime import datetime
import sys

# Ajout du chemin racine
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.O5_deep_learning.cnn_3d import CNN3D, SimpleCNN3D, UNet3D, compile_model, get_callbacks
from src.O5_deep_learning.data_generator import VolumeDataGenerator, create_generators

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningTrainer:
    """
    Entraîneur de modèles deep learning pour la détection d'embolie.
    
    Args:
        model_name: 'simple_cnn', 'cnn_3d', ou 'unet_3d'
        split_json: Chemin vers le fichier JSON des splits
        output_dir: Répertoire de sortie pour les modèles et logs
        batch_size: Taille des batches
        target_size: Taille cible des volumes (height, width, depth)
        window_name: Type de fenêtrage ('pulmonary', 'mediastinal', 'soft_tissue')
        learning_rate: Taux d'apprentissage initial
        epochs: Nombre maximum d'époques
        early_stopping_patience: Patience pour early stopping
        use_augmentation: Appliquer la data augmentation
    """
    
    def __init__(
        self,
        model_name: str = 'simple_cnn',
        split_json: str = "data/splits/train_val_test.json",
        output_dir: str = "results/deep_learning",
        batch_size: int = 4,
        target_size: Tuple[int, int, int] = (128, 128, 64),
        window_name: str = 'pulmonary',
        learning_rate: float = 1e-4,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        use_augmentation: bool = True
    ):
        self.model_name = model_name
        self.split_json = Path(split_json)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.target_size = target_size
        self.window_name = window_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_augmentation = use_augmentation
        
        # Créer les répertoires de sortie
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.figures_dir = self.output_dir / "figures"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser les générateurs
        self._init_generators()
        
        # Construire le modèle
        self.model = self._build_model()
        
        logger.info(f"✅ DeepLearningTrainer initialisé: {model_name}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def _init_generators(self):
        """Initialise les générateurs Train/Val/Test"""
        logger.info("📊 Création des générateurs de données...")
        
        # Ajuster la taille si besoin (multiplier par 32 pour les convolutions)
        # Certains modèles nécessitent des dimensions multiples de 2^depth
        target_size = list(self.target_size)
        for i in range(3):
            target_size[i] = (target_size[i] // 32) * 32
        self.target_size = tuple(target_size)
        
        self.train_gen, self.val_gen, self.test_gen = create_generators(
            split_json=str(self.split_json),
            batch_size=self.batch_size,
            target_size=self.target_size,
            window_name=self.window_name,
            augment_train=self.use_augmentation
        )
        
        logger.info(f"✅ Générateurs créés:")
        logger.info(f"   Train: {len(self.train_gen)} batches")
        logger.info(f"   Val: {len(self.val_gen)} batches")
        logger.info(f"   Test: {len(self.test_gen)} batches")
        logger.info(f"   Target size: {self.target_size}")
    
    def _build_model(self) -> tf.keras.Model:
        """Construit le modèle selon le nom spécifié"""
        input_shape = (*self.target_size, 1)
        
        if self.model_name == 'simple_cnn':
            model = SimpleCNN3D(
                input_shape=input_shape,
                num_classes=1,
                name=f"simple_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif self.model_name == 'cnn_3d':
            model = CNN3D(
                input_shape=input_shape,
                num_classes=1,
                dropout_rate=0.5,
                name=f"cnn_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif self.model_name == 'unet_3d':
            model = UNet3D(
                input_shape=input_shape,
                num_classes=2,
                features_start=32,
                name=f"unet_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            raise ValueError(f"Modèle inconnu: {self.model_name}")
        
        # Compiler le modèle
        if self.model_name == 'unet_3d':
            from tensorflow.keras.losses import SparseCategoricalCrossentropy
            loss = SparseCategoricalCrossentropy()
        else:
            loss = 'binary_crossentropy'
        
        model = compile_model(
            model=model,
            learning_rate=self.learning_rate,
            loss=loss
        )
        
        return model
    
    def train(self) -> Dict[str, Any]:
        """
        Lance l'entraînement du modèle.
        
        Returns:
            history: Historique d'entraînement
        """
        logger.info("=" * 60)
        logger.info(f"🚀 DÉMARRAGE DE L'ENTRAÎNEMENT - {self.model_name}")
        logger.info("=" * 60)
        
        # Callbacks
        checkpoint_path = str(self.models_dir / f"{self.model_name}_best.h5")
        tensorboard_log_dir = str(self.logs_dir / self.model_name)
        
        callbacks = get_callbacks(
            checkpoint_path=checkpoint_path,
            early_stopping_patience=self.early_stopping_patience,
            reduce_lr_patience=10,
            reduce_lr_factor=0.5,
            tensorboard_log_dir=tensorboard_log_dir
        )
        
        # Entraînement
        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder l'historique
        self._save_history(history)
        
        # Afficher les courbes
        self._plot_training_history(history)
        
        # Évaluer sur le test set
        test_metrics = self.evaluate_on_test()
        
        # Sauvegarder les métriques
        self._save_metrics(test_metrics)
        
        logger.info("=" * 60)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 60)
        
        return {
            'history': history.history,
            'test_metrics': test_metrics,
            'best_model_path': checkpoint_path
        }
    
    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Évalue le modèle sur le jeu de test.
        
        Returns:
            metrics: Dictionnaire des métriques
        """
        logger.info("📊 Évaluation sur le jeu de test...")
        
        # Charger le meilleur modèle sauvegardé
        best_model_path = self.models_dir / f"{self.model_name}_best.h5"
        if best_model_path.exists():
            logger.info(f"📥 Chargement du meilleur modèle: {best_model_path}")
            self.model.load_weights(best_model_path)
        
        # Évaluation
        results = self.model.evaluate(self.test_gen, verbose=1)
        
        # Organiser les métriques
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])
        
        logger.info("✅ Métriques sur test:")
        for k, v in metrics.items():
            logger.info(f"   {k}: {v:.4f}")
        
        return metrics
    
    def _save_history(self, history):
        """Sauvegarde l'historique d'entraînement"""
        history_path = self.output_dir / f"{self.model_name}_history.json"
        
        # Convertir les valeurs numpy en float
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"💾 Historique sauvegardé: {history_path}")
    
    def _save_metrics(self, metrics: Dict[str, float]):
        """Sauvegarde les métriques dans un CSV"""
        metrics_path = self.output_dir / "test_metrics.csv"
        
        # Charger les métriques existantes ou créer nouveau DataFrame
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
        else:
            df = pd.DataFrame()
        
        # Ajouter les nouvelles métriques
        new_row = {'model': self.model_name, 'timestamp': datetime.now().isoformat()}
        new_row.update(metrics)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df.to_csv(metrics_path, index=False)
        logger.info(f"💾 Métriques sauvegardées: {metrics_path}")
    
    def _plot_training_history(self, history):
        """Trace les courbes d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Loss over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC
        if 'auc' in history.history:
            axes[1].plot(history.history['auc'], label='Train AUC')
            axes[1].plot(history.history['val_auc'], label='Val AUC')
            axes[1].set_title('AUC over Epochs')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUC')
            axes[1].legend()
            axes[1].grid(True)
        elif 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Train Acc')
            axes[1].plot(history.history['val_accuracy'], label='Val Acc')
            axes[1].set_title('Accuracy over Epochs')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder
        fig_path = self.figures_dir / f"{self.model_name}_training_history.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Figure sauvegardée: {fig_path}")
        
        plt.show()


def train_all_models(
    split_json: str = "data/splits/train_val_test.json",
    output_dir: str = "results/deep_learning",
    batch_size: int = 4,
    target_size: Tuple[int, int, int] = (128, 128, 64)
) -> Dict[str, Any]:
    """
    Entraîne tous les modèles et compare les résultats.
    
    Args:
        split_json: Chemin vers les splits
        output_dir: Répertoire de sortie
        batch_size: Taille des batches
        target_size: Taille cible des volumes
    
    Returns:
        results: Dictionnaire des résultats par modèle
    """
    models_to_train = ['simple_cnn', 'cnn_3d']
    results = {}
    
    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info(f"🚀 Entraînement de {model_name}")
        logger.info("=" * 60)
        
        trainer = DeepLearningTrainer(
            model_name=model_name,
            split_json=split_json,
            output_dir=output_dir,
            batch_size=batch_size,
            target_size=target_size,
            window_name='pulmonary',
            learning_rate=1e-4,
            epochs=100,
            early_stopping_patience=15,
            use_augmentation=True
        )
        
        result = trainer.train()
        results[model_name] = result
    
    # Afficher le comparatif
    print("\n" + "=" * 60)
    print("📊 COMPARAISON DES MODÈLES")
    print("=" * 60)
    
    for model_name, result in results.items():
        metrics = result['test_metrics']
        print(f"\n{model_name.upper()}:")
        for k, v in metrics.items():
            if 'loss' not in k:
                print(f"  {k}: {v:.4f}")
    
    return results


if __name__ == "__main__":
    # Configuration
    SPLIT_JSON = "data/splits/train_val_test.json"
    OUTPUT_DIR = "results/deep_learning"
    
    # Vérifier si les splits existent
    if not Path(SPLIT_JSON).exists():
        print(f"⚠️ Fichier {SPLIT_JSON} non trouvé")
        print("Crée d'abord les splits avec src/O1_preprocessing/split_data.py")
    else:
        # Entraîner tous les modèles
        results = train_all_models(
            split_json=SPLIT_JSON,
            output_dir=OUTPUT_DIR,
            batch_size=2,  # Ajuste selon ta mémoire GPU
            target_size=(96, 96, 64)
        )