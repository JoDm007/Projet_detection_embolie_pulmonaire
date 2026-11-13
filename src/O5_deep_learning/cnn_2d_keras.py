"""
Module: src.O5_deep_learning.cnn_2d.py
Auteur : Joseph DATE-MASSE
Description: Réseau de neurones convolutif 2D pour la détection d'embolie pulmonaire (Objectif 5)

Architecture : CNN 2D traitant chaque coupe indépendamment.
Approche plus légère que le 3D, idéale pour :
- Démarrer rapidement
- GPU avec mémoire limitée
- Transfer learning (ImageNet)

Stratégie :
1. Chaque coupe 2D du volume CT est traitée individuellement
2. Agrégation des prédictions par patient (max, moyenne, ou LSTM)
3. Classification binaire (embolie présent ou non)

Compatible avec :
- Les données générées par VolumeDataGenerator (modifié pour 2D)
- Transfer learning avec ResNet, EfficientNet, etc.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.metrics import AUC, Recall, Precision
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class CNN2D(tf.keras.Model):
    """
    CNN 2D pour classification de coupes CT.
    
    Architecture :
        - 4 blocs convolutionnels avec pooling
        - Dropout pour éviter le sur-apprentissage
        - Sortie binaire par coupe
        - Agrégation finale par patient
    
    Args:
        input_shape: Tuple (height, width, channels) - ex: (128, 128, 1)
        num_classes: Nombre de classes (défaut: 1 pour binaire)
        dropout_rate: Taux de dropout (0.5)
        l2_reg: Facteur de régularisation L2 (0.0001)
        use_batch_norm: Utiliser BatchNormalization
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 1),
        num_classes: int = 1,
        dropout_rate: float = 0.5,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True,
        name: str = "cnn_2d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Regularizer
        self.regularizer = regularizers.l2(l2_reg)
        
        # Build architecture
        self.model = self._build_architecture()
        
        logger.info(f"✅ CNN 2D initialisé: input={input_shape}, classes={num_classes}")
    
    def _conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        name: str = "conv_block"
    ) -> tf.Tensor:
        """Bloc convolutionnel: Conv2D + BatchNorm + ReLU + MaxPooling"""
        x = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name}_conv"
        )(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn")(x)
        
        x = layers.Activation('relu', name=f"{name}_relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)
        
        return x
    
    def _build_architecture(self) -> tf.keras.Model:
        """Construit l'architecture du réseau"""
        
        inputs = tf.keras.Input(shape=self.input_shape, name="input_slice")
        
        # Block 1: 32 filtres
        x = self._conv_block(inputs, 32, name="block1")
        
        # Block 2: 64 filtres
        x = self._conv_block(x, 64, name="block2")
        
        # Block 3: 128 filtres
        x = self._conv_block(x, 128, name="block3")
        
        # Block 4: 256 filtres
        x = self._conv_block(x, 256, name="block4")
        
        # Global Average Pooling (remplace Flatten)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        
        # Fully Connected layers
        x = layers.Dense(256, kernel_regularizer=self.regularizer, name="fc1")(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name="fc1_bn")(x)
        
        x = layers.Activation('relu', name="fc1_relu")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout1")(x)
        
        x = layers.Dense(128, kernel_regularizer=self.regularizer, name="fc2")(x)
        
        if self.use_batch_norm:
            x = layers.BatchNormalization(name="fc2_bn")(x)
        
        x = layers.Activation('relu', name="fc2_relu")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout2")(x)
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name="output")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        return model
    
    def call(self, inputs, training=False):
        """Forward pass"""
        return self.model(inputs, training=training)
    
    def summary(self):
        """Affiche le résumé du modèle"""
        self.model.summary()


class CNN2DTransferLearning(tf.keras.Model):
    """
    CNN 2D avec transfer learning (ResNet50, EfficientNet, etc.).
    
    Idéal quand on a peu de données médicales, on peut
    utiliser des poids pré-entraînés sur ImageNet.
    
    Args:
        input_shape: Tuple (height, width, channels) - ex: (224, 224, 1) ou (224, 224, 3)
        base_model_name: 'resnet50', 'efficientnetb0', 'mobilenetv2'
        num_classes: Nombre de classes
        trainable_base: Entraîner ou non la base (False = transfer learning pur)
        dropout_rate: Taux de dropout
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 1),
        base_model_name: str = 'efficientnetb0',
        num_classes: int = 1,
        trainable_base: bool = False,
        dropout_rate: float = 0.5,
        name: str = "cnn_2d_transfer"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.trainable_base = trainable_base
        self.dropout_rate = dropout_rate
        
        # Build architecture
        self.model = self._build_architecture()
        
        logger.info(f"✅ CNN2D Transfer Learning: base={base_model_name}, trainable={trainable_base}")
    
    def _get_base_model(self):
        """Récupère le modèle de base pré-entraîné"""
        
        # Adapter les canaux si besoin (grayscale → RGB)
        if self.input_shape[-1] == 1:
            # On répète le canal 3 fois pour simuler RGB
            input_layer = tf.keras.Input(shape=self.input_shape)
            x = layers.Conv2D(3, (3, 3), padding='same')(input_layer)
            x = layers.Activation('relu')(x)
            effective_input = x
        else:
            effective_input = tf.keras.Input(shape=self.input_shape)
        
        # Choix du modèle de base
        if self.base_model_name == 'resnet50':
            base_model = applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=effective_input if self.input_shape[-1] == 3 else None,
                input_shape=(self.input_shape[0], self.input_shape[1], 3) if self.input_shape[-1] == 1 else self.input_shape
            )
        elif self.base_model_name == 'efficientnetb0':
            base_model = applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=effective_input if self.input_shape[-1] == 3 else None,
                input_shape=(self.input_shape[0], self.input_shape[1], 3) if self.input_shape[-1] == 1 else self.input_shape
            )
        elif self.base_model_name == 'mobilenetv2':
            base_model = applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_tensor=effective_input if self.input_shape[-1] == 3 else None,
                input_shape=(self.input_shape[0], self.input_shape[1], 3) if self.input_shape[-1] == 1 else self.input_shape
            )
        else:
            raise ValueError(f"Modèle de base inconnu: {self.base_model_name}")
        
        # Geler ou non la base
        base_model.trainable = self.trainable_base
        
        return base_model, effective_input
    
    def _build_architecture(self) -> tf.keras.Model:
        """Construit l'architecture avec transfer learning"""
        
        base_model, base_input = self._get_base_model()
        
        # Forward pass
        x = base_model(base_input)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Têtes de classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name="output")(x)
        
        model = tf.keras.Model(inputs=base_input, outputs=outputs, name=self.name)
        
        return model
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def summary(self):
        self.model.summary()


class SliceAggregator(tf.keras.Model):
    """
    Agrégateur de prédictions par coupe pour obtenir une prédiction par patient.
    
    Stratégies disponibles:
    - 'max': Probabilité maximale parmi toutes les coupes
    - 'mean': Moyenne des probabilités
    - 'attention': Pondération attentionnelle (apprise)
    - 'rnn': LSTM pour modéliser la séquence de coupes
    """
    
    def __init__(
        self,
        cnn_2d_model: tf.keras.Model,
        aggregation_strategy: str = 'max',
        rnn_units: int = 64,
        name: str = "slice_aggregator"
    ):
        super().__init__(name=name)
        
        self.cnn_2d_model = cnn_2d_model
        self.aggregation_strategy = aggregation_strategy
        self.rnn_units = rnn_units
        
        # Pour les stratégies qui nécessitent des couches apprises
        if aggregation_strategy == 'attention':
            self.attention_weights = layers.Dense(1, activation='sigmoid')
        elif aggregation_strategy == 'rnn':
            self.lstm = layers.LSTM(rnn_units, return_sequences=False)
            self.dense = layers.Dense(1, activation='sigmoid')
        
        logger.info(f"✅ SliceAggregator: strategy={aggregation_strategy}")
    
    def call(self, volume_batch: tf.Tensor, training=False):
        """
        Args:
            volume_batch: Tensor de forme (batch, depth, height, width, channels)
        
        Returns:
            predictions: Tensor de forme (batch, 1)
        """
        batch_size = tf.shape(volume_batch)[0]
        depth = tf.shape(volume_batch)[1]
        
        # Reshape pour traiter chaque coupe indépendamment
        # (batch * depth, height, width, channels)
        slices = tf.reshape(volume_batch, (-1, *volume_batch.shape[2:]))
        
        # Prédictions par coupe
        slice_probs = self.cnn_2d_model(slices, training=training)
        slice_probs = tf.reshape(slice_probs, (batch_size, depth, -1))
        
        if self.aggregation_strategy == 'max':
            # Probabilité maximale parmi les coupes
            predictions = tf.reduce_max(slice_probs, axis=1)
            
        elif self.aggregation_strategy == 'mean':
            # Moyenne des probabilités
            predictions = tf.reduce_mean(slice_probs, axis=1)
            
        elif self.aggregation_strategy == 'attention':
            # Pondération attentionnelle
            weights = self.attention_weights(slice_probs)
            weights = tf.nn.softmax(weights, axis=1)
            predictions = tf.reduce_sum(slice_probs * weights, axis=1)
            
        elif self.aggregation_strategy == 'rnn':
            # Modélisation séquentielle
            lstm_out = self.lstm(slice_probs)
            predictions = self.dense(lstm_out)
            
        else:
            raise ValueError(f"Stratégie inconnue: {self.aggregation_strategy}")
        
        return predictions
    
    def summary(self):
        """Affiche le résumé du modèle"""
        self.cnn_2d_model.summary()
        if hasattr(self, 'lstm'):
            print("\nLSTM Aggregator:")
            print(f"  Units: {self.rnn_units}")
        print(f"\nAggregation Strategy: {self.aggregation_strategy}")


class SimpleCNN2D(tf.keras.Model):
    """
    CNN 2D ultra-léger pour prototypage rapide.
    
    Architecture minimaliste pour tester le pipeline.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 1),
        num_classes: int = 1,
        name: str = "simple_cnn_2d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.model = self._build_architecture()
        
        logger.info(f"✅ SimpleCNN2D initialisé: {input_shape}")
    
    def _build_architecture(self) -> tf.keras.Model:
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def summary(self):
        self.model.summary()


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def compile_cnn_2d(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    metrics: Optional[List[str]] = None
) -> tf.keras.Model:
    """
    Compile un modèle CNN 2D.
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
        metrics: Liste des métriques
    
    Returns:
        Modèle compilé
    """
    if metrics is None:
        metrics = [
            'accuracy',
            AUC(name='auc'),
            Recall(name='sensitivity'),
            Precision(name='precision')
        ]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    logger.info(f"✅ CNN 2D compilé: lr={learning_rate}")
    
    return model


def create_cnn_2d_generator_compatible(
    volume_3d: np.ndarray,
    labels_per_slice: Optional[np.ndarray] = None,
    slice_axis: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convertit un volume 3D en une séquence de slices 2D.
    Utile pour adapter le data_generator 3D au CNN 2D.
    
    Args:
        volume_3d: Volume 3D (depth, height, width)
        labels_per_slice: Labels par slice (optionnel)
        slice_axis: Axe des slices (0 = profondeur)
    
    Returns:
        slices: Array (n_slices, height, width, 1)
        slice_labels: Labels par slice (si fournis)
    """
    n_slices = volume_3d.shape[slice_axis]
    
    # Extraire toutes les slices
    slices = []
    for i in range(n_slices):
        if slice_axis == 0:
            slice_2d = volume_3d[i, :, :]
        elif slice_axis == 1:
            slice_2d = volume_3d[:, i, :]
        else:
            slice_2d = volume_3d[:, :, i]
        
        # Ajouter dimension canal
        slice_2d = np.expand_dims(slice_2d, axis=-1)
        slices.append(slice_2d)
    
    slices = np.array(slices)
    
    if labels_per_slice is not None:
        return slices, labels_per_slice
    
    return slices, None


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DES MODÈLES CNN 2D")
    print("=" * 60)
    
    # Paramètres de test
    BATCH_SIZE = 4
    INPUT_SHAPE = (128, 128, 1)
    
    print(f"\n📊 Configuration de test:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Input shape: {INPUT_SHAPE}")
    
    # Test 1: SimpleCNN2D
    print("\n" + "-" * 40)
    print("Test 1: SimpleCNN2D (Ultra-léger)")
    
    simple_cnn = SimpleCNN2D(input_shape=INPUT_SHAPE, num_classes=1)
    simple_cnn.summary()
    
    dummy_input = tf.random.normal((BATCH_SIZE, *INPUT_SHAPE))
    dummy_output = simple_cnn(dummy_input)
    print(f"✅ SimpleCNN2D output shape: {dummy_output.shape}")
    
    # Test 2: CNN2D standard
    print("\n" + "-" * 40)
    print("Test 2: CNN2D (Standard)")
    
    cnn_2d = CNN2D(input_shape=INPUT_SHAPE, num_classes=1)
    cnn_2d.summary()
    
    dummy_output = cnn_2d(dummy_input)
    print(f"✅ CNN2D output shape: {dummy_output.shape}")
    
    # Test 3: Transfer Learning (optionnel, plus lent)
    print("\n" + "-" * 40)
    print("Test 3: CNN2D Transfer Learning (EfficientNetB0)")
    print("   (Peut prendre quelques secondes au premier chargement)")
    
    try:
        transfer_cnn = CNN2DTransferLearning(
            input_shape=(224, 224, 1),
            base_model_name='efficientnetb0',
            num_classes=1,
            trainable_base=False
        )
        transfer_cnn.summary()
        print("✅ Transfer Learning CNN chargé")
    except Exception as e:
        print(f"⚠️ Transfer Learning non disponible: {e}")
    
    # Test 4: Compilation
    print("\n" + "-" * 40)
    print("Test 4: Compilation du modèle")
    
    compiled_model = compile_cnn_2d(simple_cnn, learning_rate=1e-4)
    print("✅ Modèle compilé avec succès")
    
    print("\n" + "=" * 60)
    print("✅ Tous les tests sont passés!")
    print("=" * 60)