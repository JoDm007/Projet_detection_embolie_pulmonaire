# src/O5_deep_learning/cnn_3d.py
"""
Réseau de neurones convolutionnel 3D pour la détection d'embolie pulmonaire (Objectif 5)

Architecture inspirée de VGG-like, adaptée aux volumes 3D de scanners pulmonaires.

Features :
- U-Net-like pour la segmentation (détection voxel-wise)
- CNN 3D classique pour classification patch-based
- Support multi-résolution

Compatible avec :
- Les données générées par VolumeDataGenerator
- Les splits Train/Val/Test
- TensorFlow 2.x / Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.metrics import MeanIoU, AUC, Recall, Precision
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CNN3D(tf.keras.Model):
    """
    CNN 3D pour classification patch-based.
    
    Architecture:
        - 3 blocs convolutionnels avec pooling
        - Dropout pour éviter le sur-apprentissage
        - Couches denses finales
        - Sortie binaire (embolie ou non)
    
    Args:
        input_shape: Tuple (height, width, depth, channels)
        num_classes: Nombre de classes (défaut: 1 pour binaire)
        dropout_rate: Taux de dropout (0.5)
        l2_reg: Facteur de régularisation L2 (0.0001)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (64, 64, 64, 1),
        num_classes: int = 1,
        dropout_rate: float = 0.5,
        l2_reg: float = 1e-4,
        name: str = "cnn_3d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Regularizer
        self.regularizer = regularizers.l2(l2_reg)
        
        # Build architecture
        self.model = self._build_architecture()
        
        logger.info(f"✅ CNN 3D initialisé: input={input_shape}, classes={num_classes}")
    
    def _build_architecture(self) -> tf.keras.Model:
        """Construit l'architecture du réseau"""
        
        # Input layer
        inputs = tf.keras.Input(shape=self.input_shape, name="input_volume")
        
        # Block 1 : 64 filtres
        x = layers.Conv3D(64, 3, padding='same', kernel_regularizer=self.regularizer)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv3D(64, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Block 2 : 128 filtres
        x = layers.Conv3D(128, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv3D(128, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Block 3 : 256 filtres
        x = layers.Conv3D(256, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv3D(256, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Block 4 : 512 filtres
        x = layers.Conv3D(512, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv3D(512, 3, padding='same', kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Global Average Pooling (remplace Flatten)
        x = layers.GlobalAveragePooling3D()(x)
        
        # Fully Connected layers
        x = layers.Dense(512, kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, kernel_regularizer=self.regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
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


class UNet3D(tf.keras.Model):
    """
    U-Net 3D pour segmentation voxel-wise (détection au niveau pixel).
    
    Architecture adaptée de U-Net classique en 3D.
    Idéale pour la détection précise des embolies.
    
    Args:
        input_shape: Tuple (height, width, depth, channels)
        num_classes: Nombre de classes (2: fond + embolie)
        features_start: Nombre initial de filtres (32)
        depth: Profondeur du réseau (4)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (128, 128, 64, 1),
        num_classes: int = 2,
        features_start: int = 32,
        depth: int = 4,
        name: str = "unet_3d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.features_start = features_start
        self.depth = depth
        
        # Build architecture
        self.model = self._build_architecture()
        
        logger.info(f"✅ U-Net 3D initialisé: input={input_shape}, features_start={features_start}")
    
    def _conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        name: str = "conv_block"
    ) -> tf.Tensor:
        """Bloc convolutionnel: 2x Conv3D + BN + ReLU"""
        x = layers.Conv3D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation('relu', name=f"{name}_relu1")(x)
        
        x = layers.Conv3D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation('relu', name=f"{name}_relu2")(x)
        
        return x
    
    def _up_conv_block(
        self,
        x: tf.Tensor,
        skip: tf.Tensor,
        filters: int,
        kernel_size: int = 2,
        name: str = "up_conv_block"
    ) -> tf.Tensor:
        """Bloc de up-convolution avec connexion de skip"""
        # Up-convolution
        x = layers.Conv3DTranspose(
            filters, kernel_size, strides=2, padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_up_conv"
        )(x)
        
        # Concaténation avec skip connection
        x = layers.Concatenate(axis=-1, name=f"{name}_concat")([x, skip])
        
        # Double convolution
        x = self._conv_block(x, filters, name=name)
        
        return x
    
    def _build_architecture(self) -> tf.keras.Model:
        """Construit l'architecture U-Net 3D"""
        
        inputs = tf.keras.Input(shape=self.input_shape, name="input_volume")
        
        # Encoder (down-sampling)
        encoders = []
        x = inputs
        features = self.features_start
        
        for d in range(self.depth):
            x = self._conv_block(x, features, name=f"encoder_{d}")
            encoders.append(x)
            x = layers.MaxPooling3D(pool_size=2, name=f"pool_{d}")(x)
            features *= 2
        
        # Bottleneck
        x = self._conv_block(x, features, name="bottleneck")
        
        # Decoder (up-sampling)
        for d in reversed(range(self.depth)):
            features //= 2
            x = self._up_conv_block(x, encoders[d], features, name=f"decoder_{d}")
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Conv3D(
                1, 1, activation='sigmoid', padding='same',
                name="output"
            )(x)
        else:
            outputs = layers.Conv3D(
                self.num_classes, 1, activation='softmax', padding='same',
                name="output"
            )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        return model
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def summary(self):
        self.model.summary()


class SimpleCNN3D(tf.keras.Model):
    """
    CNN 3D léger pour classification.
    Version simplifiée pour entraînement rapide.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (64, 64, 32, 1),
        num_classes: int = 1,
        name: str = "simple_cnn_3d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.model = self._build_architecture()
        
        logger.info(f"✅ SimpleCNN3D initialisé: {input_shape}")
    
    def _build_architecture(self) -> tf.keras.Model:
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        x = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling3D(2)(x)
        
        x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling3D(2)(x)
        
        x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling3D()(x)
        
        x = layers.Dense(64, activation='relu')(x)
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
# Fonctions utilitaires pour la compilation et l'entraînement
# ============================================================================

def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    loss: str = 'binary_crossentropy',
    metrics: Optional[List[str]] = None
) -> tf.keras.Model:
    """
    Compile un modèle avec optimiseur Adam.
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage initial
        loss: Fonction de perte
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
        loss=loss,
        metrics=metrics
    )
    
    logger.info(f"✅ Modèle compilé: lr={learning_rate}, loss={loss}")
    
    return model


def get_callbacks(
    checkpoint_path: str,
    early_stopping_patience: int = 20,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.5,
    tensorboard_log_dir: Optional[str] = None
) -> List[tf.keras.callbacks.Callback]:
    """
    Crée les callbacks pour l'entraînement.
    
    Args:
        checkpoint_path: Chemin pour sauvegarder le meilleur modèle
        early_stopping_patience: Patience pour early stopping
        reduce_lr_patience: Patience pour réduire le learning rate
        reduce_lr_factor: Facteur de réduction du learning rate
        tensorboard_log_dir: Dossier pour TensorBoard (optionnel)
    
    Returns:
        Liste des callbacks
    """
    callbacks = []
    
    # Checkpoint - sauvegarde le meilleur modèle
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard (optionnel)
    if tensorboard_log_dir:
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
    
    return callbacks


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DES MODÈLES 3D")
    print("=" * 60)
    
    # Paramètres de test
    BATCH_SIZE = 2
    INPUT_SHAPE = (64, 64, 32, 1)
    
    print(f"\n📊 Configuration de test:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Input shape: {INPUT_SHAPE}")
    
    # Test 1: CNN 3D
    print("\n" + "-" * 40)
    print("Test 1: CNN 3D (Classification)")
    
    cnn_model = CNN3D(input_shape=INPUT_SHAPE, num_classes=1)
    cnn_model.summary()
    
    # Tester forward pass
    dummy_input = tf.random.normal((BATCH_SIZE, *INPUT_SHAPE))
    dummy_output = cnn_model(dummy_input)
    print(f"✅ CNN3D output shape: {dummy_output.shape}")
    
    # Test 2: SimpleCNN3D
    print("\n" + "-" * 40)
    print("Test 2: SimpleCNN3D (Version légère)")
    
    simple_cnn = SimpleCNN3D(input_shape=INPUT_SHAPE, num_classes=1)
    simple_cnn.summary()
    
    dummy_output = simple_cnn(dummy_input)
    print(f"✅ SimpleCNN3D output shape: {dummy_output.shape}")
    
    # Test 3: U-Net 3D
    print("\n" + "-" * 40)
    print("Test 3: U-Net 3D (Segmentation)")
    
    unet_model = UNet3D(input_shape=INPUT_SHAPE, num_classes=2)
    unet_model.summary()
    
    dummy_output = unet_model(dummy_input)
    print(f"✅ UNet3D output shape: {dummy_output.shape}")
    
    # Test 4: Compilation
    print("\n" + "-" * 40)
    print("Test 4: Compilation du modèle")
    
    compiled_model = compile_model(simple_cnn, learning_rate=1e-4)
    print("✅ Modèle compilé avec succès")
    
    print("\n" + "=" * 60)
    print("✅ Tous les tests sont passés!")
    print("=" * 60)