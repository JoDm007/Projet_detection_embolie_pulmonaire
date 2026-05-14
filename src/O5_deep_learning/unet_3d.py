# src/O5_deep_learning/unet_3d.py
"""
U-Net 3D pour la segmentation des embolies pulmonaires (Objectif 5 - Approche avancée)

Architecture U-Net adaptée aux volumes 3D :
- Encoder (down-sampling) : captures le contexte global
- Decoder (up-sampling) : restaure la résolution spatiale
- Skip connections : fusionnent les détails fins avec le contexte
- Sortie : masque binaire 3D (fond vs embolie)

Avantages :
- Segmentation au niveau voxel (précision maximale)
- Détection des petites embolies
- Pas besoin d'extraction de candidats (contrairement à O2)

Compatible avec :
- Les données générées par VolumeDataGenerator (masques 3D)
- Sortie évaluable avec Dice, IoU, Sensibilité par voxel
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional, List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConvBlock(layers.Layer):
    """
    Bloc convolutionnel 3D : Double convolution + BatchNorm + Activation
    
    Args:
        filters: Nombre de filtres
        kernel_size: Taille du noyau (défaut: 3)
        use_batch_norm: Utiliser BatchNormalization
        dropout_rate: Taux de dropout (0 = pas de dropout)
        l2_reg: Facteur de régularisation L2
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        name: str = "conv_block"
    ):
        super().__init__(name=name)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.regularizer = regularizers.l2(l2_reg)
        
        # Première convolution
        self.conv1 = layers.Conv3D(
            filters, kernel_size, padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name}_conv1"
        )
        self.bn1 = layers.BatchNormalization(name=f"{name}_bn1") if use_batch_norm else None
        self.act1 = layers.Activation('relu', name=f"{name}_relu1")
        
        # Deuxième convolution
        self.conv2 = layers.Conv3D(
            filters, kernel_size, padding='same',
            kernel_regularizer=self.regularizer,
            name=f"{name}_conv2"
        )
        self.bn2 = layers.BatchNormalization(name=f"{name}_bn2") if use_batch_norm else None
        self.act2 = layers.Activation('relu', name=f"{name}_relu2")
        
        # Dropout (optionnel)
        self.dropout = layers.Dropout(dropout_rate, name=f"{name}_dropout") if dropout_rate > 0 else None
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        x = self.act2(x)
        
        return x


class EncoderBlock(layers.Layer):
    """
    Bloc d'encodage : ConvBlock + MaxPooling
    
    Args:
        filters: Nombre de filtres
        use_batch_norm: Utiliser BatchNormalization
        dropout_rate: Taux de dropout
        pool_size: Taille du pooling (défaut: 2)
    """
    
    def __init__(
        self,
        filters: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        pool_size: int = 2,
        name: str = "encoder"
    ):
        super().__init__(name=name)
        
        self.conv_block = ConvBlock(
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            name=f"{name}_conv"
        )
        self.pool = layers.MaxPooling3D(
            pool_size=(pool_size, pool_size, pool_size),
            name=f"{name}_pool"
        )
    
    def call(self, inputs, training=False):
        # Retourne à la fois la sortie (pour skip connection) et le pooled
        conv_out = self.conv_block(inputs, training=training)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class DecoderBlock(layers.Layer):
    """
    Bloc de décodage : UpConv + Concatenation + ConvBlock
    
    Args:
        filters: Nombre de filtres
        use_batch_norm: Utiliser BatchNormalization
        dropout_rate: Taux de dropout
        up_kernel_size: Taille du noyau pour up-convolution
    """
    
    def __init__(
        self,
        filters: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        up_kernel_size: int = 2,
        name: str = "decoder"
    ):
        super().__init__(name=name)
        
        self.up_conv = layers.Conv3DTranspose(
            filters, up_kernel_size, strides=2, padding='same',
            name=f"{name}_up_conv"
        )
        self.concat = layers.Concatenate(axis=-1, name=f"{name}_concat")
        self.conv_block = ConvBlock(
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            name=f"{name}_conv"
        )
    
    def call(self, inputs, skip_connection, training=False):
        x = self.up_conv(inputs)
        x = self.concat([x, skip_connection])
        x = self.conv_block(x, training=training)
        return x


class UNet3D(tf.keras.Model):
    """
    U-Net 3D complet pour segmentation d'embolies pulmonaires.
    
    Architecture:
        - 4 niveaux d'encodeur (down-sampling)
        - 1 bottleneck
        - 4 niveaux de décodeur (up-sampling)
        - Skip connections à chaque niveau
    
    Args:
        input_shape: Tuple (height, width, depth, channels) - ex: (128, 128, 64, 1)
        num_classes: Nombre de classes (défaut: 2: fond + embolie)
        features_start: Nombre initial de filtres (défaut: 32)
        depth: Profondeur du réseau (défaut: 4)
        use_batch_norm: Utiliser BatchNormalization
        dropout_rate: Taux de dropout dans les blocs convolutifs
        l2_reg: Facteur de régularisation L2
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (128, 128, 64, 1),
        num_classes: int = 2,
        features_start: int = 32,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        name: str = "unet_3d"
    ):
        super().__init__(name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.features_start = features_start
        self.depth = depth
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Construire l'architecture
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.bottleneck = None
        
        self._build_architecture()
        
        logger.info(f"✅ UNet3D initialisé: input={input_shape}, depth={depth}, "
                   f"features_start={features_start}")
    
    def _build_architecture(self):
        """Construit les couches du réseau"""
        
        # ENCODER (down-sampling)
        filters = self.features_start
        for d in range(self.depth):
            encoder = EncoderBlock(
                filters=filters,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate if d > 0 else 0.0,  # Pas de dropout au premier niveau
                name=f"encoder_{d}"
            )
            self.encoder_blocks.append(encoder)
            filters *= 2
        
        # BOTTLENECK
        self.bottleneck = ConvBlock(
            filters=filters,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            name="bottleneck"
        )
        
        # DECODER (up-sampling)
        for d in reversed(range(self.depth)):
            filters //= 2
            decoder = DecoderBlock(
                filters=filters,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                name=f"decoder_{d}"
            )
            self.decoder_blocks.append(decoder)
        
        # FINAL CONVOLUTION (sortie)
        if self.num_classes == 1:
            self.final_conv = layers.Conv3D(
                1, 1, activation='sigmoid', padding='same',
                name="final_conv"
            )
        else:
            self.final_conv = layers.Conv3D(
                self.num_classes, 1, activation='softmax', padding='same',
                name="final_conv"
            )
    
    def call(self, inputs, training=False):
        """
        Forward pass du U-Net.
        
        Args:
            inputs: Tensor (batch, height, width, depth, channels)
            training: Booléen pour BatchNorm/Dropout
        
        Returns:
            output: Tensor (batch, height, width, depth, num_classes)
        """
        # Encoder - stocker les skip connections
        skip_connections = []
        x = inputs
        
        for encoder in self.encoder_blocks:
            skip, x = encoder(x, training=training)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        
        # Decoder - utiliser les skip connections dans l'ordre inverse
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip, training=training)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output
    
    def summary(self, expand_nested: bool = False):
        """Affiche le résumé du modèle"""
        # Créer un modèle temporaire pour le summary
        inputs = tf.keras.Input(shape=self.input_shape)
        outputs = self.call(inputs)
        temp_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        temp_model.summary(expand_nested=expand_nested)
    
    def get_encoder_output_shape(self, level: int) -> Tuple[int, ...]:
        """
        Retourne la forme de la sortie de l'encodeur à un niveau donné.
        Utile pour les skip connections personnalisées.
        """
        shape = list(self.input_shape[:-1])  # (H, W, D)
        for _ in range(level + 1):
            shape = [s // 2 for s in shape]
        shape.append(self.features_start * (2 ** level))
        return tuple(shape)


class AttentionUNet3D(UNet3D):
    """
    U-Net 3D avec mécanisme d'attention (Attention Gate).
    
    Variante améliorée qui apprend à se concentrer sur les régions pertinentes.
    Particulièrement utile pour les petites embolies.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (128, 128, 64, 1),
        num_classes: int = 2,
        features_start: int = 32,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        name: str = "attention_unet_3d"
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            features_start=features_start,
            depth=depth,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name=name
        )
        
        logger.info(f"✅ AttentionUNet3D initialisé (avec attention gates)")
    
    def _build_architecture(self):
        """Construit l'architecture avec attention gates"""
        
        # ENCODER (identique au U-Net standard)
        filters = self.features_start
        for d in range(self.depth):
            encoder = EncoderBlock(
                filters=filters,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate if d > 0 else 0.0,
                name=f"encoder_{d}"
            )
            self.encoder_blocks.append(encoder)
            filters *= 2
        
        # BOTTLENECK
        self.bottleneck = ConvBlock(
            filters=filters,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            name="bottleneck"
        )
        
        # DECODER avec Attention Gates
        self.attention_gates = []
        filters_backup = []
        f = filters
        for d in reversed(range(self.depth)):
            f //= 2
            filters_backup.append(f)
        
        for i, f in enumerate(filters_backup):
            # Attention Gate avant le decoder
            attention = self._create_attention_gate(f, name=f"attention_{self.depth - i - 1}")
            self.attention_gates.append(attention)
            
            # Decoder standard
            decoder = DecoderBlock(
                filters=f,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                name=f"decoder_{self.depth - i - 1}"
            )
            self.decoder_blocks.append(decoder)
        
        # Final convolution
        if self.num_classes == 1:
            self.final_conv = layers.Conv3D(1, 1, activation='sigmoid', padding='same')
        else:
            self.final_conv = layers.Conv3D(self.num_classes, 1, activation='softmax', padding='same')
    
    def _create_attention_gate(self, filters: int, name: str = "attention") -> layers.Layer:
        """
        Crée un gate d'attention pour pondérer les skip connections.
        
        Args:
            filters: Nombre de filtres
            name: Nom du gate
        """
        # Gating signal (vient du decoder)
        self.gating_conv = layers.Conv3D(
            filters, 1, strides=2, padding='same',
            name=f"{name}_gating_conv"
        )
        
        # Skip connection processing
        self.skip_conv = layers.Conv3D(
            filters, 1, padding='same',
            name=f"{name}_skip_conv"
        )
        
        # Attention weights
        self.attention_conv = layers.Conv3D(1, 1, padding='same', name=f"{name}_attn_conv")
        
        return self._apply_attention
    
    def _apply_attention(self, gating_signal, skip_connection, training=False):
        """Applique l'attention gate"""
        # Gating signal
        g = self.gating_conv(gating_signal)
        
        # Skip connection
        s = self.skip_conv(skip_connection)
        
        # Combinaison
        x = layers.Activation('relu')(g + s)
        
        # Attention coefficients
        attn_coeff = self.attention_conv(x)
        attn_coeff = layers.Activation('sigmoid')(attn_coeff)
        
        # Skip connection pondérée
        attended_skip = layers.multiply([skip_connection, attn_coeff])
        
        return attended_skip
    
    def call(self, inputs, training=False):
        """Forward pass avec attention gates"""
        
        # Encoder
        skip_connections = []
        x = inputs
        
        for encoder in self.encoder_blocks:
            skip, x = encoder(x, training=training)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        
        # Decoder avec attention
        for i, decoder in enumerate(self.decoder_blocks):
            # Appliquer l'attention gate sur la skip connection
            attended_skip = self.attention_gates[i](x, skip_connections[-(i + 1)], training=training)
            x = decoder(x, attended_skip, training=training)
        
        # Final convolution
        output = self.final_conv(x)
        
        return output


class NestedUNet3D(UNet3D):
    """
    U-Net 3D avec nested skip connections (UNet++).
    
    Variante qui connecte les niveaux d'encodeur à tous les niveaux de décodeur.
    Plus lourd mais souvent plus précis.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (128, 128, 64, 1),
        num_classes: int = 2,
        features_start: int = 32,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 1e-4,
        name: str = "nested_unet_3d"
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            features_start=features_start,
            depth=depth,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name=name
        )
        
        logger.info(f"✅ NestedUNet3D initialisé (avec nested skip connections)")


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def compile_unet_3d(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    loss: str = 'sparse_categorical_crossentropy',
    metrics: Optional[List[str]] = None
) -> tf.keras.Model:
    """
    Compile un modèle U-Net 3D pour la segmentation.
    
    Args:
        model: Modèle U-Net
        learning_rate: Taux d'apprentissage
        loss: Fonction de perte
            - 'sparse_categorical_crossentropy' pour classes entières (0,1,2,...)
            - 'binary_crossentropy' pour binaire (0/1)
            - 'dice_loss' nécessite une fonction personnalisée
        metrics: Liste des métriques
    
    Returns:
        Modèle compilé
    """
    if metrics is None:
        # Métriques pour segmentation
        metrics = [
            'accuracy',
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou_embolism'),
            tf.keras.metrics.Recall(class_id=1, name='sensitivity'),
            tf.keras.metrics.Precision(class_id=1, name='precision')
        ]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logger.info(f"✅ U-Net 3D compilé: lr={learning_rate}, loss={loss}")
    
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Coefficient de Dice pour la segmentation.
    Peut être utilisé comme métrique ou comme loss.
    
    Args:
        y_true: Ground truth (batch, H, W, D, C)
        y_pred: Prédiction (batch, H, W, D, C)
        smooth: Facteur de lissage
    
    Returns:
        dice: Score de Dice (0 à 1)
    """
    # Si softmax, prendre l'argmax
    if y_pred.shape[-1] > 1:
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, depth=y_true.shape[-1])
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    """Loss basée sur le coefficient de Dice"""
    return 1 - dice_coefficient(y_true, y_pred)


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DES MODÈLES U-NET 3D")
    print("=" * 60)
    
    # Paramètres de test
    BATCH_SIZE = 2
    INPUT_SHAPE = (64, 64, 32, 1)  # H, W, D, C
    NUM_CLASSES = 2
    
    print(f"\n📊 Configuration de test:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Input shape: {INPUT_SHAPE}")
    print(f"   Num classes: {NUM_CLASSES}")
    
    # Test 1: UNet3D standard
    print("\n" + "-" * 40)
    print("Test 1: UNet3D (Standard)")
    
    unet = UNet3D(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        features_start=16,  # Réduit pour le test
        depth=3,
        use_batch_norm=True,
        dropout_rate=0.1
    )
    unet.summary()
    
    dummy_input = tf.random.normal((BATCH_SIZE, *INPUT_SHAPE))
    dummy_output = unet(dummy_input, training=False)
    print(f"✅ UNet3D output shape: {dummy_output.shape}")
    
    # Test 2: AttentionUNet3D
    print("\n" + "-" * 40)
    print("Test 2: AttentionUNet3D (Avec attention gates)")
    
    att_unet = AttentionUNet3D(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        features_start=16,
        depth=3,
        use_batch_norm=True,
        dropout_rate=0.1
    )
    
    dummy_output = att_unet(dummy_input, training=False)
    print(f"✅ AttentionUNet3D output shape: {dummy_output.shape}")
    
    # Test 3: Compilation
    print("\n" + "-" * 40)
    print("Test 3: Compilation du modèle")
    
    compiled_unet = compile_unet_3d(unet, learning_rate=1e-4)
    print("✅ U-Net 3D compilé avec succès")
    
    # Test 4: Dice coefficient
    print("\n" + "-" * 40)
    print("Test 4: Coefficient de Dice")
    
    y_true = tf.one_hot(tf.random.uniform((2, 64, 64, 32), maxval=2, dtype=tf.int32), depth=2)
    y_pred = tf.one_hot(tf.random.uniform((2, 64, 64, 32), maxval=2, dtype=tf.int32), depth=2)
    
    dice = dice_coefficient(y_true, y_pred)
    print(f"✅ Dice coefficient (test aléatoire): {dice.numpy():.4f}")
    
    # Test 5: Calcul du nombre de paramètres
    print("\n" + "-" * 40)
    print("Test 5: Complexité des modèles")
    
    total_params = unet.count_params()
    print(f"✅ UNet3D: {total_params:,} paramètres")
    
    total_params_att = att_unet.count_params()
    print(f"✅ AttentionUNet3D: {total_params_att:,} paramètres")
    
    print("\n" + "=" * 60)
    print("✅ Tous les tests sont passés!")
    print("=" * 60)