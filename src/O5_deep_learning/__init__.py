# src/O5_deep_learning/__init__.py
"""
Objectif 5 — Deep Learning pour la détection et segmentation de l'embolie pulmonaire.

Pipeline principal (PyTorch, nnU-Net-style) :
    from src.O5_deep_learning.nnunet_pe import (
        NNUNetPreprocessor,
        NNUNetModel,
        NNUNetDataset,
        NNUNetTrainer,
        NNUNetPredictor,
    )

Les fichiers *_keras.py sont des archives de l'implémentation TensorFlow/Keras initiale.
Ils sont conservés pour référence mais ne sont plus le pipeline actif.
"""

# Import conditionnel pour éviter une erreur si PyTorch n'est pas installé
try:
    from src.O5_deep_learning.nnunet_pe import (
        NNUNetPreprocessor,
        NNUNetModel,
        NNUNetDataset,
        NNUNetTrainer,
        NNUNetPredictor,
        combined_loss,
        dice_loss,
    )
    __all__ = [
        "NNUNetPreprocessor",
        "NNUNetModel",
        "NNUNetDataset",
        "NNUNetTrainer",
        "NNUNetPredictor",
        "combined_loss",
        "dice_loss",
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"src.O5_deep_learning : import partiel ({e}). "
        "Installer PyTorch : pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
