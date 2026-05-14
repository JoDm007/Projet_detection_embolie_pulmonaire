# src/O6_evaluation/compare_models.py
"""
Comparaison des modèles classiques (O4) et deep learning (O5)
Répond aux questions de recherche de l'Objectif 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
import json
import sys

from src.O6_evaluation.metrics import calculate_all_metrics, mcnemar_test
from src.O6_evaluation.visualization import plot_roc_curves, plot_comparison_barplot
from src.O4_classical_classifier.predict_classifier import ClassicalClassifierPredictor
from src.O5_deep_learning.predict import DeepLearningPredictor

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Comparateur entre modèles classiques et deep learning.
    C'est le cœur de l'Objectif 6.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.results: Dict[str, Dict[str, float]] = {}
        
        # Initialiser les prédicteurs
        self._init_predictors()
    
    def _init_predictors(self):
        """Initialise les prédicteurs depuis O4 et O5"""
        # Classiques
        self.classical_predictor = ClassicalClassifierPredictor(
            models_dir="results/classical_classifier/models",
            scaler_path="results/classical_classifier/models/scaler.pkl"
        )
        
        # Deep learning
        self.deep_predictor = DeepLearningPredictor(
            model_path="results/deep_learning/models/best_model.h5",
            model_type='cnn_3d'
        )
    
    def evaluate_all(self, test_data_path: str) -> pd.DataFrame:
        """Évalue tous les modèles et retourne un DataFrame comparatif"""
        # ... code ici ...
        pass
    
    def statistical_comparison(self) -> pd.DataFrame:
        """Test de McNemar entre les meilleurs modèles"""
        # ... code ici ...
        pass