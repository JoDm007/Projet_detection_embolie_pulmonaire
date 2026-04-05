# src/O4_classical_classifier/predict_classifier.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassicalClassifierPredictor:
    """
    Utilitaire pour charger des modèles entraînés et faire des prédictions
    """
    
    def __init__(self, models_dir: str, scaler_path: str = None):
        self.models_dir = Path(models_dir)
        self.scaler_path = scaler_path
        self.models: Dict[str, Any] = {}
        self.scaler: StandardScaler = None
        
        logger.info(f"✅ Predictor initialisé : {models_dir}")
    
    def load_models(self):
        """Charge tous les modèles depuis le répertoire"""
        logger.info(f"📥 Chargement des modèles depuis {self.models_dir}...")
        
        # Charger le scaler si disponible
        if self.scaler_path:
            scaler_path = Path(self.scaler_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler chargé : {scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler non trouvé : {scaler_path}")
        
        # Charger tous les modèles .pkl
        model_files = list(self.models_dir.glob('*.pkl'))
        
        for model_file in model_files:
            if 'scaler' in model_file.name:
                continue
            
            model_name = model_file.stem
            model = joblib.load(model_file)
            self.models[model_name] = model
            logger.info(f"✅ {model_name} chargé")
        
        logger.info(f"✅ {len(self.models)} modèles chargés")
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prétraite les données (standardisation)"""
        if self.scaler is None:
            logger.warning("⚠️ Scaler non disponible, utilisation des données brutes")
            return X
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def predict(self, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Fait des prédictions avec un modèle spécifique"""
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        logger.info(f"🚀 Prédiction avec {model_name}...")
        
        X_processed = self.preprocess(X)
        model = self.models[model_name]
        
        y_pred = model.predict(X_processed)
        y_proba = model.predict_proba(X_processed)[:, 1]
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """Évalue un modèle sur un jeu de test"""
        logger.info(f"📊 Évaluation de {model_name}...")
        
        predictions = self.predict(X_test, model_name)
        y_pred = predictions['predictions']
        y_proba = predictions['probabilities']
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auc_pr': average_precision_score(y_test, y_proba),
        }
        
        logger.info(f"✅ {model_name} - AUC={metrics['auc_roc']:.3f}, F1={metrics['f1']:.3f}")
        
        return metrics
    
    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Évalue tous les modèles sur le jeu de test"""
        results = {}
        
        for model_name in self.models.keys():
            metrics = self.evaluate_on_test(X_test, y_test, model_name)
            results[model_name] = metrics
        
        return results
    
    def get_best_model(self, results: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
        """Retourne le meilleur modèle selon l'AUC"""
        best_model = max(results.items(), key=lambda x: x[1]['auc_roc'])
        logger.info(f"🏆 Meilleur modèle: {best_model[0]} (AUC={best_model[1]['auc_roc']:.3f})")
        return best_model[0], best_model[1]['auc_roc']
    
    def save_predictions(self, X: pd.DataFrame, model_name: str, path: str):
        """Sauvegarde les prédictions dans un fichier CSV"""
        predictions = self.predict(X, model_name)
        
        df_pred = pd.DataFrame({
            'candidate_id': X.index if isinstance(X.index, pd.RangeIndex) else X.index,
            'prediction': predictions['predictions'],
            'probability': predictions['probabilities']
        })
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_pred.to_csv(output_path, index=False)
        logger.info(f"💾 Prédictions sauvegardées : {output_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    MODELS_DIR = "results/classical_classifier/models"
    TEST_FEATURES_PATH = "data/processed/radiomics_features_test.csv"
    TEST_LABEL_COLUMN = "has_embolism"
    OUTPUT_DIR = "results/classical_classifier/predictions"
    
    # Créer les répertoires
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialiser le predictor
    predictor = ClassicalClassifierPredictor(
        models_dir=MODELS_DIR,
        scaler_path=f"{MODELS_DIR}/scaler.pkl"
    )
    
    # Charger les modèles
    predictor.load_models()
    
    # Charger les données de test
    logger.info(f"📥 Chargement des données de test depuis {TEST_FEATURES_PATH}...")
    df_test = pd.read_csv(TEST_FEATURES_PATH)
    
    # Identifier les colonnes
    id_col = 'candidate_id' if 'candidate_id' in df_test.columns else None
    feature_cols = [c for c in df_test.columns if c not in [id_col, TEST_LABEL_COLUMN]]
    
    X_test = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = df_test[TEST_LABEL_COLUMN]
    
    # Évaluer tous les modèles
    results = predictor.evaluate_all(X_test, y_test)
    
    # Afficher les résultats
    print("\n" + "="*50)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*50)
    for model, metrics in results.items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Obtenir le meilleur modèle
    best_model_name, best_auc = predictor.get_best_model(results)
    
    # Sauvegarder les prédictions pour le meilleur modèle
    output_path = f"{OUTPUT_DIR}/predictions_{best_model_name}.csv"
    predictor.save_predictions(X_test, best_model_name, output_path)
    
    print("\n" + "="*50)
    print(f"MEILLEUR MODÈLE: {best_model_name}")
    print(f"AUC: {best_auc:.4f}")
    print(f"Prédictions sauvegardées : {output_path}")
    print("="*50)
