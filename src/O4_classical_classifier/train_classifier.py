# src/O4_classical_classifier/train_classifier.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassicalClassifierTrainer:
    """
    Entraînement de classificateurs classiques sur features radiomiques
    
    Modèles supportés :
    - Random Forest
    - SVM
    - Logistic Regression
    - XGBoost
    """
    
    def __init__(self, 
                 features_path: str, 
                 label_column: str = 'has_embolism',
                 train_ratio: float = 0.7,
                 test_ratio: float = 0.3,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.features_path = features_path
        self.label_column = label_column
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.scaler: StandardScaler = StandardScaler()
        
        logger.info(f"✅ Classificateur initialisé : {features_path}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Charge les données et sépare features/labels"""
        logger.info(f"📥 Chargement des données depuis {self.features_path}...")
        df = pd.read_csv(self.features_path)
        
        # Identifier les colonnes de features (exclure ID et label)
        id_col = 'candidate_id' if 'candidate_id' in df.columns else None
        feature_cols = [c for c in df.columns if c not in [id_col, self.label_column]]
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[self.label_column]
        
        logger.info(f"📊 Features: {X.shape[1]}, Samples: {X.shape[0]}, Positives: {y.sum()}")
        
        return X, y
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardise les features"""
        logger.info("🔄 Standardisation des features...")
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Entraîne un modèle spécifique"""
        logger.info(f"🚀 Entraînement de {model_name}...")
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200, 
                random_state=self.random_state, 
                class_weight='balanced', 
                n_jobs=-1
            )
        elif model_name == 'svm':
            model = SVC(
                C=1.0, 
                kernel='rbf', 
                class_weight='balanced', 
                probability=True, 
                random_state=self.random_state
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                class_weight='balanced', 
                random_state=self.random_state, 
                max_iter=1000
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=200, 
                max_depth=10, 
                learning_rate=0.1, 
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Modèle '{model_name}' non supporté")
        
        model.fit(X_train, y_train)
        logger.info(f"✅ {model_name} entraîné")
        
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue un modèle sur le jeu de test"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auc_pr': average_precision_score(y_test, y_proba),
        }
        
        return metrics
    
    def train_with_cross_validation(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Entraînement avec validation croisée stratifiée"""
        logger.info(f"🔄 Validation croisée pour {model_name}...")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.train_model(model_name, X_train, y_train)
            metrics = self.evaluate_model(model, X_val, y_val)
            fold_metrics.append(metrics)
            
            logger.info(f"  Fold {fold+1}: AUC={metrics['auc_roc']:.3f}, F1={metrics['f1']:.3f}")
        
        # Moyenne des métriques
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        avg_metrics['std_auc'] = np.std([m['auc_roc'] for m in fold_metrics])
        
        logger.info(f"✅ {model_name} - Moyenne: AUC={avg_metrics['auc_roc']:.3f}, F1={avg_metrics['f1']:.3f}")
        
        return avg_metrics
    
    def train_final_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Entraîne le modèle final sur toutes les données"""
        logger.info(f"🚀 Entraînement final de {model_name}...")
        
        model = self.train_model(model_name, X, y)
        self.models[model_name] = model
        
        return model
    
    def train_all(self, models: list = None) -> Dict[str, Any]:
        """Entraîne tous les modèles"""
        if models is None:
            models = ['random_forest', 'svm', 'logistic_regression', 'xgboost']
        
        X, y = self.load_data()
        X = self.preprocess(X)
        
        results = {}
        
        for model_name in models:
            # Validation croisée
            cv_metrics = self.train_with_cross_validation(model_name, X, y)
            results[model_name] = cv_metrics
            
            # Entraînement final
            self.train_final_model(model_name, X, y)
        
        self.results = results
        return results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Retourne le meilleur modèle selon l'AUC"""
        best_model = max(self.results.items(), key=lambda x: x[1]['auc_roc'])
        logger.info(f"🏆 Meilleur modèle: {best_model[0]} (AUC={best_model[1]['auc_roc']:.3f})")
        return best_model[0], self.models[best_model[0]]
    
    def save_models(self, path: str):
        """Sauvegarde tous les modèles entraînés"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_path / f'{name}.pkl'
            joblib.dump(model, model_path)
            logger.info(f"💾 {name} sauvegardé : {model_path}")
        
        # Sauvegarder le scaler
        scaler_path = output_path / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"💾 Scaler sauvegardé : {scaler_path}")
    
    def save_results(self, path: str):
        """Sauvegarde les résultats d'évaluation"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_results = pd.DataFrame.from_dict(self.results, orient='index')
        df_results.to_csv(output_path)
        logger.info(f"💾 Résultats sauvegardés : {output_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    FEATURES_PATH = "data/processed/radiomics_features_selected.csv"
    RESULTS_DIR = "results/classical_classifier"
    
    # Créer le répertoire de résultats
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialiser et entraîner les classificateurs
    trainer = ClassicalClassifierTrainer(features_path=FEATURES_PATH)
    results = trainer.train_all()
    
    # Afficher les meilleurs résultats
    best_model_name, best_model = trainer.get_best_model()
    
    # Sauvegarder les modèles et résultats
    trainer.save_models(f"{RESULTS_DIR}/models")
    trainer.save_results(f"{RESULTS_DIR}/results.csv")
    
    print("\n" + "="*50)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*50)
    for model, metrics in results.items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*50)
    print(f"MEILLEUR MODÈLE: {best_model_name}")
    print(f"AUC: {results[best_model_name]['auc_roc']:.4f}")
    print(f"F1-score: {results[best_model_name]['f1']:.4f}")
    print("="*50)
    
