#Auteur: Joseph DATE-MASSE

# src/O3_radiomics/feature_selector.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Optional, Dict
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadiomicsFeatureSelector:
    """
    Sélection de features radiomiques pour éviter le sur-apprentissage
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 top_k_features: int = 20,
                 selection_method: str = 'random_forest'):
        self.var_thresh = variance_threshold
        self.corr_thresh = correlation_threshold
        self.top_k = top_k_features
        self.selection_method = selection_method
        
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"✅ Sélecteur initialisé : variance={variance_threshold}, "
                   f"correlation={correlation_threshold}, top_k={top_k_features}")
    
    def filter_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Étape 1 : Supprime les features quasi-constantes"""
        if df.empty:
            return df
        
        id_col = 'candidate_id' if 'candidate_id' in df.columns else None
        feature_cols = [c for c in df.columns if c != id_col]
        
        df_clean = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        selector = VarianceThreshold(threshold=self.var_thresh)
        try:
            selector.fit(df_clean)
            keep_mask = selector.get_support()
            kept_features = [feature_cols[i] for i in range(len(feature_cols)) if keep_mask[i]]
            
            logger.info(f"📊 Variance: {len(feature_cols)} → {len(kept_features)} features")
            
            result_cols = [id_col] + kept_features if id_col else kept_features
            return df[result_cols]
            
        except Exception as e:
            logger.error(f"⚠️ Erreur filtrage variance: {e}")
            return df
    
    def filter_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Étape 2 : Supprime les features fortement corrélées"""
        if df.empty:
            return df
        
        id_col = 'candidate_id' if 'candidate_id' in df.columns else None
        feature_cols = [c for c in df.columns if c != id_col]
        
        if len(feature_cols) < 2:
            return df
        
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_thresh)]
        
        kept_features = [c for c in feature_cols if c not in to_drop]
        
        logger.info(f"📊 Corrélation (>{self.corr_thresh}): {len(feature_cols)} → {len(kept_features)} features")
        
        result_cols = [id_col] + kept_features if id_col else kept_features
        return df[result_cols]
    
    def select_by_random_forest(self, df: pd.DataFrame, y: pd.Series) -> List[str]:
        """Étape 3 : Sélection par importance Random Forest"""
        id_col = 'candidate_id' if 'candidate_id' in df.columns else None
        feature_cols = [c for c in df.columns if c != id_col]
        
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            self.feature_importance = {feature_cols[i]: float(importances[i]) 
                                       for i in range(len(feature_cols))}
            
            top_indices = importances.argsort()[-self.top_k:][::-1]
            top_features = [feature_cols[i] for i in top_indices]
            
            logger.info(f"📊 Random Forest: {len(feature_cols)} → {len(top_features)} features")
            
            self.selected_features = top_features
            return top_features
            
        except Exception as e:
            logger.error(f"⚠️ Erreur Random Forest: {e}")
            return feature_cols[:self.top_k]
    
    def select(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Pipeline complet de sélection de features"""
        logger.info("🚀 Démarrage sélection de features...")
        original_cols = len(df.columns)
        
        df = self.filter_low_variance(df)
        df = self.filter_high_correlation(df)
        
        if y is not None:
            top_features = self.select_by_random_forest(df, y)
            
            id_col = 'candidate_id' if 'candidate_id' in df.columns else None
            keep_cols = [id_col] + top_features if id_col else top_features
            df = df[keep_cols]
        
        final_cols = len(df.columns)
        logger.info(f"✅ Sélection terminée: {original_cols} → {final_cols} features")
        
        return df
    
    def save(self, path: str):
        """Sauvegarde le sélecteur"""
        data = {
            'var_thresh': self.var_thresh,
            'corr_thresh': self.corr_thresh,
            'top_k': self.top_k,
            'selection_method': self.selection_method,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"💾 Sélecteur sauvegardé : {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RadiomicsFeatureSelector':
        """Charge un sélecteur sauvegardé"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        selector = cls(
            variance_threshold=data['var_thresh'],
            correlation_threshold=data['corr_thresh'],
            top_k_features=data['top_k'],
            selection_method=data['selection_method']
        )
        selector.selected_features = data['selected_features']
        selector.feature_importance = data['feature_importance']
        
        logger.info(f"📂 Sélecteur chargé : {path}")
        return selector
    
    def plot_importance(self, top_n: int = 15, save_path: str = None):
        """
        Visualise l'importance des features (NOUVEAU - pour ton code)
        """
        import matplotlib.pyplot as plt
        
        if not self.feature_importance:
            logger.warning("⚠️ Aucune importance disponible")
            return
        
        # Trier par importance
        sorted_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), [f[:40] for f in features])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features Radiomiques')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Graphique sauvegardé : {save_path}")
        
        plt.show()


# ═══════════════════════════════════════════════════════════════
# FONCTION UTILITAIRE (pour le code)
# ═══════════════════════════════════════════════════════════════

def select_features(X_train: pd.DataFrame, 
                    y_train: pd.Series, 
                    X_test: pd.DataFrame,
                    top_k_features: int = 20,
                    selection_method: str = 'random_forest',
                    variance_threshold: float = 0.01,
                    correlation_threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame, RadiomicsFeatureSelector]:
    """
    Fonction utilitaire pour sélectionner les features (Train + Test)
    
    Usage:
        X_train_sel, X_test_sel, selector = select_features(X_train, y_train, X_test)
    """
    logger.info("🚀 Sélection de features pour ML...")
    
    selector = RadiomicsFeatureSelector(
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        top_k_features=top_k_features,
        selection_method=selection_method
    )
    
    X_train_sel = selector.select(X_train.copy(), y_train)
    
    feature_cols = [c for c in X_train_sel.columns if c not in ['candidate_id', 'PatientID']]
    X_test_sel = X_test[feature_cols].copy()
    
    logger.info(f"✅ Sélection terminée: {len(X_train.columns)} → {len(feature_cols)} features")
    
    return X_train_sel, X_test_sel, selector