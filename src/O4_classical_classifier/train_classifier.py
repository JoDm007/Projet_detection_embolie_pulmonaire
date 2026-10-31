# src/O4_classical_classifier/train_classifier.py
#
# Phase 3 (2026-10) — cf. docs/planning/plan1.md :
#   - Validation croisée GROUPÉE PAR PATIENT (StratifiedGroupKFold) : un patient est
#     entièrement en train OU en validation (avant : fuite inter-patients via StratifiedKFold).
#   - PatientID / candidate_id exclus des features.
#   - Métriques adaptées à ~1.7 % de VP : AUC-PR, et surtout FROC = sensibilité patient
#     à budget de FP/patient fixé (1, 2, 4, 8). Indépendant de l'échelle des probabilités,
#     donc comparable entre modèles (RF sort des probas basses, LogReg 'balanced' des hautes).
#     Critère de sélection : sensibilité patient @ 4 FP/patient. L'accuracy n'est plus utilisée.
#   - Prédictions out-of-fold sauvegardées pour analyse.
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, List
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Budgets de faux positifs par patient pour la FROC
FP_BUDGETS = [1, 2, 4, 8]
SELECTION_BUDGET = 4  # le meilleur modèle est celui qui détecte le plus de patients à 4 FP/patient


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
                 group_column: str = 'PatientID',
                 n_carriers: int = None,
                 train_ratio: float = 0.7,
                 test_ratio: float = 0.3,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.features_path = features_path
        self.label_column = label_column
        self.group_column = group_column
        # Nombre total de porteurs (GT non vide), y compris ceux sans VP candidat :
        # permet la sensibilité patient de bout en bout (O2 + O4). Non déductible du CSV.
        self.n_carriers = n_carriers
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.random_state = random_state

        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.operating_points: Dict[str, pd.DataFrame] = {}
        self.oof: pd.DataFrame = None
        self.scaler: StandardScaler = StandardScaler()
        # Modèles de chaque fold + fold de test de chaque patient : permettent à l'application
        # (backend/engines/classical.py) d'analyser un cas FUMPE avec un modèle qui ne l'a pas vu.
        self.fold_models: Dict[str, List[Dict[str, Any]]] = {}
        self.fold_assignments: Dict[str, int] = {}

        logger.info(f"✅ Classificateur initialisé : {features_path}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Charge les données et sépare features / labels / groupes (patient)"""
        logger.info(f"📥 Chargement des données depuis {self.features_path}...")
        df = pd.read_csv(self.features_path)

        if self.group_column not in df.columns:
            raise ValueError(f"Colonne '{self.group_column}' absente : impossible de grouper "
                             f"la CV par patient. Régénère le CSV avec build_ml_dataset.py.")

        # Exclure identifiants et label des features
        excluded = {self.label_column, self.group_column, 'candidate_id'}
        feature_cols = [c for c in df.columns if c not in excluded]

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df[self.label_column].astype(int)
        groups = df[self.group_column]
        self.ids = df[[self.group_column] + (['candidate_id'] if 'candidate_id' in df.columns else [])]

        logger.info(f"📊 Features: {X.shape[1]}, Candidats: {X.shape[0]}, VP: {y.sum()} "
                    f"({y.mean():.2%}), Patients: {groups.nunique()}, "
                    f"Patients avec >=1 VP: {groups[y == 1].nunique()}")

        return X, y, groups

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
            # Équivalent de class_weight='balanced' + profondeur réduite (peu de positifs)
            n_pos = max(int(y_train.sum()), 1)
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=(len(y_train) - n_pos) / n_pos,
                eval_metric='aucpr',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Modèle '{model_name}' non supporté")

        model.fit(X_train, y_train)
        logger.info(f"✅ {model_name} entraîné")

        return model

    @staticmethod
    def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        proba_matrix = model.predict_proba(X)
        if proba_matrix.shape[1] == 1:
            # Le modèle n'a vu qu'une classe — on retourne des probas constantes
            return np.full(len(X), float(model.classes_[0] == 1))
        return proba_matrix[:, 1]

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue un modèle sur un jeu de test (métriques niveau candidat, seuil 0.5)"""
        y_proba = self._positive_proba(model, X_test)
        return self._candidate_metrics(y_test, y_proba)

    @staticmethod
    def _candidate_metrics(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:
        y_pred = (y_proba >= threshold).astype(int)
        has_both = len(np.unique(y_true)) == 2
        return {
            'auc_pr': average_precision_score(y_true, y_proba) if has_both else np.nan,
            'auc_roc': roc_auc_score(y_true, y_proba) if has_both else np.nan,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

    def froc_table(self, y: pd.Series, proba: np.ndarray, groups: pd.Series) -> pd.DataFrame:
        """
        FROC : pour chaque budget B (FP/patient), on prend le seuil le plus bas tel que
        FP total <= B x nb_patients, et on mesure la sensibilité à ce seuil.
        Un patient est détecté si l'un de ses VP dépasse le seuil.
        - sens_patients_O4      : sur les patients ayant >=1 VP candidat (O4 seul)
        - sens_patients_globale : sur tous les porteurs (O2 + O4), si n_carriers est connu
        """
        y = np.asarray(y)
        groups = np.asarray(groups)
        proba = np.asarray(proba)
        n_patients = len(np.unique(groups))
        neg_sorted = np.sort(proba[y == 0])[::-1]
        pos = proba[y == 1]
        # Meilleure probabilité VP de chaque patient : détecté si elle passe le seuil
        patient_max = pd.Series(pos).groupby(groups[y == 1]).max().values

        rows = []
        for budget in FP_BUDGETS:
            k = int(budget * n_patients)  # nb de FP autorisés au total
            # Seuil strict : on garde les candidats > k-ième plus forte proba négative
            t = neg_sorted[k] if k < len(neg_sorted) else -np.inf
            detected = int((patient_max > t).sum())
            row = {
                'FP_par_patient_max': budget,
                'FP_par_patient_reel': (neg_sorted > t).sum() / n_patients,
                'seuil': t,
                'sens_candidats': (pos > t).sum() / max(len(pos), 1),
                'patients_detectes': f"{detected}/{len(patient_max)}",
                'sens_patients_O4': detected / max(len(patient_max), 1),
            }
            if self.n_carriers:
                row['sens_patients_globale'] = detected / self.n_carriers
            rows.append(row)
        return pd.DataFrame(rows)

    def train_with_cross_validation(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                                    groups: pd.Series = None) -> Dict[str, Any]:
        """
        CV groupée par patient. Retourne les métriques calculées sur les probabilités
        out-of-fold (OOF) poolées + la moyenne/écart-type de l'AUC-PR par fold.
        """
        if groups is None:
            raise ValueError("groups (PatientID) requis : la CV doit être groupée par patient.")
        logger.info(f"🔄 Validation croisée groupée par patient pour {model_name}...")
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(y), dtype=float)
        fold_auc_pr = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # ✅ Fit du scaler UNIQUEMENT sur le train fold
            scaler_fold = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler_fold.fit_transform(X_train), columns=X.columns)
            X_val_scaled = pd.DataFrame(scaler_fold.transform(X_val), columns=X.columns)

            model = self.train_model(model_name, X_train_scaled, y_train)
            oof_proba[val_idx] = self._positive_proba(model, X_val_scaled)
            self.fold_models.setdefault(model_name, []).append({'scaler': scaler_fold, 'model': model})
            for pid in groups.iloc[val_idx].unique():
                self.fold_assignments[pid] = fold
            m = self._candidate_metrics(y_val, oof_proba[val_idx])
            fold_auc_pr.append(m['auc_pr'])

            logger.info(f"  Fold {fold+1}: {groups.iloc[val_idx].nunique()} patients, "
                        f"{int(y_val.sum())} VP / {len(y_val)} | AUC-PR={m['auc_pr']:.3f}, "
                        f"AUC-ROC={m['auc_roc']:.3f}")

        # Précision/rappel au seuil 0.5 non retenus : l'échelle des probas diffère entre modèles
        full = self._candidate_metrics(y, oof_proba)
        froc = self.froc_table(y, oof_proba, groups)
        sel = froc[froc['FP_par_patient_max'] == SELECTION_BUDGET].iloc[0]
        metrics = {
            f'sens_patients_O4@{SELECTION_BUDGET}FP': float(sel['sens_patients_O4']),
            'auc_pr': full['auc_pr'],
            'auc_pr_fold_mean': float(np.nanmean(fold_auc_pr)),
            'auc_pr_fold_std': float(np.nanstd(fold_auc_pr)),
            'auc_pr_hasard': float(y.mean()),  # AUC-PR d'un classifieur aléatoire
            'auc_roc': full['auc_roc'],
        }
        if self.n_carriers:
            metrics[f'sens_patients_globale@{SELECTION_BUDGET}FP'] = float(sel['sens_patients_globale'])

        self.operating_points[model_name] = froc
        self.oof[f'proba_{model_name}'] = oof_proba

        logger.info(f"✅ {model_name} - OOF: AUC-PR={metrics['auc_pr']:.3f} "
                    f"(hasard={metrics['auc_pr_hasard']:.3f}), "
                    f"patients @{SELECTION_BUDGET}FP = {sel['patients_detectes']}")
        return metrics

    def train_final_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Entraîne le modèle final sur toutes les données"""
        logger.info(f"🚀 Entraînement final de {model_name}...")

        model = self.train_model(model_name, X, y)
        self.models[model_name] = model

        return model

    def train_all(self, models: list = None) -> Dict[str, Any]:
        if models is None:
            models = ['random_forest', 'svm', 'logistic_regression', 'xgboost']

        X, y, groups = self.load_data()
        self.oof = self.ids.copy()
        self.oof[self.label_column] = y.values
        # ⚠️ Ne PAS scaler ici avant la CV. Le scaler final sera fit sur tout X après la CV.
        results = {}

        for model_name in models:
            results[model_name] = self.train_with_cross_validation(model_name, X, y, groups)

        # ✅ Fit final sur TOUT le dataset pour sauvegarde du modèle de prod
        self.scaler.fit(X)
        X_final = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        for model_name in models:
            self.train_final_model(model_name, X_final, y)

        self.results = results
        return results

    def get_best_model(self) -> Tuple[str, Any]:
        """Meilleur modèle : sensibilité patient @ SELECTION_BUDGET FP/patient, puis AUC-PR"""
        key = f'sens_patients_O4@{SELECTION_BUDGET}FP'
        best_model = max(self.results.items(), key=lambda x: (x[1][key], x[1]['auc_pr']))
        logger.info(f"🏆 Meilleur modèle: {best_model[0]} ({key}={best_model[1][key]:.3f}, "
                    f"AUC-PR={best_model[1]['auc_pr']:.3f})")
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

        # Modèles par fold (même split pour tous les modèles : random_state fixe)
        if self.fold_models:
            folds_path = output_path / 'folds'
            folds_path.mkdir(exist_ok=True)
            for name, bundles in self.fold_models.items():
                for k, bundle in enumerate(bundles):
                    joblib.dump(bundle, folds_path / f'{name}_fold{k}.pkl')
            pd.DataFrame(sorted(self.fold_assignments.items()), columns=['PatientID', 'fold']) \
                .to_csv(folds_path / 'fold_assignments.csv', index=False)
            logger.info(f"💾 Modèles par fold sauvegardés : {folds_path}")

    def save_results(self, path: str):
        """Sauvegarde les métriques, les points de fonctionnement et les prédictions OOF"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_results = pd.DataFrame.from_dict(self.results, orient='index')
        df_results.to_csv(output_path, float_format='%.4f')
        logger.info(f"💾 Résultats sauvegardés : {output_path}")

        op_path = output_path.with_name('froc.csv')
        pd.concat(self.operating_points, names=['model']).to_csv(op_path, float_format='%.4f')
        logger.info(f"💾 FROC sauvegardée : {op_path}")

        oof_path = output_path.with_name('oof_predictions.csv')
        self.oof.to_csv(oof_path, index=False, float_format='%.5f')
        logger.info(f"💾 Prédictions out-of-fold sauvegardées : {oof_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset_balanced.csv"
    RESULTS_DIR = PROJECT_ROOT / "results" / "classical_classifier"
    # Porteurs (GT non vide) : valeur loguée par build_ml_dataset.py ("Porteurs avec >=1 VP : 24/33")
    N_CARRIERS = 33
    models_arg = sys.argv[1:] or None  # ex: python ... random_forest xgboost

    # Créer le répertoire de résultats
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialiser et entraîner les classificateurs
    trainer = ClassicalClassifierTrainer(features_path=str(FEATURES_PATH), n_carriers=N_CARRIERS)
    results = trainer.train_all(models_arg)

    # Afficher les meilleurs résultats
    best_model_name, best_model = trainer.get_best_model()

    # Sauvegarder les modèles et résultats
    trainer.save_models(str(RESULTS_DIR / "models"))
    trainer.save_results(str(RESULTS_DIR / "results.csv"))

    print("\n" + "="*60)
    print("RÉSUMÉ (métriques out-of-fold, CV groupée par patient)")
    print("="*60)
    for model, metrics in results.items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("  FROC (sensibilité à budget de FP/patient fixé) :")
        print(trainer.operating_points[model].to_string(index=False, float_format='%.3f'))

    print("\n" + "="*60)
    print("COMPARAISON @ {} FP/patient".format(SELECTION_BUDGET))
    print(pd.DataFrame.from_dict(results, orient='index').to_string(float_format='%.3f'))
    print(f"\nMEILLEUR MODÈLE: {best_model_name}")
    print("sens_patients_O4      = sur les patients ayant >=1 VP candidat (O4 seul)")
    print(f"sens_patients_globale = sur les {N_CARRIERS} porteurs (O2 + O4, de bout en bout)")
    print("="*60)
