# Objectif 4 : Implémentation d'un Classificateur Classique

Ce plan couvre la création d'un notebook `05_classical_classifiers.ipynb` pour l'entraînement et l'évaluation rigoureuse de modèles de Machine Learning (SVM, Random Forest, Régression Logistique, KNN) sur les caractéristiques radiomiques extraites lors de l'Objectif 3.

> [!CAUTION]
> **Bloquant Actuel sur les Données (IMPORTANT)**
> Le fichier `radiomics_features.csv` que nous venons de générer via `extract_all.py` a été exécuté *uniquement* sur les masques d'embolie réels (Ground Truth). Cela signifie que le CSV ne contient **que des exemples de la classe 1 (Embolies vraies)**.
> 
> Un classificateur supervisé a obligatoirement besoin d'exemples de la **classe 0 (Faux Positifs, tissus sains, bruit d'acquisition)** pour s'entraîner à distinguer le vrai du faux.

## User Review Required

Avant de passer à la rédaction complète du code du notebook pour l'Entraînement des modèles, nous devons clarifier le point de la classe "0" (Faux Positifs) :

1. Avez-vous déjà généré des candidats "Faux Positifs" lors de l'**Objectif 2** (Seuillage, morphologie) qu'il faut maintenant extraire avec `extract_all.py` ?
2. Souhaitez-vous que l'on mette en place d'abord un module pour générer automatiquement ces Faux Positifs, ou voulez-vous que l'on rédige *tout de suite* le notebook final d'entraînement (que vous exécuterez plus tard quand vous aurez les faux positifs) ?

## Proposed Changes

Voici comment sera structuré le futur Notebook `05_classical_classifiers.ipynb` :

### 1. Chargement et Préparation des Données
- Lecture de `radiomics_features.csv`.
- Équilibrage des classes (SMOTE ou sous-échantillonnage aléatoire) face au déséquilibre naturel des embolies vs faux positifs.
- Séparation Train/Test (`test_size=0.2` avec `stratify=y`).
- Normalisation avec `StandardScaler`.

### 2. Sélection de Caractéristiques (Feature Selection)
- Utilisation de votre `RadiomicsFeatureSelector` (Variance, Corrélation, Random Forest) calibré sur l'ensemble d'entraînement uniquement pour éviter la fuite de données (Data Leakage).

### 3. Entraînement et Optimisation des Modèles
Nous entraînerons les 4 algorithmes demandés avec validation croisée (GridSearchCV) :
- **SVM (Support Vector Machine)** : Test de noyaux (Linéaire, RBF).
- **Random Forest** : Ajustement du nombre d'arbres et profondeur.
- **Régression Logistique** : Test de régularisation (L1/L2).
- **KNN** : Optimisation du paramètre $k$.

### 4. Évaluation (Objectif 6)
- Calcul complet : Accuracy, Sensibilité (Rappel), Spécificité, Matrice de Confusion.
- Tracé des courbes ROC et calcul de l'AUC-ROC pour comparaison.

## Open Questions

Est-ce que l'architecture de ce plan correspond exactement à vos attentes pour valider votre objectif 4, et surtout, **comment souhaitez-vous gérer la création des "Faux Positifs" dont nous manquons actuellement pour entraîner l'algorithme ?**

## Verification Plan

- Exécution du code dans l'environnement Jupyter.
- Affichage comparatif (Dataframe) des performances (ROC/Sensibilité) des 4 modèles.
