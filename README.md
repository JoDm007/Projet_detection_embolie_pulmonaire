Voici la feuille de route complète du projet. Elle lie chaque **Objectif** aux **fichiers** `src` et aux **Notebooks** correspondants

---

```
FUMPE_Detection/
├── config.yaml                    # Paramètres globaux
├── setup.py                       # Installation du package
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── CT_scans_01/
│   │   ├── CT_scans_02/
│   │   ├── CT_scans_03/
│   │   └── GroundTruth/
│   ├── processed/                 # Sera rempli par preprocessing
│   ├── splits/                     # Sera créé par le script de split
│   └── metadata/
│
├── src/
│   ├── O1_preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py              # Chargement des données
│   │   ├── normalisation.py        # Fenêtrage Hounsfield
│   │   ├── resampling.py          # Redimensionnement
│   │   └── split_data.py          # Train/Val/Test split
│   │
│   ├── O2_classical_detection/
│   │   ├── __init__.py
│   │   ├── thresholding.py        # Seuillage adaptatif
│   │   ├── morphology.py          # Morphologie mathématique
│   │   └── connected_components.py # Composantes connexes 3D
│   │
│   ├── O3_radiomics/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py   # Extraction des caractéristiques
│   │   └── feature_selector.py    # Sélection des features pertinentes
│   │
│   ├── O4_classical_classifier/
│   │   ├── __init__.py
│   │   ├── train_classifier.py    # SVM, Random Forest, etc.
│   │   └── predict_classifier.py
│   │
│   ├── O5_deep_learning/
│   │   ├── __init__.py
│   │   ├── cnn_2d.py              # CNN 2D slice-based
│   │   ├── cnn_3d.py              # CNN 3D
│   │   ├── unet_3d.py             # U-Net 3D pour segmentation
│   │   ├── data_generator.py      # Générateur pour entraînement
│   │   └── train.py
│   │
│   ├── O6_evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Dice, Sensibilité, Spécificité, etc.
│   │   ├── statistical_tests.py   # Test de McNemar, etc.
│   │   └── visualization.py       # Courbes ROC, exemples
│   │
│   ├── O7_optimization/
│   │   ├── __init__.py
│   │   ├── data_augmentation.py   # Augmentation 3D
│   │   ├── class_balancing.py     # Équilibrage des classes
│   │   └── hyperparameter_tuning.py # Grid search, etc.
│   │
│   ├── O8_clinical_validation/
│   │   ├── __init__.py
│   │   ├── size_analysis.py       # Performance selon taille des lésions
│   │   └── limitations.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py       # Visualisation 3D/2D
│       ├── io.py                   # Chargement/sauvegarde
│       └── logger.py
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_classical_detection_demo.ipynb
│   ├── 04_radiomics_demo.ipynb
│   ├── 05_classical_classifier_demo.ipynb
│   ├── 06_deep_learning_demo.ipynb
│   ├── 07_evaluation_demo.ipynb
│   └── 08_clinical_analysis.ipynb
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_classical_pipeline.py
│   ├── train_deep_model.py
│   └── evaluate_all.py
│
├── models/                         # Modèles sauvegardés
├── results/                        # Métriques, figures
├── logs/                           # Logs d'entraînement
├── tests/                          # Tests unitaires
├── docs/                           # Documentation
└── figures/                        # Figures pour le rapport
```

## 🗺️ Cartographie du Projet par Objectif

### Objectif 1 : Prétraitement & Organisation

* **Fichiers `src/O1_preprocessing/**` : `loader.py` (Chargement), `normalisation.py` (Fenêtrage), `resampling.py` (Iso-résolution), `split_data.py` (Train/Val/Test).
* **Notebook** : `01_exploration.ipynb` (Visualisation brute) et `02_preprocessing_demo.ipynb` (Validation du pipeline).
* **Résultat** : Nos données passeneront de `data/raw/` (.dcm/.mat) à `data/processed/` (.npy).

### Objectif 2 : Détection Classique (Vaisseaux & PE)

* **Fichiers `src/O2_classical_detection/**` : `thresholding.py` (Segmentation des poumons), `morphology.py` (Nettoyage), `connected_components.py` (Isoler les caillots potentiels).
* **Notebook** : `03_classical_detection_demo.ipynb`.
* **Résultat** : Une première carte de détection sans IA, basée sur les formes et les densités.

### Objectif 3 : Extraction de Caractéristiques (Radiomique)

* **Fichiers `src/O3_radiomics/**` : `feature_extractor.py` (Utilisation de `PyRadiomics`), `feature_selector.py`.
* **Notebook** : `04_radiomics_demo.ipynb`.
* **Résultat** : Un fichier CSV contenant des descripteurs (texture, forme) pour chaque zone suspecte.

### Objectif 4 : Machine Learning Traditionnel

* **Fichiers `src/O4_classical_classifier/**` : `train_classifier.py` (Random Forest ou SVM), `predict_classifier.py`.
* **Notebook** : `05_classical_classifier_demo.ipynb`.
* **Résultat** : Un modèle capable de dire "Embolie ou non" basé sur les caractéristiques de l'Objectif 3.

### Objectif 5 : Deep Learning (Segmentation/Détection)

* **Fichiers `src/O5_deep_learning/**` : `unet_3d.py` (Architecture), `data_generator.py` (Lecture des .npy), `train.py`.
* **Notebook** : `06_deep_learning_demo.ipynb`.
* **Résultat** : Le modèle de pointe (U-Net ou CNN) entraîné.

### Objectif 6 : Évaluation des Performances

* **Fichiers `src/O6_evaluation/**` : `metrics.py` (Dice, Sensibilité, Spécificité), `visualization.py` (Courbes ROC).
* **Notebook** : `07_evaluation_demo.ipynb`.
* **Résultat** : Les chiffres finaux de ton projet (ex: "Mon IA détecte 92% des embolies").

### Objectif 7 : Optimisation

* **Fichiers `src/O7_optimization/**` : `data_augmentation.py` (Rotation/Flip), `hyperparameter_tuning.py`.
* **Résultat** : Une version améliorée et plus robuste de ton modèle Deep Learning.

### Objectif 8 : Analyse Clinique

* **Fichiers `src/O8_clinical_validation/**` : `size_analysis.py`.
* **Notebook** : `08_clinical_analysis.ipynb`.
* **Résultat** : Rapport sur les limites du modèle (ex: "L'IA a du mal sur les embolies distales de moins de 2mm").

---

Pour faire  le **push** sur Git

### 1. Le cas classique (le plus courant)

déjà lié le dépôt et que tu es sur la branche `main` (ou `master`) :

```bash
git push origin main

```

---

### 2. Si c'est ton tout premier push

Si tu viens de créer le dépôt et que tu dois configurer la branche distante par défaut :

```bash
git set-upstream origin main
# OU plus court :
git push -u origin main

```

*L'option `-u` permet de mémoriser la destination. La prochaine fois, tu n'auras qu'à taper `git push`.*

---

### 3. Le flux complet (Rappel)

N'oublie pas que pour que le push fonctionne, tes fichiers doivent être indexés et "commités" :

1. **`git add .`** (Prépare tous les fichiers)
2. **`git commit -m "Ton message ici"`** (Enregistre les modifications)
3. **`git push origin main`** (Envoie sur le serveur)
