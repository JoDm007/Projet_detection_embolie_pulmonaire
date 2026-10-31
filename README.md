# PE-Seg AI — Détection et segmentation de l'embolie pulmonaire

> **Transformer un volume complexe d'images en information clinique exploitable.**

PE-Seg AI est une **seconde paire d'yeux pour le radiologue**. Sur un angioscanner pulmonaire (CTPA),
l'IA analyse tout le volume 3D et signale les zones suspectes d'embolie : **où** elles sont, **avec quelle
confiance**, et à terme **leur forme et leur volume**. Elle ne remplace pas la lecture du radiologue.

Projet présenté au hackathon **AI4Youth** (Neuractif), équipe de 5.

---

## Sommaire

1. [Où en est le projet](#où-en-est-le-projet)
2. [Données](#données)
3. [Arborescence](#arborescence)
4. [Installation](#installation)
5. [Pipeline de recherche (O1 → O8)](#pipeline-de-recherche-o1--o8)
6. [Application locale (démo)](#application-locale-démo)
7. [Résultats mesurés](#résultats-mesurés)
8. [Documentation](#documentation)
9. [Travailler sur le dépôt](#travailler-sur-le-dépôt)

---

## Où en est le projet

*État au 31 octobre 2026. Les chiffres sont mesurés en validation croisée groupée par patient.*

| Objectif | Contenu | État |
|---|---|---|
| **O1** Prétraitement | Lecture DICOM + masques, fenêtre HU `[-600, 400]`, normalisation | ✅ Fait (orientation du GT corrigée) |
| **O2** Détection classique | Candidats « brillants » dans les poumons | ✅ Fait — au moins un candidat sur l'embolie chez **73 %** des porteurs |
| **O3** Radiomique | 20 features (intensité, forme, texture) par candidat | ✅ Fait — dataset sans fuite de vérité terrain |
| **O4** Classifieur ML | Régression logistique, RF, XGBoost, SVM | ✅ Baseline figée — **45 %** des porteurs détectés à 4 fausses alertes/patient |
| **O5** Deep learning | U-Net 3D pour la segmentation voxel par voxel | ❌ Architecture codée, **pas encore entraînée** |
| **O6–O8** | Évaluation comparative, optimisation, analyse clinique | ❌ À faire |
| **Application** | Interface web locale, hors ligne et sécurisée | 🟡 v0.1 écrite (moteur classique), à valider de bout en bout |

**Prochaine étape :** entraîner le U-Net 3D (O5) et l'évaluer avec la même métrique (FROC) que O4.

---

## Données

**FUMPE** : jeu public de **35 angioscanners**, 8 792 coupes, 3 438 régions d'embolie annotées par deux
radiologues. **67 % des régions sont périphériques**, là où la détection est la plus difficile.

Les données ne sont pas versionnées (trop lourdes). Les placer ainsi :

```text
data/raw/
├── CT_scans 01/CT_scans 01/PAT001/*.dcm   (idem CT_scans 02, 03)
└── GroundTruth/GroundTruth/PAT001.mat …
```

---

## Arborescence

```text
Projet_detection_embolie_pulmonaire/
├── config.yaml              # paramètres globaux (fenêtre HU, détection O2…)
├── requirements.txt
│
├── src/                     # pipeline de recherche, un dossier par objectif
│   ├── O1_preprocessing/    #   loader DICOM/.mat, normalisation, run_preprocessing
│   ├── O2_classical_detection/  # segmentation des poumons, détecteur 'bright'
│   ├── O3_radiomics/        #   extraction des features, construction du dataset ML
│   ├── O4_classical_classifier/ # entraînement (CV groupée patient + FROC)
│   ├── O5_deep_learning/    #   U-Net 3D, CNN 2D/3D, générateur de données
│   ├── O6_evaluation/ · O7_optimization/ · O8_clinical_validation/
│   └── utils/
│
├── notebooks/               # 01 → 08 : démonstrations pas à pas de chaque objectif
├── scripts/
│   ├── diagnostics/         #   outils de contrôle (alignement GT, segmentation poumons…)
│   └── run_app.ps1          #   lancement de l'application en une commande
│
├── backend/                 # application : API FastAPI (voir backend/README.md)
├── frontend/                # application : interface web + viewer NiiVue (voir frontend/README.md)
│
├── data/                    # non versionné : raw/ (source) et processed/ (.npy)
├── results/                 # métriques, FROC, modèles O4
├── app_data/                # non versionné : base, analyses et modèles de l'application
├── landing_page/            # page de présentation du projet
└── docs/                    # plans de marche, livrables, mémoire du projet
```

---

## Installation

Python 3.12, Windows (PowerShell).

```powershell
python -m venv EP_env
EP_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Sous Windows, si la console plante sur les emojis des logs : `$env:PYTHONIOENCODING='utf-8'`.
- **GPU (O5)** : TensorFlow n'utilise pas le GPU sous Windows natif. L'entraînement du U-Net se fait
  dans WSL2 (distribution `EP-Ubuntu`, venv `~/venvs/ep`). O1 à O4 et l'application tournent sur CPU.

---

## Pipeline de recherche (O1 → O8)

Chaque objectif a son code dans `src/` et son notebook de démonstration dans `notebooks/`.

| Objectif | Code | Notebook | Commande principale |
|---|---|---|---|
| O1 | `src/O1_preprocessing/` | `01_exploration`, `02_preprocessing_demo` | `python -m src.O1_preprocessing.run_preprocessing --force` |
| O2 | `src/O2_classical_detection/` | `03_classical_detection_demo` | depuis le notebook 03 (mode `bright`) |
| O3 | `src/O3_radiomics/` | `04_radiomics_demo` | `python -m src.O3_radiomics.build_ml_dataset` |
| O4 | `src/O4_classical_classifier/` | `05_classical_classifier_demo` | `python -m src.O4_classical_classifier.train_classifier` |
| O5 | `src/O5_deep_learning/` | `06_deep_learning_demo` | à venir (WSL + GPU) |
| O6–O8 | `src/O6_…` à `src/O8_…` | `07`, `08` | à venir |

⚠️ Ne jamais lancer un prétraitement en même temps qu'un job O2 : les deux écrivent dans `data/processed/`.

**Outils de contrôle** (`scripts/diagnostics/`) : `check_gt_alignment.py`, `measure_gate.py`,
`compare_detection_ab.py`, `check_lung_segmentation.py`, `sweep_vessel_threshold.py`.

---

## Application locale (démo)

Une application web qui tourne **entièrement sur la machine** (PC du service ou laptop de démo) : aucune
image ne transite par Internet. Ce choix répond à deux contraintes des hôpitaux africains : une connexion
faible ou instable, et des données patient qui ne doivent pas sortir de l'établissement.

```text
Navigateur ──HTTP local──▶ FastAPI (127.0.0.1) ──▶ moteur IA ──▶ src/ (O1 → O4, puis O5)
  viewer NiiVue                 │  auth, anonymisation DICOM, file d'analyses
                                ▼
                          SQLite + fichiers temporaires (purgés après 24 h)
```

**Ce que fait l'interface :** connexion, analyse d'un cas FUMPE ou d'une archive `.zip` DICOM importée,
suivi de la progression, viewer 2D/3D avec les zones retenues par l'IA (rouge) et, pour les cas FUMPE,
la vérité terrain des radiologues (vert). Un tableau liste chaque zone avec son score, son côté, ses
coupes et son volume.

**Sécurité :** écoute locale uniquement, mots de passe Argon2, sessions HttpOnly/SameSite, anti-CSRF,
contrôle des archives (zip slip, zip bomb), anonymisation DICOM dès l'import, effacement automatique,
modèles vérifiés par empreinte SHA-256, journal d'audit, CSP stricte.

**Honnêteté de la démo :** un cas FUMPE est toujours analysé par le modèle du fold de validation croisée
où ce patient était en test, jamais par un modèle qui l'a vu à l'entraînement.

### Démarrage

```powershell
# Une seule fois
python -m src.O4_classical_classifier.train_classifier   # produit aussi les modèles par fold
python -m backend.cli install-models
python -m backend.cli create-user admin --role admin

# À chaque démo
.\scripts\run_app.ps1          # ouvre http://127.0.0.1:8000
```

Tests : `python -m pytest backend/tests -q`. Détails : [backend/README.md](backend/README.md),
[frontend/README.md](frontend/README.md).

---

## Résultats mesurés

**Baseline classique O2 → O4** (régression logistique, validation croisée groupée par patient, 33 porteurs) :

| Fausses alertes par patient | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Porteurs avec au moins une embolie retrouvée | 15 % | 30 % | **45 %** | 58 % |

- AUC-PR 0,10 (hasard : 0,017) : le signal est réel (environ 6 fois le hasard) mais encore insuffisant
  cliniquement (cible ≥ 70 %).
- Random Forest et XGBoost ne font pas significativement mieux que la régression logistique.
- Limites connues : les emboles **centraux** (dans les hiles, hors du masque des poumons) et les vérités
  terrain très sombres sont ratés dès O2.

---

## Documentation

| Document | Contenu |
|---|---|
| `docs/planning/plan1.md` | Correction du pipeline classique O2 → O4 : décisions, mesures |
| `docs/planning/plan2.md` | Application hors ligne : architecture, API, sécurité, phases |
| `docs/conversations/conversation_EP.md` | Mémoire globale du projet (pour l'équipe et les agents IA) |
| `docs/Livrables/Cahier_des_charges.md` | Cahier des charges |

---

## Travailler sur le dépôt

- Branche de travail : `devT1` ; branche principale : `main`.
- Petits commits, message au format `feat(O4): …`, `fix(app): …`.
- Ne jamais versionner `data/`, `app_data/` ni `EP_env/` (déjà dans `.gitignore`).

```bash
git add <fichiers>
git commit -m "feat(app): description"
git push origin devT1
```
