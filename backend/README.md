# Backend PE-Seg AI (FastAPI)

API locale de l'application hors ligne. Plan et décisions : `docs/planning/plan2.md`.

## Blocs

| Bloc | Rôle | Règle |
|---|---|---|
| `core/` | config, sécurité (Argon2, sessions), en-têtes HTTP, erreurs, logs | utilisable par tous les blocs |
| `db/` | SQLite `app_data/app.db` : users, sessions, analyses, audit | **seul** bloc qui touche la base |
| `schemas/` | modèles Pydantic des échanges API | aucun accès disque |
| `api/` | routes HTTP minces : valident, vérifient les droits, délèguent | n'importe ni `src/` ni SQLite |
| `services/` | ingestion zip/DICOM, anonymisation, jobs, export NIfTI, purge | aucune notion HTTP |
| `engines/` | moteurs d'IA (classique O2→O4 ; U-Net O5 à venir) | **seul** bloc qui importe `src/` |
| `tests/` | pytest, un fichier par bloc | — |

Flux : `api → services → engines → src/`.

## Première installation (une fois)

Depuis la racine du dépôt, venv `EP_env` activé :

```powershell
pip install "python-multipart>=0.0.20" "argon2-cffi>=23.1" "httpx>=0.27" "pytest>=8"

# 1. Modèles par fold (nécessaire pour la démo honnête sur FUMPE) — résultats identiques (seed fixe)
python -m src.O4_classical_classifier.train_classifier

# 2. Copier le modèle retenu dans app_data/models + empreintes SHA-256
python -m backend.cli install-models

# 3. Créer un compte
python -m backend.cli create-user admin --role admin
python -m backend.cli create-user dr_demo
```

## Lancer

```powershell
.\scripts\run_app.ps1
# ou : python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Tests

```powershell
python -m pytest backend/tests -q
```

## Administration

`python -m backend.cli --help` : comptes (`create-user`, `set-password`, `disable-user`…),
modèles (`install-models`, `check-models`), `purge`, `audit`.

## Réglages (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `PESEG_HOST` / `PESEG_PORT` | `127.0.0.1` / `8000` | écoute (garder 127.0.0.1 hors mode LAN+HTTPS) |
| `PESEG_RETENTION_HOURS` | `24` | effacement automatique des analyses |
| `PESEG_MAX_UPLOAD_MB` | `1024` | taille max d'une archive |
| `PESEG_SESSION_HOURS` | `8` | durée d'une session |
| `PESEG_SECURE_COOKIES` | `0` | `1` derrière HTTPS |
| `PESEG_ENABLE_DOCS` | `0` | `/docs` (Swagger charge un CDN : en ligne seulement) |
| `PESEG_DATA_DIR` | `app_data/` | base, jobs, modèles installés |

## Points d'attention

- **Spacing des features** : le dataset O4 actuel a été construit sans `meta.json`, donc avec un
  spacing de 1×1×1 mm. Le moteur reproduit ce choix (`feature_spacing: "unit"`). Si le dataset est
  régénéré avec le vrai spacing : `install-models --feature-spacing dicom`.
- **Démo FUMPE** : chaque cas est analysé par le modèle du fold où il était en test. Sans les
  modèles par fold, les cas de démo sont refusés (message explicite).
- **Orientation** : le côté (droit/gauche) suppose une acquisition axiale standard.
