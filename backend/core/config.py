# backend/core/config.py
# Réglages de l'application. Valeurs par défaut sûres (écoute locale, rétention courte),
# surchargeables par variables d'environnement préfixées PESEG_ (ex. PESEG_RETENTION_HOURS=4).
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PESEG_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "oui"}


@dataclass(frozen=True)
class Settings:
    # Réseau : 127.0.0.1 = accessible uniquement depuis cette machine (cf. plan2 § 4.1)
    host: str = "127.0.0.1"
    port: int = 8000
    # True uniquement derrière HTTPS (mode LAN) : le cookie n'est alors envoyé qu'en TLS
    secure_cookies: bool = False
    # Swagger /docs charge ses fichiers depuis un CDN → désactivé par défaut (hors ligne + surface réduite)
    enable_docs: bool = False

    # Chemins
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "app_data"          # base SQLite, jobs, modèles installés
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    fumpe_dir: Path = PROJECT_ROOT / "data"             # cas de démo FUMPE (data/raw + data/processed)
    models_source_dir: Path = PROJECT_ROOT / "results" / "classical_classifier"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Upload (cf. plan2 § 4.4)
    max_upload_mb: int = 1024
    max_unzipped_mb: int = 3072
    max_zip_files: int = 5000
    max_compression_ratio: int = 200

    # Sessions et connexion (cf. plan2 § 4.2)
    session_hours: int = 8
    login_max_failures: int = 5
    lockout_minutes: int = 15

    # Rétention des analyses (cf. plan2 § 4.6)
    retention_hours: int = 24

    # RAM libre exigée pour démarrer une analyse (pic mesuré ~2–3 Go sur un CT 512×512×~250).
    # En dessous, l'analyse est refusée plutôt que de saturer la machine (swap, gel du PC).
    min_free_ram_gb: float = 4.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    """Lit l'environnement une seule fois. Les tests appellent get_settings.cache_clear()."""
    defaults = Settings()
    return Settings(
        host=_env("HOST", defaults.host),
        port=int(_env("PORT", str(defaults.port))),
        secure_cookies=_env_bool("SECURE_COOKIES", defaults.secure_cookies),
        enable_docs=_env_bool("ENABLE_DOCS", defaults.enable_docs),
        data_dir=Path(_env("DATA_DIR", str(defaults.data_dir))),
        frontend_dir=Path(_env("FRONTEND_DIR", str(defaults.frontend_dir))),
        fumpe_dir=Path(_env("FUMPE_DIR", str(defaults.fumpe_dir))),
        models_source_dir=Path(_env("MODELS_SOURCE_DIR", str(defaults.models_source_dir))),
        logs_dir=Path(_env("LOGS_DIR", str(defaults.logs_dir))),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", str(defaults.max_upload_mb))),
        max_unzipped_mb=int(_env("MAX_UNZIPPED_MB", str(defaults.max_unzipped_mb))),
        max_zip_files=int(_env("MAX_ZIP_FILES", str(defaults.max_zip_files))),
        session_hours=int(_env("SESSION_HOURS", str(defaults.session_hours))),
        login_max_failures=int(_env("LOGIN_MAX_FAILURES", str(defaults.login_max_failures))),
        lockout_minutes=int(_env("LOCKOUT_MINUTES", str(defaults.lockout_minutes))),
        retention_hours=int(_env("RETENTION_HOURS", str(defaults.retention_hours))),
        min_free_ram_gb=float(_env("MIN_FREE_RAM_GB", str(defaults.min_free_ram_gb))),
    )
