# backend/core/logging.py
# Logs applicatifs : console + fichier logs/app.log. Aucune donnée patient n'y est écrite
# (les identifiants d'analyse sont des UUID, les cas de démo des identifiants FUMPE publics).
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(logs_dir: Path) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=5_000_000,
                                       backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)

    app_logger = logging.getLogger("backend")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(file_handler)
    _CONFIGURED = True
