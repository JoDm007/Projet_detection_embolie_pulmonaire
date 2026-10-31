# backend/services/retention.py
# Rétention minimale (plan2 § 4.6) : les analyses terminées sont effacées (disque + base)
# après settings.retention_hours. Exécuté au démarrage puis toutes les heures.
import logging
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone

from backend.db.repositories import AnalysisRepo, AuditRepo, SessionRepo

logger = logging.getLogger("backend.services.retention")


def purge_expired(settings) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=settings.retention_hours)).isoformat(timespec="seconds")
    purged = 0
    for row in AnalysisRepo.older_than(cutoff):
        shutil.rmtree(settings.jobs_dir / row["id"], ignore_errors=True)
        AnalysisRepo.delete(row["id"])
        purged += 1

    # Dossiers orphelins (ex. arrêt brutal entre la création du dossier et l'insertion en base)
    if settings.jobs_dir.exists():
        for d in settings.jobs_dir.iterdir():
            try:
                uuid.UUID(d.name)
            except ValueError:
                continue
            if AnalysisRepo.get(d.name) is None:
                shutil.rmtree(d, ignore_errors=True)

    SessionRepo.purge_expired()
    if purged:
        AuditRepo.log("purge_automatique", None, None, f"{purged} analyse(s) effacée(s)")
        logger.info(f"Purge : {purged} analyse(s) effacée(s)")
    return purged


class PurgeLoop:
    """Relance purge_expired() à intervalle régulier dans un thread démon."""

    def __init__(self, settings, interval_s: int = 3600):
        self.settings = settings
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="purge", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                purge_expired(self.settings)
            except Exception:
                logger.exception("Échec de la purge automatique")
