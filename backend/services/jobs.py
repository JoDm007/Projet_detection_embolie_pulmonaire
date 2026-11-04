# backend/services/jobs.py
# File d'analyses : un seul worker (une analyse occupe le CPU 1–2 min et beaucoup de RAM ;
# en lancer deux à la fois ralentirait les deux). L'état vit en base, donc l'interface peut
# suivre la progression et un redémarrage ne laisse pas d'analyse « en cours » fantôme.
import gc
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import psutil

from backend.db.repositories import AnalysisRepo, AuditRepo
from backend.engines.base import EngineUnavailable

from . import export, ingestion
from .anonymisation import DicomRejected

logger = logging.getLogger("backend.services.jobs")

# Erreurs dont le message est fait pour l'utilisateur ; les autres sont masquées (plan2 § 4.10)
USER_FACING_ERRORS = (ingestion.UploadRejected, DicomRejected, EngineUnavailable)


def _gb(n_bytes: float) -> float:
    return round(n_bytes / 2**30, 1)


class JobManager:
    def __init__(self, settings, engines: dict):
        self.settings = settings
        self.engines = engines
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyse")

    def job_dir(self, analysis_id: str) -> Path:
        # analysis_id est toujours un UUID généré ici : pas de chemin contrôlé par l'utilisateur
        return self.settings.jobs_dir / str(uuid.UUID(analysis_id))

    def slot(self, name: str):
        """EngineSlot du moteur (instance ou raison d'indisponibilité), None s'il est inconnu."""
        return self.engines.get(name)

    def engine_or_none(self, name: str):
        slot = self.slot(name)
        return slot.engine if slot else None

    # ── soumission ──
    def submit(self, user, engine_name: str, source: str, case_id: Optional[str] = None,
               upload_path: Optional[Path] = None) -> str:
        analysis_id = str(uuid.uuid4())
        job_dir = self.job_dir(analysis_id)
        job_dir.mkdir(parents=True)
        if upload_path is not None:
            upload_path = upload_path.replace(job_dir / "upload.zip")
        AnalysisRepo.create(analysis_id, user["id"], source, engine_name, case_id)
        AuditRepo.log("analyse_lancee", user["username"], analysis_id,
                      f"source={source} moteur={engine_name}" + (f" cas={case_id}" if case_id else ""))
        self._pool.submit(self._run, analysis_id, engine_name, source, case_id, upload_path)
        return analysis_id

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── exécution (thread du worker) ──
    def _run(self, analysis_id, engine_name, source, case_id, upload_path):
        job_dir = self.job_dir(analysis_id)

        def progress(step: str, value: float):
            AnalysisRepo.set_progress(analysis_id, step, round(min(max(value, 0.0), 1.0), 3))

        try:
            engine = self.engine_or_none(engine_name)
            if engine is None:
                raise EngineUnavailable(f"Moteur « {engine_name} » indisponible.")
            self._check_free_memory(analysis_id)

            groundtruth = None
            if source == "upload":
                progress("O1 — lecture et anonymisation des DICOM", 0.02)
                volume, spacing, _ = ingestion.ingest_upload(upload_path, job_dir, self.settings)
            else:
                progress(f"O1 — chargement du cas de démo {case_id}", 0.02)
                volume, spacing, groundtruth = ingestion.ingest_demo_case(case_id, self.settings.fumpe_dir)

            result = engine.analyse(volume, spacing, progress,
                                    case_id=case_id if source == "demo" else None)

            progress("Préparation de l'affichage", 0.93)
            summary = export.export_result(job_dir, volume, spacing, result, groundtruth)
            AnalysisRepo.set_done(analysis_id, summary)
            logger.info(f"Analyse {analysis_id} terminée : {summary['n_findings']} zones retenues "
                        f"(pic mémoire du serveur {self._peak_rss_gb()} Go)")
        except USER_FACING_ERRORS as exc:
            AnalysisRepo.set_failed(analysis_id, str(exc))
            logger.warning(f"Analyse {analysis_id} refusée : {exc}")
        except MemoryError:
            AnalysisRepo.set_failed(analysis_id, "Mémoire insuffisante pour ce volume.")
            logger.exception(f"Analyse {analysis_id} : mémoire insuffisante")
        except Exception:
            error_id = uuid.uuid4().hex[:8]
            AnalysisRepo.set_failed(analysis_id, f"Erreur interne pendant l'analyse (réf. {error_id}).")
            logger.exception(f"Analyse {analysis_id} : erreur interne {error_id}")
        finally:
            if upload_path is not None:
                Path(upload_path).unlink(missing_ok=True)
            # Rend tout de suite les gros tableaux (volume, labels, overlay) au système
            gc.collect()

    def _check_free_memory(self, analysis_id: str) -> None:
        available = psutil.virtual_memory().available
        logger.info(f"Analyse {analysis_id} : {_gb(available)} Go de RAM libre au démarrage")
        if available < self.settings.min_free_ram_gb * 2**30:
            raise EngineUnavailable(
                f"Mémoire insuffisante : {_gb(available)} Go de RAM libre, "
                f"{self.settings.min_free_ram_gb:g} Go nécessaires. Fermez d'autres applications "
                f"(ou arrêtez WSL : « wsl --shutdown ») puis relancez l'analyse.")

    @staticmethod
    def _peak_rss_gb():
        info = psutil.Process().memory_info()
        # peak_wset n'existe que sous Windows
        return _gb(getattr(info, "peak_wset", info.rss))

    # ── suppression ──
    def delete(self, analysis_id: str) -> None:
        shutil.rmtree(self.job_dir(analysis_id), ignore_errors=True)
        AnalysisRepo.delete(analysis_id)
