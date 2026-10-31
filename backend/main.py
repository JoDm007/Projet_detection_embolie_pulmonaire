# backend/main.py
# Assemble les blocs : configuration → base → moteurs → file de jobs → routes → frontend.
# Lancement (depuis la racine du dépôt, pour que « src » soit importable) :
#   uvicorn backend.main:app --host 127.0.0.1 --port 8000
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api.routes import analyses, auth, cases, health
from backend.core.config import get_settings
from backend.core.errors import register_error_handlers
from backend.core.headers import SecurityHeadersMiddleware
from backend.core.logging import setup_logging
from backend.db.database import init_db
from backend.db.repositories import AnalysisRepo
from backend.engines.registry import load_engines
from backend.services.jobs import JobManager
from backend.services.retention import PurgeLoop, purge_expired

logger = logging.getLogger("backend.main")


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.logs_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        init_db(settings.db_path)
        interrupted = AnalysisRepo.fail_interrupted()
        if interrupted:
            logger.warning(f"{interrupted} analyse(s) interrompue(s) par l'arrêt précédent")
        purge_expired(settings)
        app.state.jobs = JobManager(settings, load_engines(settings.models_dir, settings.project_root))
        purge_loop = PurgeLoop(settings)
        purge_loop.start()
        logger.info(f"PE-Seg AI {__version__} prêt sur http://{settings.host}:{settings.port}")
        yield
        purge_loop.stop()
        app.state.jobs.shutdown()

    app = FastAPI(
        title="PE-Seg AI — API locale",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.enable_docs else None,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    register_error_handlers(app)

    for module in (health, auth, cases, analyses):
        app.include_router(module.router)

    # Frontend statique servi à la même origine (pas de CORS à ouvrir). Monté en dernier :
    # les routes /api/* gardent la priorité.
    if settings.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
