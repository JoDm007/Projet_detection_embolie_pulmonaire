# backend/api/routes/health.py
from fastapi import APIRouter, Request

from backend import __version__
from backend.schemas.analyses import EngineOut, HealthOut

router = APIRouter(prefix="/api", tags=["santé"])


@router.get("/health", response_model=HealthOut)
def health(request: Request):
    """État du serveur et des moteurs (public : ne révèle aucune donnée)."""
    engines = [EngineOut(name=s.name, label=s.label, available=s.engine is not None, reason=s.reason)
               for s in request.app.state.jobs.engines.values()]
    return HealthOut(status="ok", version=__version__, engines=engines)
