# backend/api/routes/cases.py
from fastapi import APIRouter, Depends

from backend.api.deps import current_user
from backend.core.config import get_settings
from backend.schemas.analyses import DemoCaseOut
from backend.services.ingestion import list_demo_cases

router = APIRouter(prefix="/api/cases", tags=["cas de démo"])


@router.get("/demo", response_model=list[DemoCaseOut], dependencies=[Depends(current_user)])
def demo_cases():
    """Cas FUMPE (jeu public) disponibles localement dans data/raw."""
    return [DemoCaseOut(case_id=c) for c in list_demo_cases(get_settings().fumpe_dir)]
