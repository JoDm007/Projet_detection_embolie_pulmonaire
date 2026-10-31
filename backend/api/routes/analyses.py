# backend/api/routes/analyses.py
import json
import re
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.api.deps import csrf_protect, current_user, job_manager
from backend.core.config import get_settings
from backend.db.repositories import AnalysisRepo, AuditRepo
from backend.schemas.analyses import AnalysisOut, DemoRequest, SubmitOut
from backend.services import export
from backend.services.ingestion import UploadRejected, list_demo_cases, save_upload_stream

router = APIRouter(prefix="/api/analyses", tags=["analyses"])

CASE_ID_RE = re.compile(r"^PAT\d{3}$")
# Fichiers servis au viewer : liste fermée, jamais un nom venant de la requête
SERVED_FILES = {
    "volume.nii.gz": export.VOLUME_FILE,
    "overlay.nii.gz": export.OVERLAY_FILE,
    "groundtruth.nii.gz": export.GROUNDTRUTH_FILE,
}


def _owned_analysis(analysis_id: str, user):
    """L'analyse, si elle existe ET appartient à l'utilisateur (ou s'il est admin).
    Sinon 404 dans les deux cas : on ne révèle pas l'existence des analyses des autres."""
    try:
        uuid.UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    row = AnalysisRepo.get(analysis_id)
    if row is None or (row["user_id"] != user["id"] and user["role"] != "admin"):
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return row


def _check_engine(jobs, engine: str):
    slot = jobs.engines.get(engine)
    if slot is None:
        raise HTTPException(status_code=400, detail=f"Moteur inconnu : {engine}.")
    if slot.engine is None:
        raise HTTPException(status_code=409, detail=f"Moteur indisponible : {slot.reason}")


@router.post("", status_code=202, response_model=SubmitOut, dependencies=[Depends(csrf_protect)])
def submit_upload(file: UploadFile = File(...), engine: str = Form("classique"),
                  user=Depends(current_user), jobs=Depends(job_manager)):
    """Importe une archive .zip contenant UNE série DICOM d'angioscanner et lance l'analyse."""
    _check_engine(jobs, engine)
    settings = get_settings()
    incoming = settings.data_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / f"{uuid.uuid4()}.zip"
    try:
        save_upload_stream(file.file.read, tmp, settings.max_upload_mb * 2**20)
    except UploadRejected as exc:
        tmp.unlink(missing_ok=True)
        status = 413 if "volumineuse" in str(exc) else 400
        AuditRepo.log("upload_refuse", user["username"], None, str(exc))
        raise HTTPException(status_code=status, detail=str(exc))
    analysis_id = jobs.submit(user, engine, "upload", upload_path=tmp)
    return SubmitOut(analysis_id=analysis_id)


@router.post("/demo/{case_id}", status_code=202, response_model=SubmitOut,
             dependencies=[Depends(csrf_protect)])
def submit_demo(case_id: str, body: DemoRequest = DemoRequest(),
                user=Depends(current_user), jobs=Depends(job_manager)):
    """Lance l'analyse d'un cas FUMPE (jeu public) disponible localement."""
    if not CASE_ID_RE.match(case_id) or case_id not in list_demo_cases(get_settings().fumpe_dir):
        raise HTTPException(status_code=404, detail="Cas de démo introuvable.")
    _check_engine(jobs, body.engine)
    analysis_id = jobs.submit(user, body.engine, "demo", case_id=case_id)
    return SubmitOut(analysis_id=analysis_id)


@router.get("", response_model=list[AnalysisOut])
def list_analyses(user=Depends(current_user)):
    rows = AnalysisRepo.list_for_user(user["id"], include_all=user["role"] == "admin")
    return [AnalysisOut.from_row(r) for r in rows]


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: str, user=Depends(current_user)):
    return AnalysisOut.from_row(_owned_analysis(analysis_id, user))


@router.get("/{analysis_id}/findings")
def get_findings(analysis_id: str, user=Depends(current_user), jobs=Depends(job_manager)):
    row = _owned_analysis(analysis_id, user)
    path = jobs.job_dir(analysis_id) / export.FINDINGS_FILE
    if row["status"] != "done" or not path.exists():
        raise HTTPException(status_code=409, detail="Résultats pas encore disponibles.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.api_route("/{analysis_id}/{filename}", methods=["GET", "HEAD"])
def get_file(analysis_id: str, filename: str, user=Depends(current_user), jobs=Depends(job_manager)):
    """Fichiers NIfTI lus par le viewer (même origine, cookie de session requis)."""
    row = _owned_analysis(analysis_id, user)
    if filename not in SERVED_FILES or row["status"] != "done":
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    path = jobs.job_dir(analysis_id) / SERVED_FILES[filename]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(path, media_type="application/gzip", filename=filename)


@router.delete("/{analysis_id}", status_code=204, dependencies=[Depends(csrf_protect)])
def delete_analysis(analysis_id: str, user=Depends(current_user), jobs=Depends(job_manager)):
    row = _owned_analysis(analysis_id, user)
    if row["status"] in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Analyse en cours : attendre la fin pour la supprimer.")
    jobs.delete(analysis_id)
    AuditRepo.log("analyse_supprimee", user["username"], analysis_id)
