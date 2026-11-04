# backend/services/ingestion.py
# Entrée des données dans l'application.
#   - Upload : archive .zip d'une série DICOM → extraction sûre → anonymisation → volume HU.
#   - Démo   : cas FUMPE lu depuis data/raw via le loader O1 (+ vérité terrain pour l'affichage).
# Les fichiers DICOM ne survivent pas à l'ingestion : seul le volume de pixels est conservé.
import logging
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from src.O1_preprocessing.loader import get_available_patients, load_dicom_volume, load_patient_data

from .anonymisation import DicomRejected, anonymise_directory

logger = logging.getLogger("backend.services.ingestion")

_CHUNK = 1 << 20


class UploadRejected(Exception):
    """Archive refusée ; message destiné à l'utilisateur."""


class UploadTooLarge(UploadRejected):
    """Archive au-delà de la taille maximale (HTTP 413)."""


def save_upload_stream(read_chunk, dest: Path, max_bytes: int) -> int:
    """Écrit un upload sur disque par blocs, en coupant dès que la taille maximale est dépassée.
    read_chunk(n) -> bytes. Vérifie aussi la signature d'une archive zip."""
    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = read_chunk(_CHUNK)
            if not chunk:
                break
            if total == 0 and not chunk.startswith(b"PK\x03\x04"):
                raise UploadRejected("Le fichier n'est pas une archive .zip.")
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge(f"Archive trop volumineuse (maximum {max_bytes // 2**20} Mo).")
            out.write(chunk)
    if total == 0:
        raise UploadRejected("Fichier vide.")
    return total


def _is_unsafe_name(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return p.is_absolute() or ".." in p.parts or (len(name) > 1 and name[1] == ":")


def safe_extract_dicom_zip(zip_path: Path, dest: Path, max_files: int, max_total_bytes: int,
                           max_ratio: int) -> int:
    """Extrait uniquement les .dcm, sous des noms générés (00001.dcm…) : le nom contenu dans
    l'archive n'est jamais utilisé comme chemin, ce qui neutralise le « zip slip ».
    Protège aussi contre les « zip bombs » (nombre de fichiers, taille décompressée, ratio)."""
    if not zipfile.is_zipfile(zip_path):
        raise UploadRejected("Archive .zip invalide ou corrompue.")
    dest.mkdir(parents=True, exist_ok=True)
    written_total, n = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise UploadRejected(f"Trop de fichiers dans l'archive (maximum {max_files}).")
        for info in infos:
            if info.is_dir():
                continue
            if _is_unsafe_name(info.filename):
                raise UploadRejected("Archive refusée : chemin de fichier suspect.")
            if not info.filename.lower().endswith(".dcm"):
                continue
            if info.flag_bits & 0x1:
                raise UploadRejected("Les archives chiffrées ne sont pas acceptées.")
            if info.compress_size > 0 and info.file_size / info.compress_size > max_ratio:
                raise UploadRejected("Archive refusée : taux de compression anormal.")
            n += 1
            target = dest / f"{n:05d}.dcm"
            with zf.open(info) as src, open(target, "wb") as out:
                # On recompte les octets réellement décompressés : les en-têtes zip peuvent mentir
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    written_total += len(chunk)
                    if written_total > max_total_bytes:
                        raise UploadRejected("Archive refusée : contenu décompressé trop volumineux.")
                    out.write(chunk)
    if n == 0:
        raise UploadRejected("Aucun fichier .dcm dans l'archive.")
    return n


def _spacing_from_meta(meta: dict) -> tuple[float, float, float]:
    ps = meta.get("spacing", [1.0, 1.0])
    slice_spacing = float(meta.get("slice_spacing", 1.0)) or float(meta.get("slice_thickness", 1.0))
    return (abs(slice_spacing), float(ps[0]), float(ps[1]))


def ingest_upload(zip_path: Path, work_dir: Path, settings) -> tuple[np.ndarray, tuple, dict]:
    """Archive importée → (volume HU (Z,Y,X) float32, spacing (z,y,x) mm, infos non identifiantes)."""
    dicom_dir = work_dir / "dicom"
    try:
        safe_extract_dicom_zip(zip_path, dicom_dir, settings.max_zip_files,
                               settings.max_unzipped_mb * 2**20, settings.max_compression_ratio)
        info = anonymise_directory(dicom_dir)
        volume, meta = load_dicom_volume(str(dicom_dir))
        if volume is None:
            raise DicomRejected("Impossible de reconstruire le volume (coupes de tailles différentes ?).")
        return volume.astype(np.float32), _spacing_from_meta(meta), info
    finally:
        # Ni l'archive ni les DICOM (même anonymisés) ne sont conservés
        shutil.rmtree(dicom_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)


# ─────────────────────────────── Cas de démo FUMPE ───────────────────────────────
def list_demo_cases(fumpe_dir: Path) -> list[str]:
    if not (fumpe_dir / "raw").exists():
        return []
    inventory = get_available_patients(str(fumpe_dir), force_raw=True)
    return list(inventory["with_both"])


def ingest_demo_case(case_id: str, fumpe_dir: Path) -> tuple[np.ndarray, tuple, np.ndarray]:
    """Cas FUMPE → (volume HU, spacing, vérité terrain). Le GT sert uniquement à l'affichage."""
    inventory = get_available_patients(str(fumpe_dir), force_raw=True)
    if case_id not in inventory["with_both"]:
        raise DicomRejected(f"Cas de démo {case_id} introuvable dans {fumpe_dir / 'raw'}.")
    volume, mask, meta = load_patient_data(case_id, inventory)
    if volume is None:
        raise DicomRejected(f"Lecture du cas {case_id} impossible.")
    gt = (mask > 0).astype(np.uint8) if mask is not None and mask.shape == volume.shape else None
    return volume.astype(np.float32, copy=False), _spacing_from_meta(meta), gt
