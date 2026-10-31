# backend/services/export.py
# Sorties d'une analyse, lues par le viewer NiiVue :
#   volume.nii.gz       CT en HU (int16)
#   overlay.nii.gz      zones retenues (label k = lésion k du tableau)
#   groundtruth.nii.gz  vérité terrain FUMPE (cas de démo uniquement, pour l'affichage)
#   findings.json       tableau des lésions + point de fonctionnement
#
# Orientation : les tableaux sont en (Z, Y, X) avec X = colonnes DICOM (vers la gauche du
# patient), Y = lignes (vers l'arrière), Z croissant vers la tête (tri du loader O1). C'est le
# repère LPS ; NIfTI attend du RAS, d'où les signes négatifs sur X et Y dans l'affine.
import json
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from backend.engines.base import AnalysisResult

VOLUME_FILE = "volume.nii.gz"
OVERLAY_FILE = "overlay.nii.gz"
GROUNDTRUTH_FILE = "groundtruth.nii.gz"
FINDINGS_FILE = "findings.json"


def _affine(spacing_zyx) -> np.ndarray:
    sz, sy, sx = (float(s) for s in spacing_zyx)
    return np.diag([-sx, -sy, sz, 1.0])


def save_nifti(array_zyx: np.ndarray, spacing_zyx, path: Path, dtype) -> None:
    data = np.ascontiguousarray(np.transpose(array_zyx, (2, 1, 0))).astype(dtype)  # → (X, Y, Z)
    img = nib.Nifti1Image(data, _affine(spacing_zyx))
    img.header.set_xyzt_units("mm")
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    nib.save(img, str(path))


def export_result(job_dir: Path, volume_hu: np.ndarray, spacing, result: AnalysisResult,
                  groundtruth: np.ndarray | None = None) -> dict:
    """Écrit les fichiers du viewer et renvoie le résumé stocké en base."""
    save_nifti(np.clip(volume_hu, -1024, 3071), spacing, job_dir / VOLUME_FILE, np.int16)
    overlay_dtype = np.uint8 if result.overlay.max(initial=0) <= 255 else np.uint16
    save_nifti(result.overlay, spacing, job_dir / OVERLAY_FILE, overlay_dtype)
    if groundtruth is not None:
        save_nifti(groundtruth, spacing, job_dir / GROUNDTRUTH_FILE, np.uint8)

    payload = {
        "engine": result.engine,
        "model_used": result.model_used,
        "operating_point": result.operating_point,
        "n_candidates": result.n_candidates,
        "warnings": result.warnings,
        "shape_zyx": list(volume_hu.shape),
        "spacing_zyx": [round(float(s), 4) for s in spacing],
        "has_groundtruth": groundtruth is not None,
        "findings": [asdict(f) for f in result.findings],
    }
    (job_dir / FINDINGS_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    summary = {k: payload[k] for k in ("engine", "model_used", "n_candidates", "has_groundtruth")}
    summary["n_findings"] = len(result.findings)
    summary["total_volume_mm3"] = round(sum(f.volume_mm3 for f in result.findings), 1)
    return summary
