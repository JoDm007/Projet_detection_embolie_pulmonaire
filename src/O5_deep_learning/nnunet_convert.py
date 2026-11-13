# src/O5_deep_learning/nnunet_convert.py
"""
O5 — Étape 2 : conversion FUMPE → format nnU-Net v2 (Dataset501_FUMPE).

Pour chaque patient :
  - image  : HU bruts relus depuis le DICOM (loader O1), sauvés en int16 NIfTI
             avec le spacing réel (pas de fenêtrage : nnU-Net fait sa propre
             normalisation CT à partir des statistiques du dataset) ;
  - label  : data/processed/PAT###/mask.npy, c.-à-d. EXACTEMENT le GT corrigé
             (transpose (2,0,1) + align_mask_z) utilisé par O3/O4 ;
  - contrôle d'alignement : clip(HU, -1000, 400) normalisé doit redonner
             data/processed/PAT###/volume.npy (écart max < 1e-3). Si ce
             contrôle échoue, le patient est REFUSÉ (pas de NIfTI écrit).

Produit aussi :
  - nnUNet_raw/Dataset501_FUMPE/dataset.json  (canal "CT" → normalisation CT nnU-Net)
  - nnUNet_preprocessed/Dataset501_FUMPE/splits_final.json  (les 5 folds O4,
    pour comparer O5 et O4 sur exactement les mêmes patients de test)

Mémoire : un patient à la fois (~0,6 Go de pic), compatible PC 15 Go.

Usage (NN_env activé + . .\\scripts\\nnunet_env.ps1) :
    python -m src.O5_deep_learning.nnunet_convert              # tous les patients
    python -m src.O5_deep_learning.nnunet_convert PAT001       # un seul (test)
    python -m src.O5_deep_learning.nnunet_convert --force      # réécrit l'existant
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.O1_preprocessing.loader import get_available_patients, load_dicom_volume  # noqa: E402

logger = logging.getLogger("nnunet_convert")

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_FUMPE"
# Fenêtre RÉELLE des volume.npy (vérifié le 2026-11-13 sur PAT001/010/035 : écart 0,0),
# seulement pour le contrôle d'alignement. ⚠️ run_preprocessing.py et meta.json annoncent
# [-600, 400], mais les fichiers du 23/10 ont été produits avec [-1000, 400].
HU_MIN, HU_MAX = -1000.0, 400.0
ALIGN_TOL = 1e-3

PROCESSED_DIR = ROOT / "data" / "processed"
FOLDS_CSV = ROOT / "results" / "classical_classifier" / "models" / "folds" / "fold_assignments.csv"


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise EnvironmentError(
            f"Variable {name} absente. Lancer d'abord :  . .\\scripts\\nnunet_env.ps1"
        )
    return Path(val)


def convert_patient(pid: str, dicom_dir: Path, images_dir: Path, labels_dir: Path,
                    force: bool = False) -> Dict:
    """Convertit un patient ; renvoie un dict de rapport (status, shape, spacing, écart…)."""
    import SimpleITK as sitk

    img_out = images_dir / f"{pid}_0000.nii.gz"
    lbl_out = labels_dir / f"{pid}.nii.gz"
    if img_out.exists() and lbl_out.exists() and not force:
        return {"PatientID": pid, "status": "déjà converti"}

    mask_path = PROCESSED_DIR / pid / "mask.npy"
    ref_path = PROCESSED_DIR / pid / "volume.npy"
    if not mask_path.exists() or not ref_path.exists():
        return {"PatientID": pid, "status": "ERREUR : mask.npy/volume.npy absent"}

    hu, meta = load_dicom_volume(str(dicom_dir))
    if hu is None:
        return {"PatientID": pid, "status": "ERREUR : DICOM illisible"}

    mask = np.load(mask_path)                 # uint8 (Z, Y, X), GT corrigé O1
    ref = np.load(ref_path, mmap_mode="r")    # float32 [0,1], référence O3/O4

    # O1 tronque en Z au min(volume, masque) depuis le début : on reproduit.
    if hu.shape[0] != mask.shape[0]:
        logger.warning(f"{pid} : Z DICOM={hu.shape[0]} ≠ Z masque={mask.shape[0]} → troncature")
        hu = hu[: mask.shape[0]]
    if hu.shape != mask.shape or hu.shape != ref.shape:
        return {"PatientID": pid,
                "status": f"ERREUR : formes DICOM {hu.shape} / mask {mask.shape} / ref {ref.shape}"}

    # Contrôle d'alignement, coupe par coupe (évite un 2e volume float32 en RAM)
    max_diff = 0.0
    for z in range(hu.shape[0]):
        win = (np.clip(hu[z], HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        max_diff = max(max_diff, float(np.abs(win - ref[z]).max()))
    if max_diff > ALIGN_TOL:
        return {"PatientID": pid, "status": f"REFUSÉ : écart vs volume.npy = {max_diff:.4f}",
                "max_diff": max_diff}

    sy, sx = (float(v) for v in meta.get("spacing", [1.0, 1.0]))
    sz = float(meta.get("slice_spacing") or meta.get("slice_thickness") or 1.0)

    img = sitk.GetImageFromArray(np.round(hu).astype(np.int16))   # (Z,Y,X) → image (x,y,z)
    img.SetSpacing((sx, sy, sz))
    lbl = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    lbl.CopyInformation(img)                                      # géométrie identique

    sitk.WriteImage(img, str(img_out), useCompression=True)
    sitk.WriteImage(lbl, str(lbl_out), useCompression=True)

    report = {"PatientID": pid, "status": "OK", "shape": list(hu.shape),
              "spacing_xyz": [sx, sy, sz], "max_diff": max_diff,
              "gt_voxels": int((mask > 0).sum()),
              "hu_range": [float(hu.min()), float(hu.max())]}
    del hu, mask, ref, img, lbl
    gc.collect()
    return report


def write_dataset_json(raw_ds: Path, n_cases: int) -> None:
    ds = {
        "name": DATASET_NAME,
        "description": "FUMPE — segmentation de l'embolie pulmonaire (CTPA), GT corrigé O1",
        "channel_names": {"0": "CT"},          # "CT" → CTNormalization de nnU-Net
        "labels": {"background": 0, "embolie": 1},
        "numTraining": n_cases,
        "file_ending": ".nii.gz",
    }
    (raw_ds / "dataset.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")


def write_splits(pre_ds: Path, case_ids: List[str]) -> pd.DataFrame:
    """splits_final.json = folds O4 (val du fold k = patients avec fold == k)."""
    folds = pd.read_csv(FOLDS_CSV)
    folds = folds[folds["PatientID"].isin(case_ids)]
    missing = sorted(set(case_ids) - set(folds["PatientID"]))
    if missing:
        raise ValueError(f"Patients convertis absents de fold_assignments.csv : {missing}")
    splits = []
    for k in sorted(folds["fold"].unique()):
        val = sorted(folds.loc[folds["fold"] == k, "PatientID"])
        train = sorted(folds.loc[folds["fold"] != k, "PatientID"])
        splits.append({"train": train, "val": val})
    pre_ds.mkdir(parents=True, exist_ok=True)
    (pre_ds / "splits_final.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return folds.groupby("fold").size().rename("n_val").reset_index()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patients", nargs="*", help="PAT### à convertir (défaut : tous)")
    ap.add_argument("--force", action="store_true", help="réécrire les NIfTI existants")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("src.O1_preprocessing.loader").setLevel(logging.WARNING)

    raw_ds = _env_dir("nnUNet_raw") / DATASET_NAME
    pre_ds = _env_dir("nnUNet_preprocessed") / DATASET_NAME
    images_dir, labels_dir = raw_ds / "imagesTr", raw_ds / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    inv = get_available_patients(str(ROOT / "data"), force_raw=True)
    pids = args.patients or inv["with_both"]
    logger.info(f"{len(pids)} patient(s) à convertir → {raw_ds}")

    reports = []
    for i, pid in enumerate(pids, 1):
        if pid not in inv["vol_paths"]:
            reports.append({"PatientID": pid, "status": "ERREUR : DICOM introuvable"})
            continue
        r = convert_patient(pid, Path(inv["vol_paths"][pid]), images_dir, labels_dir, args.force)
        logger.info(f"[{i}/{len(pids)}] {pid} : {r['status']}"
                    + (f"  shape={r['shape']} spacing={r['spacing_xyz']} écart={r['max_diff']:.1e}"
                       if r["status"] == "OK" else ""))
        reports.append(r)

    rep = pd.DataFrame(reports)
    rep.to_csv(raw_ds / "conversion_report.csv", index=False)

    # dataset.json + splits seulement sur l'ensemble des cas présents sur disque
    done = sorted(p.name.replace("_0000.nii.gz", "") for p in images_dir.glob("*_0000.nii.gz"))
    write_dataset_json(raw_ds, len(done))
    n_bad = int((~rep["status"].isin(["OK", "déjà converti"])).sum())

    print("\n" + "=" * 60)
    print(f"Cas NIfTI sur disque : {len(done)}   |   problèmes ce passage : {n_bad}")
    if n_bad:
        print(rep[~rep["status"].isin(["OK", "déjà converti"])].to_string(index=False))
    if len(done) == len(inv["with_both"]):
        sizes = write_splits(pre_ds, done)
        print(f"splits_final.json écrit (folds O4) :\n{sizes.to_string(index=False)}")
    else:
        print(f"splits_final.json NON écrit : {len(done)}/{len(inv['with_both'])} cas convertis.")
    print(f"Rapport : {raw_ds / 'conversion_report.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
