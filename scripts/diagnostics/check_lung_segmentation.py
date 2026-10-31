"""
scripts/diagnostics/check_lung_segmentation.py

Contrôle la qualité de la segmentation des poumons : pour chaque patient, le masque
pulmonaire devrait couvrir ~15-30 % du volume thoracique. Un %volume trop haut (>35 %)
signale une sur-segmentation (médiastin/cœur inclus) ; trop bas (<5 %) une sous-segmentation.

Charge les données RAW (vraies HU) et segmente -> LENT (~20-50 s/patient).

Usage (depuis la racine du projet) :
    python scripts/diagnostics/check_lung_segmentation.py --patients PAT018 PAT005 PAT002
    python scripts/diagnostics/check_lung_segmentation.py            # 6 patients par défaut
"""
import sys
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.O1_preprocessing.loader import get_available_patients, load_patient_data
from src.O2_classical_detection.lung_segmentation import segment_lungs

DEFAULT = ["PAT018", "PAT005", "PAT002", "PAT001", "PAT017", "PAT023"]
HAUT = 35.0   # au-delà : sur-segmentation probable
BAS = 5.0     # en-dessous : sous-segmentation probable


def main():
    ap = argparse.ArgumentParser(description="Qualité de la segmentation pulmonaire (%volume).")
    ap.add_argument("--patients", nargs="+", default=DEFAULT, help="Liste de PATxxx")
    args = ap.parse_args()

    inv = get_available_patients(str(PROJECT_ROOT / "data"), force_raw=True)
    print(f"{'patient':8} {'%volume':>8} {'régions':>8}  verdict")
    for pid in args.patients:
        if pid not in inv["with_both"]:
            print(f"{pid:8}  (absent de l'inventaire RAW)")
            continue
        vol, _, _ = load_patient_data(pid, inv)
        lung = segment_lungs(vol, hu_threshold=-400, min_area=1000)
        pct = 100 * lung.sum() / lung.size
        from scipy.ndimage import label
        n = label(lung)[1]
        verdict = "OK"
        if pct > HAUT:
            verdict = "SUR-SEGMENTÉ (médiastin/cœur ?)"
        elif pct < BAS:
            verdict = "SOUS-SEGMENTÉ"
        print(f"{pid:8} {pct:7.1f}% {n:8d}  {verdict}")


if __name__ == "__main__":
    main()
