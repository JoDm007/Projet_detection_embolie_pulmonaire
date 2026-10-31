"""
scripts/diagnostics/sweep_vessel_threshold.py

Balaye le seuil `vessel_min_hu` du mode 'bright' pour choisir sa valeur, SANS régénérer
tous les candidats à chaque essai. Les poumons sont segmentés UNE fois par patient, puis
on teste plusieurs seuils en mesurant : patients détectés, couverture GT, VP, FP.

Sous-ensemble par défaut = mélange de porteurs ratés à seuil 100 (GT peu contrasté) et
de porteurs déjà détectés, pour voir le compromis détection / faux positifs.

Charge les données RAW (vraies HU) -> LENT (~30 s/patient pour la segmentation).

Usage (depuis la racine du projet) :
    python scripts/diagnostics/sweep_vessel_threshold.py
    python scripts/diagnostics/sweep_vessel_threshold.py --thresholds 30 50 70 100 --patients PAT002 PAT021
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from skimage.morphology import remove_small_objects

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.O1_preprocessing.loader import get_available_patients, load_patient_data
from src.O2_classical_detection.lung_segmentation import segment_lungs

ST = generate_binary_structure(3, 1)
MAX_HU = 400
MIN_VOL = 50
# Ratés à seuil 100 (GT peu contrasté) + quelques détectés, petits et gros GT :
DEFAULT_PATIENTS = ["PAT002", "PAT010", "PAT021", "PAT022", "PAT028", "PAT006", "PAT016", "PAT019"]
DEFAULT_THRESHOLDS = [30, 50, 70, 100]


def candidates_overlap(hu, lung, gt, vmin):
    """Candidats brillants (>= vmin, <= MAX_HU, dans les poumons), sans plafond de volume."""
    bright = remove_small_objects((hu >= vmin) & (hu <= MAX_HU) & lung, 20)
    lab, n = label(bright, structure=ST)
    if n == 0:
        return 0, 0, 0.0
    counts = np.bincount(lab.ravel())
    keep = np.array([l for l in range(1, n + 1) if counts[l] >= MIN_VOL])
    if keep.size == 0:
        return 0, 0, 0.0
    kept = np.isin(lab, keep)
    lab_kept = lab * kept
    ncand = int(keep.size)
    vp = int(len(np.unique(lab_kept[gt & kept])))
    cov = 100 * (gt & kept).sum() / gt.sum() if gt.sum() else 0.0
    return ncand, vp, cov


def main():
    ap = argparse.ArgumentParser(description="Balayage du seuil vessel_min_hu (mode bright).")
    ap.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    ap.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    inv = get_available_patients(str(PROJECT_ROOT / "data"), force_raw=True)

    # 1) Charger + segmenter les poumons UNE fois par patient
    cache = {}
    for pid in args.patients:
        if pid not in inv["with_both"]:
            print(f"[SKIP] {pid} absent de l'inventaire RAW")
            continue
        print(f"... {pid} : chargement + segmentation poumons (lent)", flush=True)
        hu, mask, _ = load_patient_data(pid, inv)
        hu = hu.astype(np.float32)
        gt = mask > 0
        lung = segment_lungs(hu, hu_threshold=-400, min_area=1000)
        cache[pid] = (hu, lung, gt)

    # 2) Tester chaque seuil sur les poumons déjà segmentés
    print(f"\n{'vmin':>5} | {'détectés':>9} | {'couv.méd%':>9} | {'VP':>4} | {'FP':>6} | {'cand/pat':>8}")
    for vmin in args.thresholds:
        det = 0
        covs, tvp, tcand = [], 0, 0
        detected_ids = []
        for pid, (hu, lung, gt) in cache.items():
            nc, vp, cov = candidates_overlap(hu, lung, gt, vmin)
            covs.append(cov); tvp += vp; tcand += nc
            if vp > 0:
                det += 1; detected_ids.append(pid)
        n = len(cache)
        print(f"{vmin:5d} | {det:>4}/{n:<4} | {np.median(covs):9.1f} | {tvp:4d} | "
              f"{tcand - tvp:6d} | {tcand / n:8.0f}")

    # 3) À quel seuil chaque patient devient-il détecté ?
    print("\nSeuil le plus HAUT auquel chaque patient reste détecté (— = jamais) :")
    for pid, (hu, lung, gt) in cache.items():
        best = None
        for vmin in sorted(args.thresholds, reverse=True):
            _, vp, _ = candidates_overlap(hu, lung, gt, vmin)
            if vp > 0:
                best = vmin
                break
        print(f"  {pid:8} : {best if best is not None else '—'}")


if __name__ == "__main__":
    main()
