"""
scripts/diagnostics/compare_detection_ab.py

Compare deux stratégies de génération de candidats, en mesurant le recouvrement avec le GT :

  A - DÉFAUT SOMBRE : zones sombres (plage caillot) à l'intérieur du lumen vasculaire rempli.
      (= définition clinique de l'embolie : défaut de remplissage)
  B - BRILLANT       : composantes brillantes (lumen opacifié) sans plafond de volume.
      (= aligné sur le GT FUMPE, qui annote le lumen et non le caillot sombre)

Charge les données RAW (vraies HU) et segmente les poumons -> LENT (~45 s/patient).

Usage (depuis la racine du projet) :
    python scripts/diagnostics/compare_detection_ab.py --patients PAT002 PAT018 PAT005
    python scripts/diagnostics/compare_detection_ab.py            # 6 patients par défaut
"""
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label
from skimage.morphology import remove_small_objects

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.O1_preprocessing.loader import get_available_patients, load_patient_data
from src.O2_classical_detection.lung_segmentation import segment_lungs

ST = generate_binary_structure(3, 1)
DEFAULT = ["PAT002", "PAT022", "PAT018", "PAT019", "PAT005", "PAT017"]


def _labeled(binmask, min_vol):
    """Composantes connexes filtrées par volume minimum ; renvoie (masque labellisé, nb)."""
    lab, n = label(binmask, structure=ST)
    if n == 0:
        return lab, 0
    counts = np.bincount(lab.ravel())
    out = np.zeros_like(lab, dtype=np.int32)
    nid = 0
    for l in range(1, n + 1):
        if counts[l] >= min_vol:
            nid += 1
            out[lab == l] = nid
    return out, nid


def _overlap(labeled, gt):
    """(#candidats, #VP touchant le GT, % voxels GT couverts)."""
    nc = int(labeled.max())
    if nc == 0:
        return 0, 0, 0.0
    vp = int(len(np.unique(labeled[gt & (labeled > 0)])))
    cov = 100 * (gt & (labeled > 0)).sum() / gt.sum() if gt.sum() else 0.0
    return nc, vp, cov


def approach_A(hu, lung, vmin=150, dlo=-50, dhi=120, min_vol=10):
    """Défaut sombre : trous sombres dans le lumen vasculaire (rempli coupe par coupe)."""
    vessel = remove_small_objects((hu >= vmin) & lung, 30)
    filled = np.zeros_like(vessel)
    for z in range(vessel.shape[0]):
        if vessel[z].any():
            filled[z] = binary_fill_holes(vessel[z])
    defect = filled & ~vessel & (hu >= dlo) & (hu <= dhi) & lung
    return _labeled(defect, min_vol)


def approach_B(hu, lung, vmin=100, min_vol=50):
    """Brillant sans plafond de volume (= mode 'bright' du détecteur)."""
    bright = remove_small_objects((hu >= vmin) & lung, 20)
    return _labeled(bright, min_vol)


def main():
    ap = argparse.ArgumentParser(description="Comparaison A (défaut sombre) vs B (brillant).")
    ap.add_argument("--patients", nargs="+", default=DEFAULT, help="Liste de PATxxx")
    args = ap.parse_args()

    inv = get_available_patients(str(PROJECT_ROOT / "data"), force_raw=True)
    results = {"A (defaut sombre)": [], "B (brillant sans cap)": []}

    for pid in args.patients:
        if pid not in inv["with_both"]:
            print(f"[SKIP] {pid} absent de l'inventaire RAW")
            continue
        print(f"... {pid} : chargement + segmentation poumons (lent)")
        hu, mask, _ = load_patient_data(pid, inv)
        hu = hu.astype(np.float32)
        gt = mask > 0
        lung = segment_lungs(hu, hu_threshold=-400, min_area=1000)
        for name, fn in [("A (defaut sombre)", approach_A), ("B (brillant sans cap)", approach_B)]:
            lab, _ = fn(hu, lung)
            results[name].append((pid, *_overlap(lab, gt)))

    for name, rows in results.items():
        if not rows:
            continue
        det = sum(1 for r in rows if r[2] > 0)
        covs = [r[3] for r in rows]
        tvp = sum(r[2] for r in rows)
        tcand = sum(r[1] for r in rows)
        print(f"\n########## Approche {name} ##########")
        print(f"{'patient':8} {'cand':>5} {'VP':>4} {'GT couv%':>9}")
        for pid, nc, vp, cov in rows:
            print(f"{pid:8} {nc:5d} {vp:4d} {cov:9.1f}")
        print(f"  -> détectés {det}/{len(rows)} | couverture médiane {np.median(covs):.1f}% "
              f"| VP {tvp} / FP {tcand - tvp}")


if __name__ == "__main__":
    main()
