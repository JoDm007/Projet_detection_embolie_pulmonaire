"""
scripts/diagnostics/check_gt_alignment.py

Vérifie l'alignement spatial entre le masque de vérité terrain (GT) et le volume.
Pour chaque patient porteur, calcule la fraction de voxels GT tombant dans l'air (HU < -400).
Un caillot ne peut pas être dans l'air : un %air élevé révèle un masque mal orienté.

Utilise les données PROCESSED (volume.npy normalisé [0,1] + mask.npy). La fenêtre de
prétraitement [-600, 400] mappe -400 HU -> 0.2 en valeur normalisée.

Usage (depuis la racine du projet) :
    python scripts/diagnostics/check_gt_alignment.py
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

AIR_NORM = 0.2          # (-400 HU + 600) / 1000  -> seuil "air" en valeur normalisée
SEUIL_ALERTE = 15.0     # %air au-delà duquel on suspecte un désalignement


def proc_to_hu(x):
    """Reconvertit une valeur normalisée [0,1] en HU (fenêtre [-600, 400])."""
    return x * 1000.0 - 600.0


def main():
    if not PROCESSED.exists():
        print(f"[ERREUR] {PROCESSED} introuvable. Lance d'abord le prétraitement.")
        sys.exit(1)

    rows = []
    for d in sorted(PROCESSED.glob("PAT*")):
        vol_f, mask_f = d / "volume.npy", d / "mask.npy"
        if not (vol_f.exists() and mask_f.exists()):
            continue
        vol = np.load(vol_f)
        gt = np.load(mask_f) > 0
        n = min(vol.shape[0], gt.shape[0])
        vol, gt = vol[:n], gt[:n]
        s = int(gt.sum())
        if s == 0:
            rows.append((d.name, 0, None, None))
            continue
        hu = proc_to_hu(vol[gt])
        air = float((hu < -400).mean()) * 100
        rows.append((d.name, s, air, float(np.median(hu))))

    porteurs = [r for r in rows if r[1] > 0]
    alertes = [r for r in porteurs if r[2] > SEUIL_ALERTE]

    print(f"{'patient':8} {'GT vox':>8} {'%air':>6} {'HU median':>10}  alerte")
    for pid, s, air, med in rows:
        if s == 0:
            print(f"{pid:8} {s:8d} {'--':>6} {'--':>10}")
            continue
        flag = "  <-- SUSPECT" if air > SEUIL_ALERTE else ""
        print(f"{pid:8} {s:8d} {air:6.1f} {med:10.0f}{flag}")

    print("\n--- Synthèse ---")
    print(f"Patients porteurs            : {len(porteurs)}")
    print(f"Alignés (%air < {SEUIL_ALERTE:.0f}%)       : {len(porteurs) - len(alertes)}/{len(porteurs)}")
    if alertes:
        print(f"SUSPECTS (désalignés ?)      : {[r[0] for r in alertes]}")
    else:
        print("Aucun patient suspect : alignement OK.")


if __name__ == "__main__":
    main()
