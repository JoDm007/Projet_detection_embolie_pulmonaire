"""
scripts/diagnostics/measure_gate.py

« Gate » de la détection classique : les candidats O2 recouvrent-ils la vérité terrain (GT) ?
Pour chaque patient : nombre de candidats, combien touchent le GT (VP = Vrais Positifs),
et le pourcentage de voxels GT couverts. Sans ce recouvrement, la suite (radiomique, ML) n'a
pas de sens.

Lit data/processed/PATxxx/candidates.npz (clé 'labels', masque étiqueté) + mask.npy (GT).

Usage (depuis la racine du projet) :
    python scripts/diagnostics/measure_gate.py
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_candidates(path):
    """Charge le masque labellisé des candidats (format .npz compressé, clé 'labels')."""
    data = np.load(path)
    return data["labels"] if "labels" in data else data[data.files[0]]


def main():
    rows = []
    manquants = []
    for d in sorted(PROCESSED.glob("PAT*")):
        cf, mf = d / "candidates.npz", d / "mask.npy"
        if not mf.exists():
            continue
        if not cf.exists():
            manquants.append(d.name)
            continue
        cand = load_candidates(cf)
        gt = np.load(mf) > 0
        n = min(cand.shape[0], gt.shape[0])
        cand, gt = cand[:n], gt[:n]
        ncand = int(len(np.unique(cand[cand > 0])))
        gtv = int(gt.sum())
        vp = int(len(np.unique(cand[gt & (cand > 0)]))) if gtv else 0
        cov = 100 * (gt & (cand > 0)).sum() / gtv if gtv else None
        rows.append((d.name, ncand, vp, gtv, cov))

    if not rows:
        print("[INFO] Aucun candidates.npz trouvé. Génère-les via O2 (notebook 03 / detect_and_save).")
        if manquants:
            print(f"       candidates.npz manquants pour : {manquants}")
        sys.exit(0)

    porteurs = [r for r in rows if r[3] > 0]
    detectes = sum(1 for r in porteurs if r[2] > 0)
    tot_vp = sum(r[2] for r in rows)
    tot_cand = sum(r[1] for r in rows)

    print(f"{'patient':8} {'cand':>5} {'VP':>4} {'FP':>6} {'GTvox':>8} {'GT couv%':>9}")
    for pid, nc, vp, gtv, cov in rows:
        cov_s = f"{cov:8.1f}" if cov is not None else "   (--)"
        print(f"{pid:8} {nc:5d} {vp:4d} {nc - vp:6d} {gtv:8d} {cov_s}")

    print("\n--- Synthèse ---")
    print(f"Patients porteurs détectés (>=1 VP) : {detectes}/{len(porteurs)} "
          f"({100 * detectes / len(porteurs):.0f}%)")
    covs = [r[4] for r in porteurs if r[4] is not None]
    if covs:
        print(f"Couverture GT médiane / moyenne     : {np.median(covs):.1f}% / {np.mean(covs):.1f}%")
    print(f"Candidats totaux                    : {tot_cand}  (VP {tot_vp} / FP {tot_cand - tot_vp})")
    if manquants:
        print(f"candidates.npz manquants            : {manquants}")


if __name__ == "__main__":
    main()
