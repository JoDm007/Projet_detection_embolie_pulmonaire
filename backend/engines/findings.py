# backend/engines/findings.py
# Construction du tableau des zones et de l'overlay à partir de composantes labellisées.
# Commun à tous les moteurs : le classique (candidats O2 retenus par O4) comme le futur
# U-Net (composantes connexes de la segmentation seuillée).
import numpy as np
from scipy import ndimage

from .base import Finding


def build_findings(labeled: np.ndarray, scored_labels: list[tuple[float, int]],
                   spacing) -> tuple[list[Finding], np.ndarray]:
    """labeled : (Z, Y, X), 0 = fond, l = composante l ; scored_labels : (score, l) retenus.
    Renvoie les zones triées par score décroissant (id 1..n) et l'overlay où la zone k vaut k."""
    overlay = np.zeros(labeled.shape, dtype=np.uint16)
    findings: list[Finding] = []
    if not scored_labels:
        return findings, overlay

    objects = ndimage.find_objects(labeled)
    voxel_mm3 = float(np.prod(spacing))
    width = labeled.shape[2]
    for idx, (score, lbl) in enumerate(sorted(scored_labels, reverse=True), start=1):
        sl = objects[lbl - 1]
        region = labeled[sl] == lbl
        overlay[sl][region] = idx
        zyx = np.argwhere(region).mean(axis=0) + [s.start for s in sl]
        z, y, x = (int(round(c)) for c in zyx)
        n_vox = int(region.sum())
        findings.append(Finding(
            id=idx, score=round(float(score), 4),
            slice_range=(int(sl[0].start), int(sl[0].stop - 1)),
            centroid_vox=(x, y, z),
            centroid_mm=(round(x * spacing[2], 1), round(y * spacing[1], 1),
                         round(z * spacing[0], 1)),
            # DICOM : l'axe X des colonnes va vers la gauche du patient
            side="droit" if x < width / 2 else "gauche",
            n_voxels=n_vox, volume_mm3=round(n_vox * voxel_mm3, 1)))
    return findings, overlay
