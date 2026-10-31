# backend/engines/base.py
# Contrat commun à tous les moteurs d'IA (cf. plan2 § 1.4). L'API et l'interface ne
# manipulent que ces types : brancher le U-Net O5 = écrire un nouveau moteur, rien d'autre.
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

# progress(étape lisible, avancement entre 0 et 1)
ProgressFn = Callable[[str, float], None]


class EngineUnavailable(Exception):
    """Le moteur ne peut pas traiter cette demande ; le message est montré à l'utilisateur."""


@dataclass
class Finding:
    """Une zone suspecte retenue au point de fonctionnement du moteur."""
    id: int                          # 1..n, valeur du label dans l'overlay
    score: float                     # probabilité / confiance du moteur
    slice_range: tuple[int, int]     # première et dernière coupe axiale (indices Z)
    centroid_vox: tuple[int, int, int]  # centre en voxels, ordre NIfTI (x, y, z)
    centroid_mm: tuple[float, float, float]  # même centre en mm depuis le coin du volume
    side: str                        # 'droit' / 'gauche' (côté du patient, orientation axiale standard)
    n_voxels: int
    volume_mm3: float


@dataclass
class AnalysisResult:
    engine: str
    findings: list[Finding]
    overlay: np.ndarray              # (Z, Y, X), 0 = fond, k = Finding.id
    n_candidates: int                # candidats examinés avant le seuil
    operating_point: dict            # seuil, budget de FP, sensibilité mesurée en CV…
    model_used: str                  # description lisible du modèle appliqué
    warnings: list[str] = field(default_factory=list)


class InferenceEngine(Protocol):
    name: str          # identifiant court, utilisé dans l'API
    label: str         # nom lisible pour l'interface
    description: str

    def analyse(self, volume_hu: np.ndarray, spacing: tuple[float, float, float],
                progress: ProgressFn, case_id: Optional[str] = None) -> AnalysisResult:
        """volume_hu : (Z, Y, X) en unités Hounsfield brutes ; spacing : (z, y, x) en mm.
        case_id : identifiant FUMPE pour un cas de démo, None pour un scanner importé."""
        ...
