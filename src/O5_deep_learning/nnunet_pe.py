# src/O5_deep_learning/nnunet_pe.py
"""
Pipeline nnU-Net-style pour la détection et segmentation de l'embolie pulmonaire (FUMPE).

Ce fichier est auto-suffisant : il contient les 5 briques du pipeline.

    NNUNetPreprocessor  — normalisation CT adaptative (percentile 0.5/99.5), patch size calculé
    NNUNetModel         — U-Net 3D PyTorch, InstanceNorm, Leaky ReLU, Dice+CE loss
    NNUNetDataset       — Dataset patch-based + augmentation 3D (flip, rotation, élastique, gamma)
    NNUNetTrainer       — CV 5-folds groupée par patient (réutilise fold_assignments.csv d'O4),
                          early stopping sur Dice val, checkpoints par fold
    NNUNetPredictor     — sliding window avec overlap 50%, pondération gaussienne, overlay + volume mm³

Différences clés vs unet_3d_keras.py :
  - Framework  : PyTorch (pas TensorFlow/Keras)
  - Norm       : InstanceNorm3d (stable à batch_size=1-2)
  - Loss       : Dice + Binary Cross-Entropy combinés
  - Inférence  : sliding window gaussien (pas de resize brutal)
  - Augment.   : déformations élastiques 3D, gamma, mirroring complet
  - CV         : folds O4 réutilisés → comparaison équitable avec la baseline classique

Usage rapide :
    # Entraînement complet 5-folds
    python -m src.O5_deep_learning.nnunet_pe train

    # Prédiction sur un patient
    python -m src.O5_deep_learning.nnunet_pe predict PAT001

    # Smoke-test CPU (quelques secondes)
    python -m src.O5_deep_learning.nnunet_pe test

Dépendances : torch, numpy, scipy, pandas, pathlib (toutes déjà dans l'env EP).
  - torch n'est PAS dans requirements.txt ; à installer une seule fois :
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ou CPU-only :
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    map_coordinates,
    zoom,
)

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nnunet_pe")

# ---------------------------------------------------------------------------
# Vérification douce de PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} disponible — GPU: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch non installé. Installe-le avec :\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
        "ou CPU-only :\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    )


# ============================================================================
# 1. NNUNetPreprocessor
# ============================================================================

class NNUNetPreprocessor:
    """
    Preprocessing adaptatif inspiré de nnU-Net.

    Calcule les statistiques d'intensité sur l'ensemble du dataset (percentile 0.5/99.5),
    détermine le patch size optimal en fonction du spacing médian et de la VRAM cible,
    et fournit une fonction de normalisation prête à l'emploi.

    La résolution native est conservée (pas de rééchantillonnage isotropique) : décision
    d'équipe acée en Phase 0 — le spacing réel est disponible dans meta.json.

    Args:
        processed_dir  : chemin vers data/processed/ (dossiers PAT001..PAT035)
        patch_size     : si None, calculé automatiquement ; sinon tuple (Z, Y, X)
        vram_gb        : budget VRAM en Go pour le calcul automatique du patch size
        seed           : graine aléatoire pour reproductibilité
    """

    # Fenêtre HU validée en O1 — ne pas modifier
    HU_MIN: float = -600.0
    HU_MAX: float = 400.0

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        patch_size: Optional[Tuple[int, int, int]] = None,
        vram_gb: float = 8.0,
        seed: int = 42,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self._patch_size = patch_size
        self.vram_gb = vram_gb
        self.seed = seed

        # Statistiques calculées par analyse_dataset()
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.p005: Optional[float] = None  # percentile 0.5
        self.p995: Optional[float] = None  # percentile 99.5
        self.median_spacing: Optional[Tuple[float, float, float]] = None

    # ------------------------------------------------------------------
    # Statistiques du dataset
    # ------------------------------------------------------------------

    def analyse_dataset(self, patient_ids: Optional[List[str]] = None) -> Dict:
        """
        Parcourt les volumes prétraités pour calculer les statistiques d'intensité
        globales (percentile 0.5/99.5, moyenne, écart-type) et le spacing médian.

        Ces valeurs servent à la normalisation nnU-Net (clip + z-score).

        Args:
            patient_ids : liste de PAT### à analyser ; si None, prend tous les PAT* trouvés.

        Returns:
            dict avec 'p005', 'p995', 'mean', 'std', 'median_spacing', 'patch_size'
        """
        if patient_ids is None:
            patient_ids = sorted(
                p.name for p in self.processed_dir.glob("PAT*") if p.is_dir()
            )

        if not patient_ids:
            raise FileNotFoundError(
                f"Aucun dossier PAT* trouvé dans {self.processed_dir}"
            )

        logger.info(f"Analyse de {len(patient_ids)} patients pour les statistiques…")

        all_intensities: List[np.ndarray] = []
        spacings: List[Tuple[float, float, float]] = []

        for pid in patient_ids:
            vol_path = self.processed_dir / pid / "volume.npy"
            meta_path = self.processed_dir / pid / "meta.json"

            if not vol_path.exists():
                logger.warning(f"  {pid} : volume.npy absent, ignoré")
                continue

            # Charger le volume normalisé [0,1] et reconvertir en HU
            vol = np.load(vol_path, mmap_mode="r").astype(np.float32)
            vol_hu = vol * (self.HU_MAX - self.HU_MIN) + self.HU_MIN

            # Échantillonnage pour limiter la RAM (1 voxel sur 100)
            flat = vol_hu.ravel()
            rng = np.random.default_rng(self.seed)
            sample = rng.choice(flat, size=min(len(flat), 500_000), replace=False)
            all_intensities.append(sample)

            # Spacing depuis meta.json
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                sy = float(meta.get("spacing", [1.0, 1.0])[0])
                sx = float(meta.get("spacing", [1.0, 1.0])[1])
                sz = float(meta.get("slice_spacing", meta.get("slice_thickness", 1.0)))
                spacings.append((sz, sy, sx))

        if not all_intensities:
            raise RuntimeError("Aucun volume chargé — vérifier processed_dir.")

        combined = np.concatenate(all_intensities)
        self.p005 = float(np.percentile(combined, 0.5))
        self.p995 = float(np.percentile(combined, 99.5))
        # Clip puis z-score (comme nnU-Net)
        clipped = np.clip(combined, self.p005, self.p995)
        self.global_mean = float(clipped.mean())
        self.global_std = float(clipped.std()) + 1e-8

        if spacings:
            sp_arr = np.array(spacings)
            msp = tuple(float(np.median(sp_arr[:, i])) for i in range(3))
            self.median_spacing = msp  # type: ignore[assignment]
        else:
            self.median_spacing = (1.0, 1.0, 1.0)

        # Patch size adaptatif
        ps = self._compute_patch_size()

        stats = {
            "p005": self.p005,
            "p995": self.p995,
            "mean": self.global_mean,
            "std": self.global_std,
            "median_spacing": list(self.median_spacing),
            "patch_size": list(ps),
            "n_patients": len(patient_ids),
        }
        logger.info(
            f"Statistiques : p0.5={self.p005:.1f} HU, p99.5={self.p995:.1f} HU, "
            f"mean={self.global_mean:.1f}, std={self.global_std:.1f}"
        )
        logger.info(f"Patch size calculé : {ps}  |  Spacing médian : {self.median_spacing}")
        return stats

    def _compute_patch_size(self) -> Tuple[int, int, int]:
        """
        Calcule le patch size optimal selon la VRAM disponible.

        Heuristique nnU-Net simplifiée :
        - On vise ~100k voxels par patch pour 8 Go de VRAM avec batch=2.
        - On arrondit à un multiple de 16 (requis par 4 niveaux de pooling 2×).
        - L'axe Z est réduit si le spacing Z est plus grand (moins de coupes utiles).
        """
        if self._patch_size is not None:
            return self._patch_size

        # Budget voxels : ~100k pour 8 Go, proportionnel à la VRAM
        budget = int(100_000 * self.vram_gb / 8.0)

        sz_ratio = (
            (self.median_spacing[0] / self.median_spacing[1])
            if self.median_spacing and self.median_spacing[1] > 0
            else 1.0
        )
        # Patch en Y et X similaires, Z réduit si spacing Z > spacing Y
        side_xy = int(round(budget ** (1 / 3)))
        side_z = max(16, int(round(side_xy / max(sz_ratio, 1.0))))
        side_xy = max(32, side_xy)

        # Arrondir au multiple de 16 le plus proche
        def _round16(x: int) -> int:
            return max(16, int(round(x / 16)) * 16)

        ps = (_round16(side_z), _round16(side_xy), _round16(side_xy))
        self._patch_size = ps
        return ps

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        if self._patch_size is None:
            self._patch_size = self._compute_patch_size()
        return self._patch_size

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, volume_npy: np.ndarray) -> np.ndarray:
        """
        Normalise un volume prétraité (en [0,1]) en HU puis applique la
        normalisation nnU-Net : clip percentile + z-score.

        Args:
            volume_npy : volume float32 en [0, 1] (sortie du preprocessing O1)

        Returns:
            volume float32 normalisé, prêt pour le réseau
        """
        if self.global_mean is None:
            raise RuntimeError(
                "Appeler analyse_dataset() avant normalize(), "
                "ou charger les stats avec load_stats()."
            )
        # Reconversion en HU
        vol_hu = volume_npy.astype(np.float32) * (self.HU_MAX - self.HU_MIN) + self.HU_MIN
        # Clip percentile puis z-score
        vol_clipped = np.clip(vol_hu, self.p005, self.p995)
        vol_norm = (vol_clipped - self.global_mean) / self.global_std
        return vol_norm.astype(np.float32)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement des stats
    # ------------------------------------------------------------------

    def save_stats(self, path: str | Path) -> None:
        """Sauvegarde les statistiques dans un fichier JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "p005": self.p005,
            "p995": self.p995,
            "mean": self.global_mean,
            "std": self.global_std,
            "median_spacing": list(self.median_spacing) if self.median_spacing else None,
            "patch_size": list(self._patch_size) if self._patch_size else None,
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats sauvegardées : {path}")

    def load_stats(self, path: str | Path) -> None:
        """Charge les statistiques depuis un fichier JSON (évite de recalculer)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats non trouvées : {path}")
        with open(path) as f:
            stats = json.load(f)
        self.p005 = stats["p005"]
        self.p995 = stats["p995"]
        self.global_mean = stats["mean"]
        self.global_std = stats["std"]
        if stats.get("median_spacing"):
            self.median_spacing = tuple(stats["median_spacing"])  # type: ignore
        if stats.get("patch_size"):
            self._patch_size = tuple(stats["patch_size"])  # type: ignore
        logger.info(f"Stats chargées depuis {path}")


# ============================================================================
# 2. NNUNetModel
# ============================================================================

if TORCH_AVAILABLE:

    class _ConvNormAct(nn.Module):
        """Conv3d + InstanceNorm3d + LeakyReLU (brique de base nnU-Net)."""

        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel: int = 3,
            stride: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            pad = kernel // 2
            layers: List[nn.Module] = [
                nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class _DoubleConv(nn.Module):
        """Deux blocs ConvNormAct consécutifs (encodeur/décodeur)."""

        def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                _ConvNormAct(in_ch, out_ch, dropout=0.0),
                _ConvNormAct(out_ch, out_ch, dropout=dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class _EncoderBlock(nn.Module):
        """Bloc encodeur : DoubleConv + MaxPool3d (stride 2)."""

        def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.conv = _DoubleConv(in_ch, out_ch, dropout=dropout)
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            skip = self.conv(x)
            down = self.pool(skip)
            return skip, down

    class _DecoderBlock(nn.Module):
        """
        Bloc décodeur : ConvTranspose3d (upsampling) + concat skip + DoubleConv.
        La skip connection permet de récupérer les détails spatiaux fins.
        """

        def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            # Ajustement si dimensions ne correspondent pas exactement (padding impair)
            dz = skip.shape[2] - x.shape[2]
            dy = skip.shape[3] - x.shape[3]
            dx = skip.shape[4] - x.shape[4]
            if dz or dy or dx:
                x = F.pad(x, [0, dx, 0, dy, 0, dz])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class NNUNetModel(nn.Module):
        """
        U-Net 3D nnU-Net-style en PyTorch.

        Architecture :
            - 5 niveaux d'encodeur (features : 32 → 64 → 128 → 256 → 320)
            - Bottleneck à 320 filtres
            - 5 niveaux de décodeur avec skip connections
            - Convolution finale 1×1×1 → sigmoid (segmentation binaire)

        Choix techniques inspirés de nnU-Net v2 :
            - InstanceNorm3d (stable à batch_size = 1)
            - LeakyReLU (évite les neurones morts)
            - Dropout3d dans les couches profondes (régularisation sur 35 patients)
            - Sortie sigmoid (pas softmax) pour segmentation binaire

        Args:
            in_channels  : canaux d'entrée (1 pour CT monocanal)
            features     : nombre de filtres du premier niveau
            depth        : profondeur (niveaux d'encodeur/décodeur)
            dropout      : taux de dropout dans les blocs profonds
        """

        def __init__(
            self,
            in_channels: int = 1,
            features: int = 32,
            depth: int = 5,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.depth = depth

            # Calcul des canaux à chaque niveau — cap à 320 comme nnU-Net
            feat_map = [min(features * (2**i), 320) for i in range(depth + 1)]

            # Encodeur
            self.encoders = nn.ModuleList()
            in_ch = in_channels
            for i in range(depth):
                out_ch = feat_map[i]
                # Dropout uniquement dans les niveaux profonds (i >= depth//2)
                d = dropout if i >= depth // 2 else 0.0
                self.encoders.append(_EncoderBlock(in_ch, out_ch, dropout=d))
                in_ch = out_ch

            # Bottleneck
            self.bottleneck = _DoubleConv(feat_map[depth - 1], feat_map[depth], dropout=dropout)

            # Décodeur
            self.decoders = nn.ModuleList()
            for i in reversed(range(depth)):
                in_ch_dec = feat_map[i + 1]
                skip_ch = feat_map[i]
                out_ch_dec = feat_map[i]
                d = dropout if i >= depth // 2 else 0.0
                self.decoders.append(_DecoderBlock(in_ch_dec, skip_ch, out_ch_dec, dropout=d))

            # Sortie binaire
            self.final_conv = nn.Conv3d(feat_map[0], 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            Input  : (B, 1, Z, Y, X) float32
            Output : (B, 1, Z, Y, X) float32 en [0, 1] après sigmoid
            """
            skips: List[torch.Tensor] = []
            for enc in self.encoders:
                skip, x = enc(x)
                skips.append(skip)

            x = self.bottleneck(x)

            for dec, skip in zip(self.decoders, reversed(skips)):
                x = dec(x, skip)

            return torch.sigmoid(self.final_conv(x))

        def count_parameters(self) -> int:
            """Retourne le nombre de paramètres entraînables."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Fonctions de perte
    # ------------------------------------------------------------------

    def dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        """
        Soft Dice Loss pour la segmentation binaire.

        Robuste aux classes déséquilibrées (voxels positifs << voxels totaux),
        ce qui est critique sur FUMPE (emboles = ~0.1 % du volume).

        Args:
            pred   : prédiction sigmoid (B, 1, Z, Y, X)
            target : masque GT binaire float32 (B, 1, Z, Y, X)
            smooth : terme de lissage pour éviter la division par zéro
        """
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        return 1.0 - dice.mean()

    def combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        dice_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Dice Loss + Binary Cross-Entropy combinés (pondération 50/50 par défaut).

        - Dice : sensible aux voxels positifs rares → bon pour les emboles
        - BCE  : gradient stable sur tout le volume → stabilise l'entraînement

        Args:
            pred        : prédiction sigmoid (B, 1, Z, Y, X)
            target      : masque GT float32 (B, 1, Z, Y, X)
            dice_weight : poids du Dice dans la loss totale (1 - dice_weight pour BCE)
        """
        dl = dice_loss(pred, target)
        bce = F.binary_cross_entropy(pred, target)
        return dice_weight * dl + (1.0 - dice_weight) * bce


# ============================================================================
# 3. NNUNetDataset
# ============================================================================

if TORCH_AVAILABLE:

    class NNUNetDataset(Dataset):
        """
        Dataset PyTorch patch-based pour FUMPE.

        À chaque appel __getitem__, extrait un patch 3D aléatoire centré sur ou
        près d'un voxel positif (stratégie oversampling) ou dans le volume entier
        (stratégie background). Ratio 1:1 par défaut (1 patch positif / 1 fond).

        Augmentations 3D disponibles (mode train uniquement) :
            - Mirroring aléatoire (axes Z, Y, X)
            - Rotation 90° par multiple aléatoire
            - Bruit gaussien additif
            - Gamma augmentation (γ ∈ [0.7, 1.5])
            - Déformation élastique 3D légère

        Args:
            patient_ids   : liste des PAT### pour ce split
            preprocessor  : NNUNetPreprocessor avec stats calculées
            processed_dir : dossier data/processed/
            patch_size    : (Z, Y, X) du patch ; si None, utilise preprocessor.patch_size
            augment       : activer les augmentations (True pour train, False pour val/test)
            oversample    : fraction de patchs centrés sur un voxel positif
            patches_per_vol : nombre de patchs générés par volume par époque
            seed          : graine pour reproductibilité
        """

        def __init__(
            self,
            patient_ids: List[str],
            preprocessor: NNUNetPreprocessor,
            processed_dir: str | Path = "data/processed",
            patch_size: Optional[Tuple[int, int, int]] = None,
            augment: bool = False,
            oversample: float = 0.5,
            patches_per_vol: int = 50,
            seed: int = 42,
        ) -> None:
            self.patient_ids = patient_ids
            self.preprocessor = preprocessor
            self.processed_dir = Path(processed_dir)
            self.patch_size = patch_size or preprocessor.patch_size
            self.augment = augment
            self.oversample = oversample
            self.patches_per_vol = patches_per_vol
            self.rng = np.random.default_rng(seed)

            # Pré-chargement de tous les volumes en RAM
            # Sur 35 patients × ~240 Mo → ~8 Go max ; on charge float32 compressé
            logger.info(f"Chargement de {len(patient_ids)} volumes en mémoire…")
            self.volumes: Dict[str, np.ndarray] = {}
            self.masks: Dict[str, np.ndarray] = {}
            self.pos_indices: Dict[str, np.ndarray] = {}  # indices voxels positifs

            for pid in patient_ids:
                vol_path = self.processed_dir / pid / "volume.npy"
                mask_path = self.processed_dir / pid / "mask.npy"

                if not vol_path.exists() or not mask_path.exists():
                    logger.warning(f"  {pid} : fichiers manquants, ignoré")
                    continue

                vol = np.load(vol_path).astype(np.float32)
                mask = np.load(mask_path).astype(np.float32)

                # Normalisation nnU-Net
                vol = self.preprocessor.normalize(vol)

                self.volumes[pid] = vol
                self.masks[pid] = mask

                # Indices voxels positifs pour l'oversampling
                pos = np.argwhere(mask > 0)
                self.pos_indices[pid] = pos if len(pos) > 0 else np.empty((0, 3), dtype=int)

            self._valid_ids = list(self.volumes.keys())
            if not self._valid_ids:
                raise RuntimeError("Aucun patient chargé — vérifier processed_dir.")

            # Index plat : (patient_idx, patch_idx)
            self._index = [
                (pid, i)
                for pid in self._valid_ids
                for i in range(patches_per_vol)
            ]
            logger.info(
                f"Dataset prêt : {len(self._valid_ids)} patients, "
                f"{len(self._index)} patchs par époque"
            )

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            pid, _ = self._index[idx]
            vol = self.volumes[pid]
            mask = self.masks[pid]

            patch_vol, patch_mask = self._extract_patch(vol, mask)

            if self.augment:
                patch_vol, patch_mask = self._augment(patch_vol, patch_mask)

            # (Z, Y, X) → (1, Z, Y, X) pour PyTorch
            x = torch.from_numpy(patch_vol[np.newaxis]).float()
            y = torch.from_numpy(patch_mask[np.newaxis]).float()
            return x, y

        # ------------------------------------------------------------------
        # Extraction de patch
        # ------------------------------------------------------------------

        def _extract_patch(
            self, vol: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Extrait un patch de taille self.patch_size.

            Stratégie oversampling : avec probabilité self.oversample, le patch
            est centré sur un voxel positif (ou proche). Sinon, position aléatoire.
            """
            pz, py, px = self.patch_size
            z, y, x = vol.shape

            # Garantie que le patch tient dans le volume (zero-pad si nécessaire)
            if z < pz or y < py or x < px:
                vol, mask = self._pad_to_patch(vol, mask)
                z, y, x = vol.shape

            use_positive = (
                self.rng.random() < self.oversample
                and pid_has_positive(mask)
            )

            if use_positive:
                # Centre aléatoire parmi les voxels positifs
                pos = np.argwhere(mask > 0)
                cz, cy, cx = pos[self.rng.integers(len(pos))]
                # Décalage aléatoire autour du centre (±50 % du demi-patch)
                hz, hy, hx = pz // 2, py // 2, px // 2
                cz = int(np.clip(cz + self.rng.integers(-hz // 2, hz // 2 + 1), hz, z - pz + hz))
                cy = int(np.clip(cy + self.rng.integers(-hy // 2, hy // 2 + 1), hy, y - py + hy))
                cx = int(np.clip(cx + self.rng.integers(-hx // 2, hx // 2 + 1), hx, x - px + hx))
                z0, y0, x0 = cz - hz, cy - hy, cx - hx
            else:
                z0 = self.rng.integers(0, max(1, z - pz + 1))
                y0 = self.rng.integers(0, max(1, y - py + 1))
                x0 = self.rng.integers(0, max(1, x - px + 1))

            z0, y0, x0 = max(0, z0), max(0, y0), max(0, x0)
            patch_vol = vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px].copy()
            patch_mask = mask[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px].copy()

            # Pad final si bord du volume
            patch_vol, patch_mask = self._pad_to_patch(patch_vol, patch_mask)
            return patch_vol, patch_mask

        def _pad_to_patch(
            self, vol: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Zero-pad pour atteindre exactement self.patch_size."""
            pz, py, px = self.patch_size
            dz = max(0, pz - vol.shape[0])
            dy = max(0, py - vol.shape[1])
            dx = max(0, px - vol.shape[2])
            if dz or dy or dx:
                vol = np.pad(vol, [(0, dz), (0, dy), (0, dx)], mode="constant", constant_values=0.0)
                mask = np.pad(mask, [(0, dz), (0, dy), (0, dx)], mode="constant", constant_values=0.0)
            return vol, mask

        # ------------------------------------------------------------------
        # Augmentations 3D
        # ------------------------------------------------------------------

        def _augment(
            self, vol: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Augmentations 3D appliquées à vol ET mask simultanément.

            Ordre : mirroring → rotation → déformation élastique → bruit → gamma
            """
            # 1. Mirroring aléatoire (chaque axe indépendamment, p=0.5)
            for axis in range(3):
                if self.rng.random() < 0.5:
                    vol = np.flip(vol, axis=axis).copy()
                    mask = np.flip(mask, axis=axis).copy()

            # 2. Rotation 90° par multiple aléatoire dans le plan axial (Y-X)
            k = self.rng.integers(0, 4)
            if k > 0:
                vol = np.rot90(vol, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

            # 3. Déformation élastique 3D légère (p=0.3)
            if self.rng.random() < 0.3:
                vol, mask = self._elastic_deform(vol, mask)

            # 4. Bruit gaussien additif (p=0.5) — sur vol uniquement
            if self.rng.random() < 0.5:
                sigma = self.rng.uniform(0.0, 0.1)
                vol = vol + self.rng.normal(0, sigma, vol.shape).astype(np.float32)

            # 5. Gamma augmentation (p=0.5) — sur vol uniquement
            if self.rng.random() < 0.5:
                gamma = self.rng.uniform(0.7, 1.5)
                # Clip puis gamma correction dans [0,1] approximatif
                vmin, vmax = vol.min(), vol.max()
                if vmax > vmin:
                    vol_01 = (vol - vmin) / (vmax - vmin)
                    vol_01 = np.power(np.clip(vol_01, 0, 1), gamma)
                    vol = vol_01 * (vmax - vmin) + vmin

            return vol.astype(np.float32), mask.astype(np.float32)

        def _elastic_deform(
            self,
            vol: np.ndarray,
            mask: np.ndarray,
            alpha: float = 8.0,
            sigma: float = 3.0,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Déformation élastique 3D légère.
            alpha : amplitude de déplacement (voxels)
            sigma : lissage du champ de déformation
            """
            shape = vol.shape
            # Champs de déplacement aléatoires lissés
            dx = gaussian_filter(self.rng.random(shape).astype(np.float32) * 2 - 1, sigma) * alpha
            dy = gaussian_filter(self.rng.random(shape).astype(np.float32) * 2 - 1, sigma) * alpha
            dz = gaussian_filter(self.rng.random(shape).astype(np.float32) * 2 - 1, sigma) * alpha

            # Grille de coordonnées + déplacement
            z, y, x = np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
            )
            coords = [
                np.clip(z + dz, 0, shape[0] - 1),
                np.clip(y + dy, 0, shape[1] - 1),
                np.clip(x + dx, 0, shape[2] - 1),
            ]

            vol_def = map_coordinates(vol, coords, order=1, mode="constant").astype(np.float32)
            mask_def = map_coordinates(mask, coords, order=0, mode="constant").astype(np.float32)
            return vol_def, mask_def


    def pid_has_positive(mask: np.ndarray) -> bool:
        """Vrai si le masque contient au moins un voxel positif."""
        return bool(mask.max() > 0)


# ============================================================================
# 4. NNUNetTrainer
# ============================================================================

if TORCH_AVAILABLE:

    class NNUNetTrainer:
        """
        Entraînement CV 5-folds groupée par patient (réutilise fold_assignments.csv d'O4).

        Pour chaque fold :
            1. Construit les datasets train / val depuis les patients du fold
            2. Entraîne le modèle (early stopping sur Dice val)
            3. Sauvegarde le checkpoint au meilleur Dice val
            4. Enregistre l'historique dans results/deep_learning/

        Métriques suivies :
            - loss (Dice+CE) train et val
            - Dice Score voxel val (métrique principale d'early stopping)
            - Sensibilité patient @ seuil (calculée en fin de fold)

        Args:
            processed_dir  : dossier data/processed/
            folds_csv      : chemin vers fold_assignments.csv (O4)
            output_dir     : dossier de sortie pour checkpoints et logs
            preprocessor   : NNUNetPreprocessor (stats pré-calculées)
            patch_size     : (Z, Y, X) du patch
            patches_per_vol: patchs extraits par volume par époque
            batch_size     : taille du batch (1 ou 2 selon VRAM)
            lr             : learning rate initial
            epochs         : nombre max d'époques par fold
            patience       : patience early stopping (époques sans amélioration Dice val)
            device         : 'cuda', 'cpu', ou None (auto-détection)
        """

        def __init__(
            self,
            processed_dir: str | Path = "data/processed",
            folds_csv: str | Path = "results/classical_classifier/models/folds/fold_assignments.csv",
            output_dir: str | Path = "results/deep_learning",
            preprocessor: Optional[NNUNetPreprocessor] = None,
            patch_size: Optional[Tuple[int, int, int]] = None,
            patches_per_vol: int = 50,
            batch_size: int = 2,
            lr: float = 3e-4,
            epochs: int = 200,
            patience: int = 30,
            device: Optional[str] = None,
        ) -> None:
            self.processed_dir = Path(processed_dir)
            self.folds_csv = Path(folds_csv)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Preprocessor
            if preprocessor is None:
                preprocessor = NNUNetPreprocessor(processed_dir=self.processed_dir)
            self.preprocessor = preprocessor

            self.patch_size = patch_size
            self.patches_per_vol = patches_per_vol
            self.batch_size = batch_size
            self.lr = lr
            self.epochs = epochs
            self.patience = patience

            # Device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            logger.info(f"NNUNetTrainer — device: {self.device}")

            # Charger les assignments de folds
            if not self.folds_csv.exists():
                raise FileNotFoundError(
                    f"fold_assignments.csv non trouvé : {self.folds_csv}\n"
                    "Ce fichier est produit par O4 (train_classifier.py)."
                )
            self.fold_df = pd.read_csv(self.folds_csv)
            self.n_folds = int(self.fold_df["fold"].max()) + 1
            logger.info(
                f"Folds O4 chargés : {self.n_folds} folds, "
                f"{len(self.fold_df)} patients"
            )

        # ------------------------------------------------------------------
        # Entraînement complet 5-folds
        # ------------------------------------------------------------------

        def train_all_folds(self) -> Dict:
            """
            Lance l'entraînement sur les 5 folds et retourne un résumé des métriques.

            Returns:
                dict fold_id → {'best_dice_val', 'best_epoch', 'checkpoint_path'}
            """
            # Calculer les stats si pas encore fait
            if self.preprocessor.global_mean is None:
                stats_path = self.output_dir / "preprocessor_stats.json"
                if stats_path.exists():
                    self.preprocessor.load_stats(stats_path)
                else:
                    all_pids = self.fold_df["PatientID"].tolist()
                    self.preprocessor.analyse_dataset(all_pids)
                    self.preprocessor.save_stats(stats_path)

            ps = self.patch_size or self.preprocessor.patch_size
            results: Dict = {}

            for fold_id in range(self.n_folds):
                logger.info(f"\n{'='*60}")
                logger.info(f"  FOLD {fold_id} / {self.n_folds - 1}")
                logger.info(f"{'='*60}")

                val_ids = self.fold_df.loc[
                    self.fold_df["fold"] == fold_id, "PatientID"
                ].tolist()
                train_ids = self.fold_df.loc[
                    self.fold_df["fold"] != fold_id, "PatientID"
                ].tolist()

                logger.info(
                    f"  Train : {len(train_ids)} patients | Val : {len(val_ids)} patients"
                )

                result = self._train_one_fold(fold_id, train_ids, val_ids, ps)
                results[fold_id] = result

            # Résumé global
            dice_vals = [r["best_dice_val"] for r in results.values()]
            logger.info(
                f"\nRésumé 5-folds : Dice val moyen = {np.mean(dice_vals):.4f} "
                f"± {np.std(dice_vals):.4f}"
            )
            # Sauvegarder le résumé
            summary_path = self.output_dir / "cv_summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {str(k): v for k, v in results.items()},
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Résumé CV sauvegardé : {summary_path}")
            return results

        # ------------------------------------------------------------------
        # Entraînement d'un fold
        # ------------------------------------------------------------------

        def _train_one_fold(
            self,
            fold_id: int,
            train_ids: List[str],
            val_ids: List[str],
            patch_size: Tuple[int, int, int],
        ) -> Dict:
            # Datasets
            train_ds = NNUNetDataset(
                patient_ids=train_ids,
                preprocessor=self.preprocessor,
                processed_dir=self.processed_dir,
                patch_size=patch_size,
                augment=True,
                patches_per_vol=self.patches_per_vol,
            )
            val_ds = NNUNetDataset(
                patient_ids=val_ids,
                preprocessor=self.preprocessor,
                processed_dir=self.processed_dir,
                patch_size=patch_size,
                augment=False,
                patches_per_vol=self.patches_per_vol,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # 0 = pas de multiprocessing (Windows compatible)
                pin_memory=self.device.type == "cuda",
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )

            # Modèle
            model = NNUNetModel().to(self.device)
            logger.info(f"  Paramètres : {model.count_parameters():,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.lr / 100
            )

            checkpoint_path = self.output_dir / f"fold_{fold_id}_best.pth"
            history: Dict[str, List[float]] = {
                "train_loss": [], "val_loss": [], "val_dice": []
            }

            best_dice = -1.0
            best_epoch = 0
            no_improve = 0

            for epoch in range(1, self.epochs + 1):
                # --- Train ---
                model.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = combined_loss(pred, yb)
                    loss.backward()
                    # Gradient clipping pour stabilité
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # --- Val ---
                model.eval()
                val_losses, val_dices = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = model(xb)
                        val_losses.append(combined_loss(pred, yb).item())
                        val_dices.append(1.0 - dice_loss(pred, yb).item())

                mean_train = float(np.mean(train_losses))
                mean_val = float(np.mean(val_losses))
                mean_dice = float(np.mean(val_dices))

                history["train_loss"].append(mean_train)
                history["val_loss"].append(mean_val)
                history["val_dice"].append(mean_dice)

                scheduler.step()

                # Log toutes les 10 époques
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"  Epoch {epoch:4d}/{self.epochs} | "
                        f"train_loss={mean_train:.4f} | "
                        f"val_loss={mean_val:.4f} | "
                        f"val_dice={mean_dice:.4f}"
                    )

                # Early stopping + checkpoint
                if mean_dice > best_dice + 1e-4:
                    best_dice = mean_dice
                    best_epoch = epoch
                    no_improve = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_dice_val": best_dice,
                            "fold": fold_id,
                            "patch_size": list(patch_size),
                        },
                        checkpoint_path,
                    )
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    logger.info(
                        f"  Early stopping à l'époque {epoch} "
                        f"(meilleur Dice val : {best_dice:.4f} @ époque {best_epoch})"
                    )
                    break

            # Sauvegarder l'historique
            hist_path = self.output_dir / f"fold_{fold_id}_history.json"
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

            logger.info(
                f"  Fold {fold_id} terminé — meilleur Dice val : {best_dice:.4f} "
                f"@ époque {best_epoch}"
            )
            return {
                "fold": fold_id,
                "best_dice_val": best_dice,
                "best_epoch": best_epoch,
                "checkpoint_path": str(checkpoint_path),
                "train_patients": len(train_ids),
                "val_patients": len(val_ids),
            }


# ============================================================================
# 5. NNUNetPredictor
# ============================================================================

if TORCH_AVAILABLE:

    class NNUNetPredictor:
        """
        Inférence nnU-Net-style : sliding window avec overlap 50% et pondération gaussienne.

        La pondération gaussienne donne plus de poids aux prédictions au centre du patch
        qu'aux bords, ce qui évite les artefacts de couture visibles avec un moyennage uniforme.

        Le prédicteur lit le checkpoint du fold dont le patient était en validation
        (démo honnête : le modèle n'a jamais vu ce patient à l'entraînement).

        Args:
            output_dir     : dossier contenant les checkpoints fold_X_best.pth
            preprocessor   : NNUNetPreprocessor avec stats chargées
            folds_csv      : pour retrouver le fold de test d'un patient
            overlap        : fraction d'overlap entre patches (0.5 = 50%)
            threshold      : seuil de binarisation de la probabilité
            device         : 'cuda', 'cpu', ou None (auto-détection)
        """

        def __init__(
            self,
            output_dir: str | Path = "results/deep_learning",
            preprocessor: Optional[NNUNetPreprocessor] = None,
            folds_csv: str | Path = "results/classical_classifier/models/folds/fold_assignments.csv",
            overlap: float = 0.5,
            threshold: float = 0.5,
            device: Optional[str] = None,
        ) -> None:
            self.output_dir = Path(output_dir)
            self.folds_csv = Path(folds_csv)
            self.overlap = overlap
            self.threshold = threshold

            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            # Charger les stats du preprocessor
            if preprocessor is None:
                preprocessor = NNUNetPreprocessor()
                stats_path = self.output_dir / "preprocessor_stats.json"
                if stats_path.exists():
                    preprocessor.load_stats(stats_path)
                else:
                    raise FileNotFoundError(
                        f"preprocessor_stats.json absent de {self.output_dir}. "
                        "Lancer d'abord NNUNetTrainer.train_all_folds()."
                    )
            self.preprocessor = preprocessor

            # Charger le fold de chaque patient
            self.fold_map: Dict[str, int] = {}
            if self.folds_csv.exists():
                df = pd.read_csv(self.folds_csv)
                self.fold_map = dict(zip(df["PatientID"], df["fold"]))

            # Cache des modèles chargés par fold (évite de recharger à chaque patient)
            self._model_cache: Dict[int, Tuple[NNUNetModel, Tuple]] = {}

        # ------------------------------------------------------------------
        # Prédiction sur un patient
        # ------------------------------------------------------------------

        def predict_patient(
            self,
            patient_id: str,
            processed_dir: str | Path = "data/processed",
            spacing: Optional[Tuple[float, float, float]] = None,
        ) -> Dict:
            """
            Prédit le masque de segmentation pour un patient.

            Args:
                patient_id    : ex. 'PAT001'
                processed_dir : dossier contenant les volume.npy et mask.npy
                spacing       : (sz, sy, sx) en mm ; lu depuis meta.json si None

            Returns:
                dict avec :
                    'pred_mask'     : np.ndarray uint8 (Z, Y, X), 0/1
                    'prob_map'      : np.ndarray float32 (Z, Y, X), probabilités
                    'volume_mm3'    : volume du thrombus prédit en mm³
                    'dice_score'    : Dice vs GT si mask.npy disponible (sinon None)
                    'n_voxels'      : nombre de voxels positifs prédits
                    'fold_used'     : fold dont le modèle a été utilisé
            """
            processed_dir = Path(processed_dir)
            vol_path = processed_dir / patient_id / "volume.npy"
            mask_path = processed_dir / patient_id / "mask.npy"
            meta_path = processed_dir / patient_id / "meta.json"

            if not vol_path.exists():
                raise FileNotFoundError(f"volume.npy absent pour {patient_id}")

            vol = np.load(vol_path).astype(np.float32)
            vol_norm = self.preprocessor.normalize(vol)

            # Spacing
            if spacing is None and meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                sy = float(meta.get("spacing", [1.0, 1.0])[0])
                sx = float(meta.get("spacing", [1.0, 1.0])[1])
                sz = float(meta.get("slice_spacing", meta.get("slice_thickness", 1.0)))
                spacing = (sz, sy, sx)
            elif spacing is None:
                spacing = (1.0, 1.0, 1.0)

            # Choisir le fold (modèle entraîné sans ce patient)
            fold_id = self.fold_map.get(patient_id, 0)
            model, patch_size = self._load_model(fold_id)

            # Sliding window
            prob_map = self._sliding_window(vol_norm, model, patch_size)

            # Binarisation
            pred_mask = (prob_map >= self.threshold).astype(np.uint8)

            # Volume en mm³
            voxel_vol_mm3 = float(np.prod(spacing))
            n_voxels = int(pred_mask.sum())
            volume_mm3 = n_voxels * voxel_vol_mm3

            # Dice vs GT (si disponible)
            dice_score = None
            if mask_path.exists():
                gt = np.load(mask_path).astype(np.float32)
                inter = float((pred_mask * gt).sum())
                denom = float(pred_mask.sum() + gt.sum())
                dice_score = (2.0 * inter / (denom + 1e-7)) if denom > 0 else 0.0

            result = {
                "patient_id": patient_id,
                "pred_mask": pred_mask,
                "prob_map": prob_map,
                "volume_mm3": volume_mm3,
                "n_voxels": n_voxels,
                "dice_score": dice_score,
                "fold_used": fold_id,
                "spacing_mm": spacing,
            }

            logger.info(
                f"{patient_id} — fold={fold_id}, voxels={n_voxels}, "
                f"volume={volume_mm3:.1f} mm³"
                + (f", Dice={dice_score:.4f}" if dice_score is not None else "")
            )
            return result

        # ------------------------------------------------------------------
        # Évaluation sur tous les patients de val (par fold)
        # ------------------------------------------------------------------

        def evaluate_all_folds(
            self, processed_dir: str | Path = "data/processed"
        ) -> pd.DataFrame:
            """
            Évalue le modèle de chaque fold sur ses patients de validation.

            Retourne un DataFrame avec colonnes :
                PatientID, fold, dice_score, volume_mm3, n_voxels, has_gt_embolism
            """
            if not self.folds_csv.exists():
                raise FileNotFoundError(self.folds_csv)

            df_folds = pd.read_csv(self.folds_csv)
            rows = []

            for _, row in df_folds.iterrows():
                pid = row["PatientID"]
                try:
                    result = self.predict_patient(pid, processed_dir=processed_dir)
                    rows.append(
                        {
                            "PatientID": pid,
                            "fold": result["fold_used"],
                            "dice_score": result["dice_score"],
                            "volume_mm3": result["volume_mm3"],
                            "n_voxels": result["n_voxels"],
                            "has_embolism_pred": result["n_voxels"] > 0,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"  {pid} ignoré : {exc}")

            results_df = pd.DataFrame(rows)
            out_path = self.output_dir / "oof_predictions_dl.csv"
            results_df.to_csv(out_path, index=False)
            logger.info(f"Prédictions OOF sauvegardées : {out_path}")

            # Résumé
            detected = results_df["has_embolism_pred"].sum()
            mean_dice = results_df["dice_score"].dropna().mean()
            logger.info(
                f"Résumé : {detected}/{len(results_df)} patients avec EP détectée, "
                f"Dice moyen = {mean_dice:.4f}"
            )
            return results_df

        # ------------------------------------------------------------------
        # Sliding window gaussien
        # ------------------------------------------------------------------

        def _sliding_window(
            self,
            vol: np.ndarray,
            model: "NNUNetModel",
            patch_size: Tuple[int, int, int],
        ) -> np.ndarray:
            """
            Inférence par fenêtre glissante avec overlap 50% et pondération gaussienne.

            Pour chaque patch :
              1. Passer dans le modèle → carte de probabilité patch
              2. Multiplier par un poids gaussien centré (bords = poids faible)
              3. Accumuler dans le volume entier

            Résultat final : somme_pondérée / somme_poids

            Args:
                vol        : volume normalisé (Z, Y, X)
                model      : modèle chargé sur self.device
                patch_size : (pz, py, px)

            Returns:
                prob_map : (Z, Y, X) float32 en [0, 1]
            """
            pz, py, px = patch_size
            z, y, x = vol.shape

            # Pad si le volume est plus petit que le patch
            pad_z = max(0, pz - z)
            pad_y = max(0, py - y)
            pad_x = max(0, px - x)
            if pad_z or pad_y or pad_x:
                vol = np.pad(vol, [(0, pad_z), (0, pad_y), (0, pad_x)], mode="reflect")
            Z, Y, X = vol.shape

            # Carte d'accumulation
            pred_sum = np.zeros((Z, Y, X), dtype=np.float32)
            weight_sum = np.zeros((Z, Y, X), dtype=np.float32)

            # Poids gaussien pour un patch (centré, sigma = patch/4)
            gauss_weight = self._gaussian_patch_weight(patch_size)

            stride_z = max(1, int(pz * (1 - self.overlap)))
            stride_y = max(1, int(py * (1 - self.overlap)))
            stride_x = max(1, int(px * (1 - self.overlap)))

            # Génération des coins de patches (z0, y0, x0)
            zs = list(range(0, max(1, Z - pz + 1), stride_z))
            ys = list(range(0, max(1, Y - py + 1), stride_y))
            xs = list(range(0, max(1, X - px + 1), stride_x))

            # S'assurer que le coin final est inclus
            if zs[-1] + pz < Z:
                zs.append(Z - pz)
            if ys[-1] + py < Y:
                ys.append(Y - py)
            if xs[-1] + px < X:
                xs.append(X - px)

            model.eval()
            with torch.no_grad():
                for z0 in zs:
                    for y0 in ys:
                        for x0 in xs:
                            patch = vol[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                            # (1, 1, pz, py, px)
                            t = torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(self.device)
                            prob = model(t)[0, 0].cpu().numpy()  # (pz, py, px)

                            pred_sum[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += (
                                prob * gauss_weight
                            )
                            weight_sum[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += gauss_weight

            prob_map = pred_sum / (weight_sum + 1e-8)

            # Supprimer le padding si ajouté
            prob_map = prob_map[: z + pad_z or z, : y + pad_y or y, : x + pad_x or x]
            prob_map = prob_map[:z, :y, :x]
            return prob_map.astype(np.float32)

        @staticmethod
        def _gaussian_patch_weight(patch_size: Tuple[int, int, int]) -> np.ndarray:
            """
            Crée un poids gaussien 3D de taille patch_size.
            Sigma = patch_size / 4 par dimension.
            """
            pz, py, px = patch_size
            gz = np.exp(-0.5 * ((np.arange(pz) - pz / 2) / (pz / 4 + 1e-8)) ** 2)
            gy = np.exp(-0.5 * ((np.arange(py) - py / 2) / (py / 4 + 1e-8)) ** 2)
            gx = np.exp(-0.5 * ((np.arange(px) - px / 2) / (px / 4 + 1e-8)) ** 2)
            weight = gz[:, np.newaxis, np.newaxis] * gy[np.newaxis, :, np.newaxis] * gx[np.newaxis, np.newaxis, :]
            return weight.astype(np.float32)

        # ------------------------------------------------------------------
        # Chargement des modèles
        # ------------------------------------------------------------------

        def _load_model(
            self, fold_id: int
        ) -> Tuple["NNUNetModel", Tuple[int, int, int]]:
            """Charge (ou retourne depuis le cache) le modèle d'un fold."""
            if fold_id in self._model_cache:
                return self._model_cache[fold_id]

            ckpt_path = self.output_dir / f"fold_{fold_id}_best.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint fold {fold_id} absent : {ckpt_path}\n"
                    "Lancer d'abord NNUNetTrainer.train_all_folds()."
                )

            ckpt = torch.load(ckpt_path, map_location=self.device)
            model = NNUNetModel()
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device)
            model.eval()

            patch_size = tuple(ckpt.get("patch_size", [64, 128, 128]))
            logger.info(
                f"Checkpoint fold {fold_id} chargé "
                f"(époque {ckpt.get('epoch', '?')}, "
                f"Dice val {ckpt.get('best_dice_val', '?'):.4f})"
            )

            self._model_cache[fold_id] = (model, patch_size)
            return model, patch_size


# ============================================================================
# Point d'entrée CLI
# ============================================================================

def _run_smoke_test() -> None:
    """
    Smoke test CPU : vérifie que le modèle compile, fait un forward pass,
    calcule la loss, et teste le sliding window sur un volume synthétique.
    Ne nécessite pas de GPU ni de données réelles.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch non disponible — smoke test impossible.")
        return

    print("\n" + "=" * 60)
    print("  SMOKE TEST — nnunet_pe.py")
    print("=" * 60)

    # 1. Modèle
    model = NNUNetModel(features=16, depth=3, dropout=0.1)
    n_params = model.count_parameters()
    print(f"\n[1] Modèle instancié — {n_params:,} paramètres (depth=3, features=16)")

    # 2. Forward pass
    B, C, Z, Y, X = 2, 1, 32, 64, 64
    x = torch.randn(B, C, Z, Y, X)
    y_gt = (torch.rand(B, C, Z, Y, X) > 0.95).float()  # très déséquilibré
    pred = model(x)
    assert pred.shape == (B, C, Z, Y, X), f"Shape inattendu : {pred.shape}"
    print(f"[2] Forward pass OK — sortie : {tuple(pred.shape)}")

    # 3. Loss
    loss = combined_loss(pred, y_gt)
    loss.backward()
    print(f"[3] Backprop OK — loss = {loss.item():.4f}")

    # 4. Sliding window (CPU)
    preprocessor = NNUNetPreprocessor.__new__(NNUNetPreprocessor)
    preprocessor.global_mean = 0.0
    preprocessor.global_std = 1.0
    preprocessor.p005 = -600.0
    preprocessor.p995 = 400.0
    preprocessor._patch_size = (32, 64, 64)

    vol_synth = np.random.randn(48, 80, 80).astype(np.float32)
    predictor = NNUNetPredictor.__new__(NNUNetPredictor)
    predictor.device = torch.device("cpu")
    predictor.overlap = 0.5
    predictor.threshold = 0.5
    prob_map = predictor._sliding_window(vol_synth, model, (32, 64, 64))
    assert prob_map.shape == vol_synth.shape, f"Shape inattendu : {prob_map.shape}"
    print(f"[4] Sliding window OK — prob_map : {prob_map.shape}, range [{prob_map.min():.3f}, {prob_map.max():.3f}]")

    # 5. Augmentations
    ds = NNUNetDataset.__new__(NNUNetDataset)
    ds.rng = np.random.default_rng(42)
    ds.patch_size = (32, 64, 64)
    v = np.random.randn(32, 64, 64).astype(np.float32)
    m = (np.random.rand(32, 64, 64) > 0.98).astype(np.float32)
    va, ma = ds._augment(v, m)
    assert va.shape == v.shape
    print(f"[5] Augmentation 3D OK — shape conservé : {va.shape}")

    print("\n✅ Tous les tests passés !")
    print("=" * 60)


def _run_train(args: List[str]) -> None:
    """Lance l'entraînement complet 5-folds."""
    if not TORCH_AVAILABLE:
        print("PyTorch requis pour l'entraînement.")
        return
    trainer = NNUNetTrainer()
    trainer.train_all_folds()


def _run_predict(args: List[str]) -> None:
    """Prédit pour un patient donné."""
    if not TORCH_AVAILABLE:
        print("PyTorch requis pour la prédiction.")
        return
    if not args:
        print("Usage : python -m src.O5_deep_learning.nnunet_pe predict PAT001")
        return
    predictor = NNUNetPredictor()
    result = predictor.predict_patient(args[0])
    print(f"\nRésultats pour {args[0]} :")
    print(f"  Voxels EP prédits : {result['n_voxels']}")
    print(f"  Volume thrombus   : {result['volume_mm3']:.1f} mm³")
    if result["dice_score"] is not None:
        print(f"  Dice vs GT        : {result['dice_score']:.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "test"

    if cmd == "train":
        _run_train(argv[1:])
    elif cmd == "predict":
        _run_predict(argv[1:])
    elif cmd == "test":
        _run_smoke_test()
    else:
        print(f"Commande inconnue : {cmd}")
        print("Commandes disponibles : train | predict <PAT_ID> | test")
