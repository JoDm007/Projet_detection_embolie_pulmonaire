"""
src/O1_preprocessing/run_preprocessing.py

Script standalone pour régénérer data/processed/ depuis data/raw/.

Pipeline appliqué sur chaque patient :
  1. Chargement DICOM → conversion HU (slope/intercept)
  2. Fenêtrage Hounsfield [-600, 400]  (fenêtre pulmonaire EP avec contraste)
  3. Normalisation Min-Max → [0, 1]
  4. Sauvegarde volume.npy (float32) + mask.npy (uint8)

Usage (depuis la racine du projet) :
    python -m src.O1_preprocessing.run_preprocessing
    python -m src.O1_preprocessing.run_preprocessing --force   # écrase les .npy existants
    python -m src.O1_preprocessing.run_preprocessing --patient PAT001  # un seul patient
"""

import sys
import argparse
import logging
import shutil
import numpy as np
from pathlib import Path

# ── Assure que la racine du projet est dans sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.O1_preprocessing.loader import get_available_patients, load_patient_data

# ── Constantes de fenêtrage (alignées avec config.yaml et normalisation.py) ───
HU_MIN = -600   # borne basse : coupe l'air profond, garde le parenchyme pulmonaire
HU_MAX =  400   # borne haute : coupe l'os, garde vaisseaux contrastés + thrombus

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_patient(patient_id: str,
                       inventory: dict,
                       output_dir: Path,
                       force: bool = False) -> bool:
    """
    Prétraite un patient et sauvegarde volume.npy + mask.npy dans output_dir/patient_id/.

    Returns:
        True si le patient a été traité avec succès, False sinon.
    """
    patient_out = output_dir / patient_id
    vol_out  = patient_out / "volume.npy"
    mask_out = patient_out / "mask.npy"

    # Ignorer si déjà traité et --force non demandé
    if vol_out.exists() and mask_out.exists() and not force:
        logger.info(f"⏭️  {patient_id} : déjà traité (--force pour écraser)")
        return True

    # Charger depuis RAW (DICOM + .mat)
    volume, mask, meta = load_patient_data(patient_id, inventory)

    if volume is None or mask is None:
        logger.error(f"❌ {patient_id} : chargement échoué (volume={volume is not None}, mask={mask is not None})")
        return False

    # ── Fenêtrage Hounsfield [-600, 400] ──────────────────────────────────────
    # Le volume est déjà en HU grâce à load_dicom_volume (slope/intercept appliqués)
    vol_proc = np.clip(volume, HU_MIN, HU_MAX)

    # ── Normalisation Min-Max → [0, 1] ────────────────────────────────────────
    vol_proc = (vol_proc - HU_MIN) / (HU_MAX - HU_MIN)   # résultat ∈ [0.0, 1.0]
    vol_proc = vol_proc.astype(np.float32)

    # Vérification de cohérence
    v_min, v_max = float(vol_proc.min()), float(vol_proc.max())
    if v_min == v_max:
        logger.error(f"❌ {patient_id} : volume constant après normalisation ({v_min:.3f}) — données corrompues ?")
        return False

    # ── Binarisation du masque (0 = fond, 1 = embolie) ───────────────────────
    mask_proc = (mask > 0).astype(np.uint8)

    # Vérification cohérence des dimensions Z
    if vol_proc.shape[0] != mask_proc.shape[0]:
        logger.warning(f"⚠️  {patient_id} : Z différents volume={vol_proc.shape[0]}, mask={mask_proc.shape[0]}. Troncature.")
        min_z = min(vol_proc.shape[0], mask_proc.shape[0])
        vol_proc  = vol_proc[:min_z]
        mask_proc = mask_proc[:min_z]

    # Sauvegarde
    patient_out.mkdir(parents=True, exist_ok=True)
    np.save(vol_out,  vol_proc)
    np.save(mask_out, mask_proc)

    voxels_embolie = int(mask_proc.sum())
    pct = 100 * voxels_embolie / mask_proc.size
    logger.info(
        f"✅ {patient_id} : {vol_proc.shape} | "
        f"HU normalisé [{v_min:.3f}, {v_max:.3f}] | "
        f"Embolie : {voxels_embolie} voxels ({pct:.4f}%)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Régénère data/processed/ depuis data/raw/ avec fenêtre HU [-600, 400]"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Écraser les fichiers .npy existants"
    )
    parser.add_argument(
        "--patient", type=str, default=None,
        help="Traiter un seul patient (ex: --patient PAT001)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(PROJECT_ROOT / "data"),
        help="Chemin vers le dossier data/ (défaut: racine du projet)"
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = data_dir / "processed"

    logger.info("=" * 60)
    logger.info("🚀 PREPROCESSING FUMPE — Fenêtre HU [-600, 400]")
    logger.info("=" * 60)
    logger.info(f"📂 Source  : {data_dir / 'raw'}")
    logger.info(f"📂 Sortie  : {output_dir}")
    logger.info(f"🔧 Fenêtre : HU_MIN={HU_MIN}, HU_MAX={HU_MAX}")
    logger.info(f"🔧 Force   : {args.force}")

    # Charger l'inventaire RAW (force_raw=True pour ignorer processed/ existant)
    inventory = get_available_patients(str(data_dir), force_raw=True)

    if not inventory['with_both']:
        logger.error("❌ Aucun patient trouvé dans data/raw/. Vérifiez la structure du dossier.")
        sys.exit(1)

    patients = inventory['with_both']

    # Filtrer sur un seul patient si demandé
    if args.patient:
        if args.patient not in patients:
            logger.error(f"❌ Patient '{args.patient}' introuvable dans l'inventaire.")
            logger.info(f"   Patients disponibles : {patients}")
            sys.exit(1)
        patients = [args.patient]

    logger.info(f"📊 {len(patients)} patient(s) à traiter")

    # ── Traitement ────────────────────────────────────────────────────────────
    success_count = 0
    fail_list     = []

    for patient_id in patients:
        ok = preprocess_patient(patient_id, inventory, output_dir, force=args.force)
        if ok:
            success_count += 1
        else:
            fail_list.append(patient_id)

    # ── Résumé ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"✅ Terminé : {success_count}/{len(patients)} patients traités avec succès")
    if fail_list:
        logger.warning(f"⚠️  Échecs : {fail_list}")
    logger.info(f"📂 Résultats dans : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
