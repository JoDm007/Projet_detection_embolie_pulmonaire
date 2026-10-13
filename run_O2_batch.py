import sys, os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.O1_preprocessing.loader import get_available_patients
from src.O2_classical_detection.detector import ClassicalDetector

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'

# Get all patients with both volume and mask (RAW)
raw_inventory = get_available_patients(str(DATA_DIR), force_raw=True)

# Config
CONFIG = {
    'lung_hu_threshold': -400,
    'threshold_method': 'percentile',
    'threshold_percentile': 85,
    'min_hu': 20,
    'max_hu': 400,
    'min_candidate_volume': 10,
    'max_candidate_volume': 5000
}

detector = ClassicalDetector(CONFIG)

# Process patients starting from PAT023 (PAT001-PAT022 already done)
patients_to_process = [p for p in raw_inventory['with_both'] if p >= 'PAT023']

print(f"🚀 Batch O2: {len(patients_to_process)} patients à traiter (PAT023-PAT035)")

for i, pid in enumerate(patients_to_process, 1):
    print(f"\n[{i}/{len(patients_to_process)}] {pid}...")
    try:
        res = detector.detect_and_save(
            patient_id=pid,
            processed_dir=str(PROCESSED_DIR),
            raw_inventory=raw_inventory,
            force=False
        )
        if res:
            print(f"  ✅ {pid}: {res['n_candidates']} candidats")
        else:
            print(f"  ⏭️ {pid}: déjà traité")
    except Exception as e:
        print(f"  ❌ {pid}: ERREUR")
        print(f"     {e}")
        traceback.print_exc()

print(f"\n✅ Batch terminé")
