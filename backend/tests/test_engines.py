# backend/tests/test_engines.py — bloc engines : équivalence des features O3 et registre des modèles
import json

import numpy as np
from scipy import ndimage

from backend.engines.classical import candidate_features
from backend.engines.registry import verify_manifest
from src.O3_radiomics.feature_extractor import EmbolismFeatureExtractor


def test_cropped_features_equal_full_volume_features():
    """Le moteur calcule les features sur une boîte englobante : elles doivent être identiques
    à celles de build_ml_dataset.py, qui travaille sur le volume entier."""
    rng = np.random.default_rng(0)
    vol = rng.normal(-300, 150, size=(40, 64, 64)).astype(np.float32)
    mask = np.zeros(vol.shape, dtype=bool)
    mask[10:16, 20:30, 20:27] = True        # candidat 1 : bloc
    mask[25:33, 40:44, 10:14] = True        # candidat 2 : tube allongé
    mask[5, 50:60, 50:52] = True            # candidat 3 : plat (cas dégénéré)
    vol[mask] = rng.normal(180, 40, size=mask.sum())
    labeled, n = ndimage.label(mask)
    vol = np.clip(vol, -600, 400)

    labels, rows = candidate_features(vol, labeled, (1.0, 1.0, 1.0))
    extractor = EmbolismFeatureExtractor()
    for lbl, feats in zip(labels, rows):
        full = extractor.extract_from_arrays(vol, (labeled == lbl).astype(np.uint8), spacing=(1.0, 1.0, 1.0))
        assert full.keys() == feats.keys()
        for k in full:
            assert np.isclose(full[k], feats[k], rtol=1e-6, atol=1e-9), k
    assert len(labels) == n


def test_manifest_detects_tampering(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"original")
    from backend.core.security import sha256_file
    (tmp_path / "manifest.json").write_text(json.dumps({"model.pkl": sha256_file(tmp_path / "model.pkl")}))
    assert verify_manifest(tmp_path) is None
    (tmp_path / "model.pkl").write_bytes(b"modifie")
    assert "invalide" in verify_manifest(tmp_path)


def test_missing_manifest(tmp_path):
    assert "non installés" in verify_manifest(tmp_path)
