# src/O2_classical_detection/detector.py
import numpy as np
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# ✅ GESTION DES IMPORTS (fonctionne en direct ET en module)
try:
    # Mode 1 : Import relatif (quand exécuté via python -m)
    from .lung_segmentation import segment_lungs, extract_lung_roi
    from .thresholding import adaptive_threshold
    from .morphology import clean_candidates
    from .connected_components import extract_candidates_3d
except ImportError:
    # Mode 2 : Import absolu (quand exécuté directement)
    # Ajoute la racine du projet à sys.path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.O2_classical_detection.lung_segmentation import segment_lungs, extract_lung_roi
    from src.O2_classical_detection.thresholding import adaptive_threshold
    from src.O2_classical_detection.morphology import clean_candidates
    from src.O2_classical_detection.connected_components import extract_candidates_3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalDetector:
    """
    Pipeline complet de détection classique d'embolies pulmonaires.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Paramètres de segmentation pulmonaire
        self.lung_hu_threshold = self.config.get('lung_hu_threshold', -400)
        self.lung_min_area = self.config.get('lung_min_area', 1000)
        
        # Paramètres de seuillage
        self.threshold_method = self.config.get('threshold_method', 'percentile')
        self.threshold_percentile = self.config.get('threshold_percentile', 95)
        self.min_hu = self.config.get('min_hu', 50)
        self.max_hu = self.config.get('max_hu', 400)
        
        # Paramètres de filtrage morphologique
        self.min_candidate_volume = self.config.get('min_candidate_volume', 10)
        self.max_candidate_volume = self.config.get('max_candidate_volume', 10000)
        
    def detect(self, volume: np.ndarray, 
               spacing: Optional[Tuple[float, float, float]] = None) -> Dict:
        """
        Exécute le pipeline de détection complet.
        """
        logger.info("🚀 Démarrage de la détection classique...")
        
        # 1. Segmentation des poumons
        lung_mask = segment_lungs(
            volume, 
            hu_threshold=self.lung_hu_threshold,
            min_area=self.lung_min_area
        )
        
        # 2. Seuillage adaptatif dans les poumons
        candidates = adaptive_threshold(
            volume,
            lung_mask,
            method=self.threshold_method,
            percentile=self.threshold_percentile,
            min_hu=self.min_hu,
            max_hu=self.max_hu
        )
        
        # 3. Nettoyage morphologique
        candidates = clean_candidates(
            candidates,
            min_volume=self.min_candidate_volume,
            max_volume=self.max_candidate_volume
        )
        
        # 4. Extraction des features
        candidates_df = extract_candidates_3d(candidates, volume, spacing=spacing)
        
        # 5. Conversion des volumes en mm³ si spacing fourni
        if spacing is not None and len(candidates_df) > 0:
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            candidates_df['volume_mm3'] = candidates_df['volume_voxels'] * voxel_volume
        
        # 6. Masque labellisé (un label unique par candidat, pour build_ml_dataset)
        from scipy.ndimage import label as nd_label
        labeled_mask, _ = nd_label(candidates)

        result = {
            'lung_mask': lung_mask,
            'candidates_mask': candidates,
            'candidates_labeled': labeled_mask,  # masque avec un label par candidat
            'candidates_df': candidates_df,
            'n_candidates': len(candidates_df),
            'total_candidate_voxels': candidates.sum()
        }
        
        logger.info(f"✅ Détection terminée : {result['n_candidates']} candidats")
        
        return result


# ═══════════════════════════════════════════════════════════════
# BLOC DE TEST (pour exécution directe)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU DÉTECTEUR CLASSIQUE")
    print("="*60)
    
    # Créer des données factices pour le test
    test_volume = np.random.randn(50, 100, 100) * 100 + 0  # Simule des HU
    test_spacing = (1.0, 0.7, 0.7)
    
    print(f"📦 Volume test : {test_volume.shape}")
    print(f"📐 Spacing : {test_spacing}")
    
    # Configuration du détecteur
    config = {
        'lung_hu_threshold': -400,
        'threshold_method': 'percentile',
        'threshold_percentile': 95,
        'min_hu': 50,
        'max_hu': 400,
        'min_candidate_volume': 20,
        'max_candidate_volume': 5000
    }
    
    detector = ClassicalDetector(config)
    
    # Lancer la détection
    result = detector.detect(test_volume, spacing=test_spacing)
    
    print(f"\n🎯 Résultats :")
    print(f"   • Voxels pulmonaires : {result['lung_mask'].sum()}")
    print(f"   • Candidats détectés : {result['n_candidates']}")
    print(f"   • Voxels candidats : {result['total_candidate_voxels']}")
    
    if len(result['candidates_df']) > 0:
        print(f"\n📋 Top candidats :")
        print(result['candidates_df'][['region_id', 'volume_voxels', 'intensity_mean']].head())
    
    print("\n✅ Test du détecteur terminé !")