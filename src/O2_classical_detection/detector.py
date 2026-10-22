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
        self.min_candidate_volume = self.config.get('min_candidate_volume', 50)
        # max_candidate_volume = None → AUCUN plafond. Le cap à 5000 jetait les grosses
        # composantes brillantes posées sur le GT (embolies centrales jusqu'à ~190k voxels)
        # → 58 %/0 % de recouvrement. Cf. plan Phase 1 (baseline mesuré).
        self.max_candidate_volume = self.config.get('max_candidate_volume', None)

        # Mode de détection : 'bright' (approche B recommandée — régions brillantes du lumen
        # vasculaire, alignées sur le GT FUMPE) ou 'percentile' (ancien seuillage, conservé).
        self.detection_mode = self.config.get('detection_mode', 'bright')
        self.vessel_min_hu = self.config.get('vessel_min_hu', 100)
        
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
        
        # 2. Génération des candidats bruts selon le mode
        if self.detection_mode == 'bright':
            # Approche B : régions brillantes (lumen vasculaire opacifié) dans les poumons.
            # Alignée sur le GT FUMPE, qui annote le lumen et non le caillot sombre.
            from skimage.morphology import remove_small_objects
            candidates = (volume >= self.vessel_min_hu) & (volume <= self.max_hu) & lung_mask
            candidates = remove_small_objects(candidates, min_size=20)
            logger.info(f"   Seuillage brillant (≥{self.vessel_min_hu} HU) : {candidates.sum()} voxels candidats")
        else:
            # Ancien mode (seuillage percentile) — conservé pour comparaison.
            # adaptive_threshold() fait déjà opening+closing+remove_small_objects en interne.
            candidates = adaptive_threshold(
                volume,
                lung_mask,
                method=self.threshold_method,
                percentile=self.threshold_percentile,
                min_hu=self.min_hu,
                max_hu=self.max_hu
            )
            logger.info(f"   Après seuillage percentile : {candidates.sum()} voxels candidats")

        # 3. Filtre volumétrique : minimum obligatoire, maximum optionnel (None = pas de plafond)
        from scipy.ndimage import label as _label
        labeled_tmp, n_tmp = _label(candidates)
        counts = np.bincount(labeled_tmp.ravel())
        mask_filtered = np.zeros_like(candidates, dtype=bool)
        for rid in range(1, n_tmp + 1):
            vol = int(counts[rid])
            if vol < self.min_candidate_volume:
                continue
            if self.max_candidate_volume is not None and vol > self.max_candidate_volume:
                continue
            mask_filtered[labeled_tmp == rid] = True
        candidates = mask_filtered
        cap = self.max_candidate_volume if self.max_candidate_volume is not None else "∞"
        logger.info(f"   Après filtre volume [{self.min_candidate_volume}–{cap}] : {candidates.sum()} voxels")
        
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

    def detect_and_save(self,
                        patient_id: str,
                        processed_dir: str,
                        raw_inventory: Dict,
                        spacing: Optional[Tuple[float, float, float]] = None,
                        force: bool = False) -> Optional[Dict]:
        """
        Exécute la détection sur les données RAW (HU vraies) et sauvegarde
        candidates.npy dans data/processed/<patient_id>/.

        Cette méthode est le point d'entrée principal pour O2 → O3 :
        - Charge le volume DICOM depuis raw_inventory (HU brutes)
        - Applique le pipeline de détection avec les seuils HU physiologiques
        - Sauvegarde candidates_labeled.npy dans processed/ pour O3

        Args:
            patient_id:      Identifiant du patient (ex: 'PAT001')
            processed_dir:   Chemin vers data/processed/
            raw_inventory:   Inventaire retourné par get_available_patients(force_raw=True)
            spacing:         Espacement voxel (Z, Y, X) en mm — lu depuis les métadonnées DICOM si None
            force:           Si False, skip si candidates.npy existe déjà

        Returns:
            result dict si succès, None si échec ou skip
        """
        from src.O1_preprocessing.loader import load_patient_data

        out_path = Path(processed_dir) / patient_id / "candidates.npz"

        if out_path.exists() and not force:
            logger.info(f"⏭️  {patient_id} : candidates.npz déjà présent (--force pour écraser)")
            return None

        # Charger depuis RAW pour avoir les HU vraies
        logger.info(f"📦 Chargement RAW de {patient_id}...")
        volume, mask, meta = load_patient_data(patient_id, raw_inventory)

        if volume is None:
            logger.error(f"❌ {patient_id} : volume RAW introuvable")
            return None

        # Récupérer le spacing depuis les métadonnées DICOM si non fourni
        if spacing is None and meta:
            pixel_spacing = meta.get('spacing', [1.0, 1.0])
            slice_spacing = meta.get('slice_spacing', 1.0)
            spacing = (float(slice_spacing), float(pixel_spacing[0]), float(pixel_spacing[1]))
            logger.info(f"   Spacing DICOM : Z={spacing[0]:.2f} Y={spacing[1]:.2f} X={spacing[2]:.2f} mm")

        # Lancer la détection sur les HU vraies
        result = self.detect(volume, spacing=spacing)

        if result['n_candidates'] == 0:
            logger.warning(f"⚠️  {patient_id} : aucun candidat détecté")

        # Sauvegarder candidates_labeled dans processed/
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Stockage compressé : le masque labellisé est quasi-vide (sparse) → .npz = quelques Mo
        # au lieu de ~200 Mo pour un .npy dense int32. int16 suffit (< 32767 labels).
        np.savez_compressed(out_path, labels=result['candidates_labeled'].astype(np.int16))
        logger.info(f"💾 {patient_id} : {result['n_candidates']} candidats sauvegardés → {out_path} (compressé)")

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