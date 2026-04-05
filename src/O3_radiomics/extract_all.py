# Auteur: Joseph DATE-MASSE
# src/O3_radiomics/extract_all.py

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter la racine du projet au PATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.O3_radiomics.feature_extractor import EmbolismFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Chemins de données
    processed_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_csv = output_dir / "radiomics_features.csv"
    
    if not processed_dir.exists():
        logger.error(f"Le dossier '{processed_dir}' n'existe pas. Assurez-vous d'avoir lancé le prétraitement.")
        return
        
    # Lister les dossiers de patients (ex: PAT001, PAT002...)
    patient_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith("PAT")])
    
    if len(patient_dirs) == 0:
        logger.warning(f"Aucun dossier patient trouvé dans '{processed_dir}'.")
        return
        
    logger.info(f"🚀 Début de l'extraction par lot sur {len(patient_dirs)} patients trouvés dans {processed_dir}")
    
    # Instanciation de l'extracteur léger
    extractor = EmbolismFeatureExtractor()
    all_features = []
    
    for i, p_dir in enumerate(patient_dirs):
        patient_id = p_dir.name
        vol_path = p_dir / "volume.npy"
        mask_path = p_dir / "mask.npy"
        
        if not vol_path.exists() or not mask_path.exists():
            logger.warning(f"⚠️ Données manquantes (volume.npy ou mask.npy) pour {patient_id}. Extraction ignorée.")
            continue
            
        logger.info(f"[{i+1}/{len(patient_dirs)}] Traitement du patient {patient_id}...")
        
        try:
            # Charger les matrices 3D numpy
            volume = np.load(vol_path)
            mask = np.load(mask_path)
            
            # NOTE: Si l'on souhaite extraire les Faux Positifs (FP), c'est ici qu'on 
            # passerait d'autres masques issus de l'Objectif 2 (Génération de candidats).
            # Par défaut, on extrait les vrais caillots (Ground Truth)
            features_dict = extractor.extract_from_arrays(volume, mask)
            
            if features_dict:
                # Ajout des identifiants dans le dictionnaire
                features_dict['PatientID'] = patient_id
                features_dict['has_embolism'] = 1  # Label=1 (Puisque ce sont les masques GT)
                
                all_features.append(features_dict)
                logger.info(f"✅ Features correctement calculées pour {patient_id}.")
            else:
                logger.warning(f"⚠️ Masque vide ou échec de l'extraction pour {patient_id}.")
                
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement de {patient_id}: {e}")
            
    # ------ Etape de concaténation finale (La Matrice) ------
    if len(all_features) > 0:
        df = pd.DataFrame(all_features)
        
        # Réorganiser les colonnes pour la lisibilité
        cols = ['PatientID', 'has_embolism'] + [c for c in df.columns if c not in ['PatientID', 'has_embolism']]
        df = df[cols]
        
        # Sauvegarder en CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"🎉 SUCCÈS TOTAL ! Extraction terminée. {len(df)} échantillons stockés dans {output_csv}")
    else:
        logger.warning("🚫 Aucune donnée n'a pu être extraite sur l'ensemble du dataset.")

if __name__ == "__main__":
    main()
