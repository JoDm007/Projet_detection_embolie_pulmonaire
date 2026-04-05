# split_data.py
import os
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_split(data_dir: str, 
                output_json: str = "data/splits/train_val_test.json", 
                train_ratio: float = 0.7, 
                val_ratio: float = 0.15, 
                test_ratio: Optional[float] = None,
                seed: int = 42,
                patient_prefix: str = "PAT",
                stratify_on: Optional[str] = None,
                metadata_file: Optional[str] = None,
                recursive: bool = True) -> Dict:
    """
    Sépare les patients en groupes Train, Val et Test.
    
    Args:
        data_dir: Dossier contenant les dossiers patients
        output_json: Chemin de sortie pour le fichier JSON
        train_ratio: Proportion pour l'entraînement (0.7)
        val_ratio: Proportion pour la validation (0.15)
        test_ratio: Proportion pour le test (si None, = 1 - train - val)
        seed: Graine aléatoire pour reproductibilité
        patient_prefix: Préfixe des dossiers patients (ex: "PAT")
        stratify_on: Stratification basée sur une métadonnée (ex: "hospital", "severity")
        metadata_file: Fichier JSON avec métadonnées pour stratification
        recursive: Si True, cherche les patients dans les sous-dossiers (CT_scans_*/PAT*)

    
    Returns:
        split_dict: Dictionnaire contenant les splits
    """

    data_path = Path(data_dir)
    
    if recursive:
        # Recherche récursive pour ta structure
        patients = sorted([d.name for d in data_path.rglob(f"{patient_prefix}*") 
                          if d.is_dir() and d.name.startswith(patient_prefix)])
        patients = list(dict.fromkeys(patients))  # Supprimer doublons
    else:
        # Comportement original
        patients = sorted([d.name for d in data_path.iterdir() 
                          if d.is_dir() and d.name.startswith(patient_prefix)])

    # 1. Validation des ratios
    if test_ratio is None:
        test_ratio = 1 - (train_ratio + val_ratio)
    
    if not 0.99 < train_ratio + val_ratio + test_ratio < 1.01:
        raise ValueError(f"Les ratios doivent sommer à 1: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
    
    # 2. Lister tous les dossiers de patients
    data_path = Path(data_dir)
    patients = sorted([d.name for d in data_path.iterdir() 
                      if d.is_dir() and d.name.startswith(patient_prefix)])
    
    if not patients:
        logger.error(f"Aucun dossier de patient trouvé dans {data_dir}")
        return {}
    
    logger.info(f"📊 Total patients trouvés: {len(patients)}")
    
    # 3. Stratification si demandée
    if stratify_on and metadata_file:
        split_dict = stratified_split(patients, metadata_file, stratify_on, 
                                     train_ratio, val_ratio, seed)
    else:
        split_dict = random_split(patients, train_ratio, val_ratio, test_ratio, seed)
    
    # 4. Ajouter les métadonnées
    split_dict["metadata"] = {
        "total_patients": len(patients),
        "seed": seed,
        "ratios": {
            "train": train_ratio, 
            "val": val_ratio, 
            "test": test_ratio
        },
        "stratification": stratify_on if stratify_on else "random",
        "data_dir": str(data_dir),
        "patient_prefix": patient_prefix
    }
    
    # 5. Statistiques supplémentaires
    split_dict["stats"] = {
        "train": {
            "count": len(split_dict["train"]),
            "percentage": f"{100*len(split_dict['train'])/len(patients):.1f}%"
        },
        "val": {
            "count": len(split_dict["val"]),
            "percentage": f"{100*len(split_dict['val'])/len(patients):.1f}%"
        },
        "test": {
            "count": len(split_dict["test"]),
            "percentage": f"{100*len(split_dict['test'])/len(patients):.1f}%"
        }
    }
    
    # 6. Sauvegarder le dictionnaire en fichier JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(split_dict, f, indent=4)
    
    logger.info(f"✅ Split terminé : {split_dict['stats']['train']['count']} Train, "
                f"{split_dict['stats']['val']['count']} Val, "
                f"{split_dict['stats']['test']['count']} Test")
    logger.info(f"📂 Fichier JSON créé : {output_json}")
    
    return split_dict



def random_split(patients: List[str], 
                train_ratio: float, 
                val_ratio: float, 
                test_ratio: float,
                seed: int) -> Dict:
    """Split aléatoire simple"""
    # Fixer la graine
    random.seed(seed)
    random.shuffle(patients)
    
    # Calculer les index de coupure
    total = len(patients)
    idx_train = int(total * train_ratio)
    idx_val = int(total * (train_ratio + val_ratio))
    
    # Répartir
    return {
        "train": patients[:idx_train],
        "val": patients[idx_train:idx_val],
        "test": patients[idx_val:]
    }

def stratified_split(patients: List[str],
                    metadata_file: str,
                    stratify_col: str,
                    train_ratio: float,
                    val_ratio: float,
                    seed: int) -> Dict:
    """
    Split stratifié pour garder la même distribution d'une caractéristique.
    Utile si certaines classes sont sous-représentées.
    """
    # Charger les métadonnées
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Créer un dictionnaire patient -> valeur de stratification
    patient_to_value = {}
    for patient in patients:
        if patient in metadata:
            patient_to_value[patient] = metadata[patient].get(stratify_col, "unknown")
        else:
            patient_to_value[patient] = "unknown"
    
    # Grouper les patients par valeur
    value_to_patients = {}
    for patient, value in patient_to_value.items():
        if value not in value_to_patients:
            value_to_patients[value] = []
        value_to_patients[value].append(patient)
    
    # Split stratifié
    train, val, test = [], [], []
    
    for value, pat_list in value_to_patients.items():
        # Mélanger localement
        random.seed(seed + hash(value) % 10000)
        random.shuffle(pat_list)
        
        # Calculer les tailles pour ce groupe
        n_total = len(pat_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Ajouter aux splits
        train.extend(pat_list[:n_train])
        val.extend(pat_list[n_train:n_train + n_val])
        test.extend(pat_list[n_train + n_val:])
    
    # Mélange final pour éviter un ordre par groupe
    random.seed(seed)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return {
        "train": train,
        "val": val,
        "test": test
    }

def load_split(json_path: str) -> Dict:
    """Charge un split sauvegardé"""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_patient_split(patient_id: str, split_dict: Dict) -> str:
    """Retourne le split d'un patient donné"""
    for split_name, patients in split_dict.items():
        if split_name in ["train", "val", "test"]:
            if patient_id in patients:
                return split_name
    return "unknown"

def create_cv_splits(data_dir: str,
                    n_folds: int = 5,
                    output_dir: str = "data/splits/cv",
                    seed: int = 42) -> List[Dict]:
    """
    Crée des splits pour cross-validation.
    Utile pour l'optimisation des hyperparamètres.
    """
    # Lister les patients
    data_path = Path(data_dir)
    patients = sorted([d.name for d in data_path.iterdir() 
                      if d.is_dir() and d.name.startswith("PAT")])
    
    random.seed(seed)
    random.shuffle(patients)
    
    # Créer les folds
    fold_size = len(patients) // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(patients)
        
        val_patients = patients[start:end]
        train_patients = [p for p in patients if p not in val_patients]
        
        fold_dict = {
            "train": train_patients,
            "val": val_patients,
            "test": [],  # Pas de test en CV
            "fold": i,
            "metadata": {
                "total_patients": len(patients),
                "seed": seed,
                "n_folds": n_folds
            }
        }
        
        # Sauvegarder
        output_path = Path(output_dir) / f"fold_{i}.json"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(fold_dict, f, indent=4)
        
        folds.append(fold_dict)
        logger.info(f"📂 Fold {i} créé: {len(train_patients)} train, {len(val_patients)} val")
    
    return folds

def verify_split_integrity(split_dict: Dict) -> bool:
    """Vérifie qu'il n'y a pas de chevauchement entre les splits"""
    train_set = set(split_dict["train"])
    val_set = set(split_dict["val"])
    test_set = set(split_dict["test"])
    
    # Vérifier les intersections
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap:
        logger.error(f"❌ Chevauchement Train-Val: {train_val_overlap}")
        return False
    if train_test_overlap:
        logger.error(f"❌ Chevauchement Train-Test: {train_test_overlap}")
        return False
    if val_test_overlap:
        logger.error(f"❌ Chevauchement Val-Test: {val_test_overlap}")
        return False
    
    logger.info("✅ Aucun chevauchement détecté entre les splits")
    return True


if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU MODULE SPLIT_DATA")
    print("="*60)
    
    # Créer un dossier temporaire avec des faux patients
    test_dir = "data/temp_test"
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(1, 51): # 50 faux patients
        os.makedirs(os.path.join(test_dir, f"PAT{i:03d}"), exist_ok=True)
    
    # Chemins des fichiers de test à nettoyer après
    test_outputs = [
        "data/splits/test_split.json",
        "data/splits/test_cv"
    ]
    
    try:
        # Test split simple
        print("\n📊 Test 1: Split aléatoire simple")
        split = create_split(
            data_dir=test_dir,
            output_json="data/splits/test_split.json",
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        # Afficher les résultats
        print(f"\nRésultats:")
        for split_name in ["train", "val", "test"]:
            print(f"  {split_name}: {len(split[split_name])} patients")
        
        # Vérifier l'intégrité
        verify_split_integrity(split)
        
        # Test chargement
        print("\n📂 Test 2: Chargement du split")
        loaded = load_split("data/splits/test_split.json")
        print(f"  Split chargé: {loaded['metadata']['total_patients']} patients")
        
        # Test cross-validation
        print("\n🔄 Test 3: Cross-validation splits")
        cv_folds = create_cv_splits(
            data_dir=test_dir,
            n_folds=5,
            output_dir="data/splits/test_cv"
        )
        print(f"  {len(cv_folds)} folds créés")    

    finally:
        # Nettoyage COMPLET
        import shutil
        
        # 1. Supprimer les faux patients
        shutil.rmtree(test_dir, ignore_errors=True)
        
        # 2. Supprimer les fichiers de split générés par le test
        for output in test_outputs:
            output_path = Path(output)
            if output_path.is_file():
                output_path.unlink()
                print(f"🗑️ Supprimé: {output}")
            elif output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
                print(f"🗑️ Supprimé dossier: {output}")
        
        print("\n✅ Tests terminés - Environnement nettoyé !")

"""
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU MODULE SPLIT_DATA")
    print("="*60)
    
    # Créer un dossier temporaire avec des faux patients
    test_dir = "data/temp_test"
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(1, 51):  # 50 faux patients
        os.makedirs(os.path.join(test_dir, f"PAT{i:03d}"), exist_ok=True)
    
    try:
        # Test split simple
        print("\n📊 Test 1: Split aléatoire simple")
        split = create_split(
            data_dir=test_dir,
            output_json="data/splits/test_split.json",
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        # Afficher les résultats
        print(f"\nRésultats:")
        for split_name in ["train", "val", "test"]:
            print(f"  {split_name}: {len(split[split_name])} patients")
        
        # Vérifier l'intégrité
        verify_split_integrity(split)
        
        # Test chargement
        print("\n📂 Test 2: Chargement du split")
        loaded = load_split("data/splits/test_split.json")
        print(f"  Split chargé: {loaded['metadata']['total_patients']} patients")
        
        # Test cross-validation
        print("\n🔄 Test 3: Cross-validation splits")
        cv_folds = create_cv_splits(
            data_dir=test_dir,
            n_folds=5,
            output_dir="data/splits/test_cv"
        )
        print(f"  {len(cv_folds)} folds créés")
        
    finally:
        # Nettoyage
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        print("\n✅ Tests terminés!")
        """