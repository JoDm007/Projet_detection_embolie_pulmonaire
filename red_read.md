notebooks\02_preprocessing_demo.ipynb

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ajout du chemin vers 'src' pour que Python trouve tes modules
sys.path.append(os.path.abspath('../'))

#Importation des fonction pour le format DICOM
try:
    from src.O1_preprocessing.loader import load_dicom_volume
    from src.O1_preprocessing.normalisation import apply_windowing
    print("✅Importation reuissie !")
except ImportError as e:
    print(f"❌ Erreur d'imporation: {e}")
    
#Chemain vers un patient
patient_path= "../data/raw/CT_scans_01/PAT001"

if os.path.exists(patient_path):
    #1. Chargement du volume complet
    volume, spacing = load_dicom_volume(patient_path)

    #2. Fenetrage (obj 1.2: Plage [-600, 600])
    #On normalise pour que l'IA recoive des données entre 0 et 1
    vol_normalized = apply_windowing(volume, -600, 600)

    #3. Visualisation de la coupe centrale
    mid = volume.shape[0] // 2

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(volume[mid], cmap='gray')
    plt.title(f"Brut (HU) - Coupe {mid}")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(vol_normalized[mid], cmap='gray')
    plt.title("Fenetrage [-600, 600] (Normalisé)")
    plt.colorbar()

    plt.show()

    print(f"Dimensions du volume: {volume.shape}")
    print(f"Plage HU originale: Min {volume.min()} / Max {volume.max()}")
else:
    print(f"❌ Dossier introuvable : {patient_path}")


    
* À gauche (Brut) : L'image est terne, les poumons sont gris foncé et on distingue mal les structures internes.

* À droite (Fenêtrage) : Les poumons sont bien noirs (air), le contraste sur les vaisseaux et le cœur est renforcé.