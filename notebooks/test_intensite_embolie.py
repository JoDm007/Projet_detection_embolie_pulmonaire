"""
Test rapide pour vérifier l'intensité des embolies vs vaisseaux
Exécutez ceci pour confirmer la théorie avant de relancer
"""
import numpy as np
from pathlib import Path

# Charger un patient avec embolies
p_dir = Path('data/processed/PAT018')  # PAT018 a des candidats détectés
vol = np.load(p_dir / 'volume.npy')
mask = np.load(p_dir / 'mask.npy')

print(f"📊 Analyse intensité pour {p_dir.name}")
print(f"   Volume range: [{vol.min():.1f}, {vol.max():.1f}] HU")

# Analyser les embolies (masque GT)
embolie_voxels = vol[mask > 0]
if len(embolie_voxels) > 0:
    print(f"\n🎯 EMBOLIES (Ground Truth):")
    print(f"   Nombre de voxels: {len(embolie_voxels)}")
    print(f"   Intensité: [{embolie_voxels.min():.1f}, {embolie_voxels.max():.1f}] HU")
    print(f"   Moyenne: {embolie_voxels.mean():.1f} HU")
    print(f"   Médiane: {np.median(embolie_voxels):.1f} HU")
else:
    print("❌ Pas d'embolies dans ce patient")

# Analyser les zones détectées (haute intensité = vaisseaux)
# Le détecteur actuel trouve des zones > 30 HU
vessel_mask = (vol > 30) & (vol < 400)
vessel_voxels = vol[vessel_mask]
print(f"\n🩸 VAISSEAUX CONTRASTÉS (détectés actuellement):")
print(f"   Nombre de voxels: {len(vessel_voxels)}")
if len(vessel_voxels) > 0:
    print(f"   Intensité: [{vessel_voxels.min():.1f}, {vessel_voxels.max():.1f}] HU")
    print(f"   Moyenne: {vessel_voxels.mean():.1f} HU")

# Analyser les zones potentielles pour embolies
# Embolies = défauts de remplissage = intensité intermédiaire
potential_emboli = (vol > -100) & (vol < 50)
potential_voxels = vol[potential_emboli]
print(f"\n🔍 ZONES POTENTIELLES EMBOLIES (plage corrigée -100 à 50 HU):")
print(f"   Nombre de voxels: {len(potential_voxels)}")
if len(potential_voxels) > 0:
    print(f"   Intensité: [{potential_voxels.min():.1f}, {potential_voxels.max():.1f}] HU")
    print(f"   Moyenne: {potential_voxels.mean():.1f} HU")

# Overlap avec GT
if len(embolie_voxels) > 0:
    overlap = np.logical_and(potential_emboli, mask > 0).sum()
    total_emboli = mask.sum()
    print(f"\n✅ Overlap embolies détectées: {overlap}/{total_emboli} ({100*overlap/total_emboli:.1f}%)")
