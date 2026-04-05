"""
Script de diagnostic pour vérifier les données prétraitées
"""
import numpy as np
from pathlib import Path

project_root = Path('.').resolve().parent
processed_dir = project_root / 'data' / 'processed'

# Prendre le premier patient
patient_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('PAT')])

if len(patient_dirs) == 0:
    print("❌ Aucun patient trouvé!")
else:
    p_dir = patient_dirs[0]
    vol_path = p_dir / 'volume.npy'
    mask_path = p_dir / 'mask.npy'
    
    print(f"📊 Analyse de {p_dir.name}")
    print(f"   Volume: {vol_path.exists()}, Mask: {mask_path.exists()}")
    
    if vol_path.exists():
        volume = np.load(vol_path)
        print(f"\n📐 Shape: {volume.shape}")
        print(f"📈 Range HU: [{volume.min():.1f}, {volume.max():.1f}]")
        print(f"📊 Mean: {volume.mean():.1f}, Std: {volume.std():.1f}")
        print(f"📉 Percentiles: 5%={np.percentile(volume, 5):.1f}, 95%={np.percentile(volume, 95):.1f}")
        
        # Vérifier la segmentation avec différents seuils
        print(f"\n🔍 Test segmentation:")
        for thresh in [-800, -600, -400, -200, 0]:
            n_voxels = (volume < thresh).sum()
            pct = 100 * n_voxels / volume.size
            print(f"   Seuil {thresh:4d} HU: {n_voxels:10d} voxels ({pct:5.2f}%)")
        
        # Afficher quelques slices
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            mid_z = volume.shape[0] // 2
            
            for i, idx in enumerate([mid_z - 20, mid_z, mid_z + 20]):
                if 0 <= idx < volume.shape[0]:
                    axes[i].imshow(volume[idx], cmap='gray', vmin=-1000, vmax=400)
                    axes[i].set_title(f'Slice {idx}')
                    axes[i].axis('off')
            
            plt.suptitle(f'{p_dir.name} - Exemples de slices')
            plt.tight_layout()
            plt.show()
        except:
            pass
