# normalisation.py

import numpy as np
import logging
from typing import Tuple, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fenêtres Hounsfield standards en scanner thoracique
HU_WINDOWS = {
    'lung': (-600, 600),      # Fenêtre pulmonaire (pour embolies)
    'mediastinal': (40, 400),  # Fenêtre médiastinale
    'bone': (250, 1500),       # Fenêtre osseuse
    'full': (-1024, 3071)      # Plage complète scanner 12 bits
}
breakpoint()

def apply_windowing(volume: np.ndarray, 
                   window_min: float, 
                   window_max: float,
                   normalize: bool = True) -> np.ndarray:
    """
    Objectif 1 : Fenêtrage Hounsfield et normalisation
    
    Args:
        volume: Volume 3D d'intensités (idéalement en unités Hounsfield)
        window_min: Borne inférieure de la fenêtre HU
        window_max: Borne supérieure de la fenêtre HU
        normalize: Si True, normalise en [0, 1] après cliping
    
    Returns:
        Volume fenêtré (et normalisé si normalize=True)
    """
    # Validation des entrées
    if volume is None or volume.size == 0:
        logger.error("Volume vide ou None")
        return volume
    
    if window_min >= window_max:
        logger.error(f"Fenêtre invalide: min={window_min} >= max={window_max}")
        return volume
    
    # 1. Cliping
    windowed = np.clip(volume, window_min, window_max)
    
    # 2. Normalisation 
    if normalize:
        # Éviter la division par zéro
        denominator = window_max - window_min
        if denominator == 0:
            logger.warning("Fenêtre de largeur nulle, normalisation ignorée")
            return windowed
        
        normalized = (windowed - window_min) / denominator
        
        # Vérification rapide
        logger.debug(f"Normalisation: min={normalized.min():.3f}, max={normalized.max():.3f}")
        return normalized
    
    return windowed

def convert_to_hounsfield(dicom_volume: np.ndarray,
                         rescale_slope: Optional[float] = None,
                         rescale_intercept: Optional[float] = None) -> np.ndarray:
    """
    Convertit les pixels DICOM en unités Hounsfield (HU)
    
    Formule HU = pixel * slope + intercept
    Par défaut, slope=1, intercept=-1024 (air)
    """
    slope = rescale_slope if rescale_slope is not None else 1.0
    intercept = rescale_intercept if rescale_intercept is not None else -1024
    
    hu_volume = dicom_volume.astype(np.float32) * slope + intercept
    
    logger.info(f"Conversion HU: plage [{hu_volume.min():.1f}, {hu_volume.max():.1f}]")
    return hu_volume

def get_lung_window(volume: np.ndarray, 
                   method: str = 'adaptive') -> Tuple[float, float]:
    """
    Détermine automatiquement la fenêtre pulmonaire optimale
    
    Args:
        volume: Volume en HU
        method: 'adaptive' (basé sur histogramme) ou 'standard'
    
    Returns:
        (window_min, window_max)
    """
    if method == 'standard':
        return HU_WINDOWS['lung']
    
    elif method == 'adaptive':
        # Analyse de l'histogramme pour trouver les pics air/poumon/tissu
        hist, bins = np.histogram(volume.flatten(), bins=1000, 
                                  range=(-1024, 1024))
        
        # Trouver le pic de l'air (autour de -1000 HU)
        air_peak = bins[np.argmax(hist[:200])]  # Cherche dans [-1024, -824]
        
        # Fenêtre adaptative centrée sur le poumon (-500 HU en moyenne)
        window_min = max(-1024, air_peak + 200)  # Éviter l'air
        window_max = window_min + 1200  # Largeur typique 1200 HU
        
        logger.info(f"Fenêtre adaptative: [{window_min:.1f}, {window_max:.1f}]")
        return (window_min, window_max)
    
    else:
        logger.warning(f"Méthode '{method}' inconnue, utilisation standard")
        return HU_WINDOWS['lung']

def normalize_volume(volume: np.ndarray,
                    method: str = 'minmax',
                    percentiles: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """
    Différentes méthodes de normalisation
    
    Args:
        volume: Volume à normaliser
        method: 'minmax', 'zscore', 'percentile'
        percentiles: Pour la méthode 'percentile' (lower, upper)
    """
    if method == 'minmax':
        # Normalisation Min-Max classique
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin == 0:
            return volume
        return (volume - vmin) / (vmax - vmin)
    
    elif method == 'zscore':
        # Standardisation (moyenne=0, écart-type=1)
        mean = volume.mean()
        std = volume.std()
        if std == 0:
            return volume - mean
        return (volume - mean) / std
    
    elif method == 'percentile':
        # Normalisation basée sur les percentiles (robuste aux outliers)
        lower = np.percentile(volume, percentiles[0])
        upper = np.percentile(volume, percentiles[1])
        
        if upper - lower == 0:
            return volume
        
        # Cliper puis normaliser
        volume_clipped = np.clip(volume, lower, upper)
        return (volume_clipped - lower) / (upper - lower)
    
    else:
        logger.error(f"Méthode inconnue: {method}")
        return volume

class HounsfieldProcessor:
    """
    Classe pour gérer le prétraitement HU de manière cohérente
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.window = self.config.get('hu_window', HU_WINDOWS['lung'])
        self.normalize = self.config.get('normalize', True)
        
    def process(self, volume: np.ndarray, 
               dicom_metadata: dict = None) -> np.ndarray:
        """
        Pipeline complet de prétraitement HU
        """
        # 1. Conversion en HU si métadonnées fournies
        if dicom_metadata:
            slope = dicom_metadata.get('RescaleSlope', 1.0)
            intercept = dicom_metadata.get('RescaleIntercept', -1024)
            volume = convert_to_hounsfield(volume, slope, intercept)
        
        # 2. Fenêtrage
        window_min, window_max = self.window
        volume = apply_windowing(volume, window_min, window_max, 
                                normalize=self.normalize)
        
        return volume
    
    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

# Test simple si exécuté directement
if __name__ == "__main__":
    # Créer un volume factice simulant des HU
    test_volume = np.random.randint(-1024, 1024, (100, 100, 50))
    
    print("Test de normalisation:")
    
    # Test fenêtrage simple
    lung = apply_windowing(test_volume, -600, 600)
    print(f"Fenêtre pulmonaire: [{lung.min():.1f}, {lung.max():.1f}]")
    
    # Test fenêtrage sans normalisation
    lung_raw = apply_windowing(test_volume, -600, 600, normalize=False)
    print(f"Sans normalisation: [{lung_raw.min():.1f}, {lung_raw.max():.1f}]")
    
    # Test conversion HU simulée
    hu = convert_to_hounsfield(test_volume)
    print(f"Conversion HU: [{hu.min():.1f}, {hu.max():.1f}]")
    
    # Test classe processor
    processor = HounsfieldProcessor({'hu_window': (-600, 600)})
    result = processor(test_volume)
    print(f"Processor output: [{result.min():.3f}, {result.max():.3f}]")


'''
import numpy as np

def apply_windowing(volume, window_min, window_max):
    """
    Objectif 1 : Fenêtrage Hounsfield
    - window_min/max : plage de HU définie dans config.yaml
    """
    # 1. On "clipe" les valeurs : tout ce qui est hors plage est ramené aux bornes
    windowed_volume = np.clip(volume, window_min, window_max)
    
    # 2. Normalisation Min-Max : transforme la plage [min, max] en [0, 1]
    # Ce qui est crucial pour le Deep Learning 
    normalized_volume = (windowed_volume - window_min) / (window_max - window_min)
    
    return normalized_volume

'''
