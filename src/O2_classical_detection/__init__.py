# src/O2_classical_detection/__init__.py
"""
Package de détection classique d'embolie pulmonaire
"""

from .lung_segmentation import segment_lungs, extract_lung_roi
from .thresholding import adaptive_threshold
from .morphology import clean_candidates
from .connected_components import extract_candidates_3d, filter_candidates
from .detector import ClassicalDetector

__all__ = [
    'segment_lungs',
    'extract_lung_roi',
    'adaptive_threshold',
    'clean_candidates',
    'extract_candidates_3d',
    'filter_candidates',
    'ClassicalDetector'
]