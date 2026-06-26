"""
BioReef.ai — Stage 1 (detect + classify).

Group 2x: everything on the per-frame detection/classification path lives here —
the backend-agnostic detector wrapper, the ViT backbone, MCEAM, and the HSLM
loss used to train the species head.

    _21_detector  -> _22_backbone -> _23_mceam   (+ _24_hslm_loss for training)
"""

from ._21_detector import (
    Detector,
    YOLODetector,
    RFDETRDetector,
    build_detector,
    Detections,
)
from ._22_backbone import ViTBackbone
from ._23_mceam import MCEAM
from ._24_hslm_loss import HSLMLoss

__all__ = [
    "Detector", "YOLODetector", "RFDETRDetector", "build_detector", "Detections",
    "ViTBackbone", "MCEAM", "HSLMLoss",
]
