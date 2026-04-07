"""
BioReef.ai — Data Factory (Water-Net Experimental Branch)
==========================================================
This is an ISOLATED copy of the data factory that uses the REAL Water-Net
Gated Fusion Network for image restoration instead of the OpenCV fallback.

The ONLY difference from data_factory.py is the restorer class used.
The ContextHarvester and MarineAugmentor are imported directly from
the original data_factory.py to ensure a FAIR comparison.

DO NOT MODIFY data_factory.py — this file exists as a parallel branch.
"""

import os
import cv2
import numpy as np
import torch
import logging
from typing import Optional

# Import identical components from the original data factory
from bioreef.data.data_factory import ContextHarvester, MarineAugmentor

# Import the full Water-Net restorer
from bioreef.models.restoration_wn import WaterNetFullRestorer

logger = logging.getLogger("bioreef.data.waternet")


class WaterNetDataPipeline:
    """
    Drop-in replacement data pipeline using the actual WaterNet model.
    
    Identical to the original pipeline EXCEPT:
        - Uses WaterNetFullRestorer (Gated Fusion Network) instead of the
          OpenCV CLAHE/White Balance fallback.
    """

    def __init__(self, target_resolution=224, small_object_threshold=0.05, is_train=True, device="cpu"):
        self.harvester = ContextHarvester(
            target_resolution=target_resolution,
            small_object_threshold=small_object_threshold,
        )
        # The REAL Water-Net Gated Fusion Network
        self.restorer = WaterNetFullRestorer(device=device)
        self.augmentor = MarineAugmentor(enabled=is_train)

    def process(self, frame: np.ndarray, bbox) -> dict:
        """
        Run the full pipeline: Restore -> Augment -> Harvest.
        
        Args:
            frame: BGR uint8 numpy array.
            bbox: [x, y, w, h] bounding box.
            
        Returns:
            Dictionary of 4 normalized tensors.
        """
        restored = self.restorer(frame)
        augmented = self.augmentor(restored)
        streams = self.harvester.harvest(augmented, bbox)
        return streams
