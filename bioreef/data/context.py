"""ContextHarvester — 4-stream multi-scale cropping (split from data_factory)."""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger("bioreef.data")


class ContextHarvester:
    """
    4-stream concentric crop extraction with Size-Adaptive ROI logic.

    Generates four synchronized, center-aligned crops at fixed resolution:
        - ROI (1x):  Morphological features (fins, eyes, scales)
        - Social (3x): School presence, predator/prey proximity
        - Habitat (5x): Micro-habitat (coral, sand, anemone substrate)
        - Full Frame:  Macro-environment (light, depth, turbidity)

    All crops undergo:
        - Aspect-ratio-preserving letterboxing (zero-pad)
        - Bicubic resize to target_resolution × target_resolution
        - ImageNet Z-score normalization (for DINOv2 compatibility)

    Size-Adaptive ROI:
        If a fish occupies < small_object_threshold of the frame area,
        the ROI is initially cropped at highres_initial (e.g. 512×512)
        before downsampling to preserve high-frequency texture details.

    Reference: Lee et al. (2026), MATANet — MCEAM requires all 4 streams
    normalized to 224×224 for spatial alignment during cross-attention.
    """

    # ImageNet normalization — mandatory for frozen DINOv2 ViT-B/14
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        crop_scales: List[int] = (1, 3, 5),
        target_resolution: int = 224,
        small_object_threshold: float = 0.05,
        highres_initial: int = 512,
        include_full_frame: bool = True,
    ):
        self.crop_scales = crop_scales
        self.target_res = target_resolution
        self.small_thresh = small_object_threshold
        self.highres_initial = highres_initial
        self.include_full_frame = include_full_frame

    def _extract_crop(
        self,
        frame: np.ndarray,
        cx: int, cy: int,
        crop_w: int, crop_h: int,
    ) -> np.ndarray:
        """
        Extract a single crop centered at (cx, cy) with zero-padding
        at frame boundaries.

        Args:
            frame:  Full-resolution image (H, W, 3).
            cx, cy: Center coordinates of the bounding box.
            crop_w: Desired crop width.
            crop_h: Desired crop height.

        Returns:
            Cropped region with zero-padding where it exceeds frame bounds.
        """
        h, w = frame.shape[:2]

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Clamp to frame boundaries
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        # Create zero-padded canvas
        crop = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)

        # Destination coordinates in the canvas
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        return crop

    def _letterbox_resize(self, image: np.ndarray, target: int) -> np.ndarray:
        """
        Resize with aspect-ratio preservation (letterboxing).

        Adds zero-padding to maintain the original aspect ratio before
        bicubic interpolation to the target square resolution.

        Ecological rationale:
            Elongated species like Great Barracuda (Sphyraena barracuda)
            would be distorted by naive square resize. Letterboxing preserves
            the body-to-head ratio critical for family-level classification.
        """
        h, w = image.shape[:2]
        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

        canvas = np.zeros((target, target, 3), dtype=image.dtype)
        pad_y = (target - new_h) // 2
        pad_x = (target - new_w) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        return canvas

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert to float tensor and apply ImageNet Z-score normalization.

        DINOv2 was self-supervised on ImageNet-normalized data; failure to
        match this normalization results in ~40% drop in feature quality
        (Oquab et al., 2023).
        """
        img = image.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    def harvest(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate the 4-stream context harvest for a single detection.

        Args:
            frame: Full-resolution BGR image (H, W, 3).
            bbox:  (x, y, w, h) bounding box of the detected fish.

        Returns:
            Dictionary with keys 'roi', 'social', 'habitat', 'full_frame',
            each a normalized tensor of shape (3, target_res, target_res).
        """
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        fish_area = w * h

        crops = {}

        for scale in self.crop_scales:
            crop_w = int(w * scale)
            crop_h = int(h * scale)

            raw_crop = self._extract_crop(frame, cx, cy, crop_w, crop_h)

            # Size-Adaptive ROI: high-res initial crop for small objects
            if scale == 1 and (fish_area / frame_area) < self.small_thresh:
                raw_crop = self._letterbox_resize(raw_crop, self.highres_initial)

            resized = self._letterbox_resize(raw_crop, self.target_res)
            tensor = self._normalize(resized)

            scale_name = {1: "roi", 3: "social", 5: "habitat"}.get(scale, f"context_{scale}x")
            crops[scale_name] = tensor

        # Full-frame macro-environment
        if self.include_full_frame:
            full_resized = self._letterbox_resize(frame, self.target_res)
            crops["full_frame"] = self._normalize(full_resized)

        return crops


