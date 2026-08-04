"""MarineAugmentor — domain augmentation (split from data_factory)."""
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

logger = logging.getLogger("bioreef._1_preprocess")


class MarineAugmentor:
    """
    Underwater-domain augmentation, kept LIGHT because the backbone is frozen —
    strong geometric distortion (esp. full 0-360 rotation) pushes crops off the
    frozen feature manifold and hurts accuracy (KNOWN_BUGS #4: measured ~20 top-1
    on DINOv2). Defaults: h-flip, small +/-30 SYMMETRIC rotation, mild
    noise/snow/blur/photometric; vertical flip OFF (fish are rarely upside-down).

    Training uses transform_streams (KNOWN_BUGS #3): the crops are already
    extracted, so flips/rotations keep the fish in frame, and the geometric
    transform is sampled ONCE and shared across streams to stay MCEAM-coherent.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,     # KNOWN_BUGS #4: off (was 0.3)
        rotation_limit: int = 30,            # KNOWN_BUGS #4: +/-30 symmetric (was 360)
        noise_var_limit: Tuple[float, float] = (5.0, 15.0),  # #4: was (10, 50)
        marine_snow_prob: float = 0.1,       # KNOWN_BUGS #4: was 0.3
        marine_snow_density: float = 0.005,
        marine_snow_opacity: float = 0.4,
        motion_blur_prob: float = 0.1,       # KNOWN_BUGS #4: was 0.2
        motion_blur_limit: int = 7,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1,
        saturation_limit: float = 0.1,
        enabled: bool = True,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_limit = rotation_limit
        self.noise_var_limit = noise_var_limit
        self.marine_snow_prob = marine_snow_prob
        self.marine_snow_density = marine_snow_density
        self.marine_snow_opacity = marine_snow_opacity
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_limit = motion_blur_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.saturation_limit = saturation_limit
        self.enabled = enabled

    def _apply_geometric(self, image: np.ndarray) -> np.ndarray:
        """Random flips and rotation (symmetric +/-rotation_limit, KNOWN_BUGS #4)."""
        if np.random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image).copy()
        if np.random.random() < self.vertical_flip_prob:
            image = np.flipud(image).copy()

        if self.rotation_limit > 0:
            angle = np.random.uniform(-self.rotation_limit, self.rotation_limit)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return image

    def _apply_turbidity_noise(self, image: np.ndarray) -> np.ndarray:
        """Poisson-Gaussian noise simulating suspended particles."""
        var = np.random.uniform(*self.noise_var_limit)
        gaussian = np.random.normal(0, var**0.5, image.shape).astype(np.float32)

        noisy = image.astype(np.float32) + gaussian
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_marine_snow(self, image: np.ndarray) -> np.ndarray:
        """Random white dot overlay simulating organic marine snow particles."""
        if np.random.random() > self.marine_snow_prob:
            return image

        h, w = image.shape[:2]
        num_particles = int(h * w * self.marine_snow_density)
        overlay = image.copy().astype(np.float32)

        for _ in range(num_particles):
            px = np.random.randint(0, w)
            py = np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            cv2.circle(
                overlay, (px, py), radius,
                (255, 255, 255), -1
            )

        # Blend with original
        blended = cv2.addWeighted(
            image.astype(np.float32), 1.0 - self.marine_snow_opacity,
            overlay, self.marine_snow_opacity, 0
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """Directional motion blur simulating camera shake or fast swimmers."""
        if np.random.random() > self.motion_blur_prob:
            return image

        ksize = int(np.random.choice(range(3, self.motion_blur_limit + 1, 2)))
        angle = np.random.uniform(0, 360)

        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize

        M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel = kernel / kernel.sum()

        return cv2.filter2D(image, -1, kernel)

    def _apply_photometric_jitter(self, image: np.ndarray) -> np.ndarray:
        """Random brightness, contrast, and saturation shifts (±10%)."""
        beta = np.random.uniform(-self.brightness_limit, self.brightness_limit)
        alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)

        # KNOWN_BUGS #6: NOT cv2.convertScaleAbs — it computes abs(alpha*px + beta)
        # before the uint8 cast, so a negative brightness offset REFLECTS dark
        # pixels back up instead of clipping (px=10, beta=-51 -> 41, not 0). On
        # dark underwater crops that inverts exactly the fish regions. Use float32
        # arithmetic + clip.
        result = image.astype(np.float32) * alpha + beta * 255
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Saturation in HSV space
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_factor = 1.0 + np.random.uniform(-self.saturation_limit, self.saturation_limit)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    # --- Stream-level augmentation (KNOWN_BUGS #3) ---------------------------
    # Training augments the ALREADY-CROPPED streams, not the frame before cropping
    # (which moves the fish out of its bbox on flips/rotations). The geometric
    # transform is sampled ONCE and shared across all streams so they stay
    # spatially aligned for MCEAM cross-attention; photometric + noise are drawn
    # per stream.

    def _sample_geometric(self):
        """Draw ONE geometric transform (flip flags + rotation angle) to share
        across all context streams."""
        return {
            "hflip": np.random.random() < self.horizontal_flip_prob,
            "vflip": np.random.random() < self.vertical_flip_prob,
            "angle": (np.random.uniform(-self.rotation_limit, self.rotation_limit)
                      if self.rotation_limit > 0 else 0.0),
        }

    def _apply_geometric_params(self, image: np.ndarray, p: dict) -> np.ndarray:
        """Apply a pre-sampled geometric transform (same params for every stream)."""
        if p["hflip"]:
            image = np.fliplr(image).copy()
        if p["vflip"]:
            image = np.flipud(image).copy()
        if p["angle"]:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), p["angle"], 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def _apply_photometric_and_noise(self, image: np.ndarray) -> np.ndarray:
        """The fish-position-preserving ops (safe to apply per-stream)."""
        image = self._apply_turbidity_noise(image)
        image = self._apply_marine_snow(image)
        image = self._apply_motion_blur(image)
        image = self._apply_photometric_jitter(image)
        return image

    def transform_streams(self, crops: dict) -> dict:
        """Augment ALREADY-CROPPED context streams (KNOWN_BUGS #3 — the correct
        order). Geometric transform sampled once and shared across streams;
        photometric + noise per stream. No-op when disabled (val/test)."""
        if not self.enabled:
            return crops
        geo = self._sample_geometric()
        out = {}
        for name, img in crops.items():
            img = self._apply_geometric_params(img, geo)
            img = self._apply_photometric_and_noise(img)
            out[name] = img
        return out

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Legacy single-image path (val/test no-op). For TRAINING use
        transform_streams on the CROPPED streams — applying geometric aug to the
        whole frame before cropping moves the fish out of its bbox (KNOWN_BUGS #3)."""
        if not self.enabled:
            return image
        image = self._apply_geometric(image)
        return self._apply_photometric_and_noise(image)


