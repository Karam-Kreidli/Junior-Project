"""
BioReef.ai — Water-Net Gated Fusion Network (Manual Implementation)
====================================================================
Full reimplementation of the WaterNet architecture from:
    Li et al. (2019), "An Underwater Image Enhancement Benchmark Dataset
    and Beyond," IEEE TIP.

Source: https://github.com/tnwei/waternet

This file bypasses torch.hub entirely and manually defines the three
sub-modules (ConfidenceMapGenerator, Refiner, WaterNet) so that we can
load the pretrained state_dict directly without the 'tuple' error.

The preprocessing transforms (White Balance, Gamma Correction, Histogram
Equalization) are also included to provide the 4-input forward pass.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger("bioreef.waternet")

WATERNET_WEIGHTS_URL = (
    "https://www.dropbox.com/s/j8ida1d86hy5tm4/"
    "waternet_exported_state_dict-daa0ee.pt?dl=1"
)

# =============================================================================
# Preprocessing Transforms (from waternet/data.py)
# =============================================================================

def white_balance_transform(im_rgb: np.ndarray) -> np.ndarray:
    """
    Simplest Color Balance algorithm for underwater white balance.
    Requires HWC uint8 RGB input.
    """
    if len(im_rgb.shape) == 3:
        R = np.sum(im_rgb[:, :, 0], axis=None)
        G = np.sum(im_rgb[:, :, 1], axis=None)
        B = np.sum(im_rgb[:, :, 2], axis=None)

        maxpix = max(R, G, B)
        ratio = np.array([maxpix / R, maxpix / G, maxpix / B])

        satLevel1 = 0.005 * ratio
        satLevel2 = 0.005 * ratio

        m, n, p = im_rgb.shape
        im_rgb_flat = np.zeros(shape=(p, m * n))
        for i in range(0, p):
            im_rgb_flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))
    else:
        satLevel1 = np.array([0.001])
        satLevel2 = np.array([0.005])
        m, n = im_rgb.shape
        p = 1
        im_rgb_flat = np.reshape(im_rgb, (1, m * n))

    wb = np.zeros(shape=im_rgb_flat.shape)
    for ch in range(p):
        q = [satLevel1[ch], 1 - satLevel2[ch]]
        tiles = np.quantile(im_rgb_flat[ch, :], q)
        temp = im_rgb_flat[ch, :].copy()
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        if top - bottom > 0:
            wb[ch, :] = (wb[ch, :] - bottom) * 255 / (top - bottom)

    if len(im_rgb.shape) == 3:
        outval = np.zeros(shape=im_rgb.shape)
        for i in range(p):
            outval[:, :, i] = np.reshape(wb[i, :], (m, n))
    else:
        outval = np.reshape(wb, (m, n))

    return outval.astype(np.uint8)


def gamma_correction(im: np.ndarray) -> np.ndarray:
    """Gamma correction with gamma=0.7 (brightens shadows)."""
    gc = np.power(im / 255.0, 0.7)
    gc = np.clip(255 * gc, 0, 255)
    return gc.astype(np.uint8)


def histogram_equalization(im_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel in LAB space."""
    im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    im_lab[:, :, 0] = clahe.apply(im_lab[:, :, 0])
    return cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)


def waternet_preprocess(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the three transformed inputs for WaterNet.
    
    Args:
        rgb_image: HWC uint8 RGB image.
        
    Returns:
        (white_balanced, gamma_corrected, histogram_equalized) — all HWC uint8 RGB.
    """
    wb = white_balance_transform(rgb_image)
    gc = gamma_correction(rgb_image)
    he = histogram_equalization(rgb_image)
    return wb, gc, he


# =============================================================================
# Water-Net Architecture (from waternet/net.py)
# =============================================================================

class ConfidenceMapGenerator(nn.Module):
    """Generates 3 confidence maps for gated fusion of refined inputs."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=7, padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding="same")
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding="same")
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, kernel_size=7, padding="same")
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, padding="same")
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(64, 3, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.sigmoid(self.conv8(out))
        out1, out2, out3 = torch.split(out, [1, 1, 1], dim=1)
        return out1, out2, out3


class Refiner(nn.Module):
    """Refines one transformed input against the original."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=7, padding="same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding="same")
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    Gated Fusion Network for Underwater Image Restoration.
    
    Takes 4 inputs:
        x:  Original RGB image tensor (N, 3, H, W)
        wb: White-balanced version
        ce: Histogram-equalized (contrast enhanced) version
        gc: Gamma-corrected version
        
    Returns:
        Restored image tensor (N, 3, H, W) in [0, 1] range.
    """

    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)
        refined_wb = self.wb_refiner(x, wb)
        refined_ce = self.ce_refiner(x, ce)
        refined_gc = self.gc_refiner(x, gc)
        return (
            torch.mul(refined_wb, wb_cm)
            + torch.mul(refined_ce, ce_cm)
            + torch.mul(refined_gc, gc_cm)
        )


# =============================================================================
# High-Level Restorer Class (Drop-In Replacement for WaterNetRestorer)
# =============================================================================

class WaterNetFullRestorer(nn.Module):
    """
    Full Water-Net restoration pipeline using pretrained weights.
    
    Unlike the fallback OpenCV implementation in data_factory.py, this
    class uses the actual trained Gated Fusion Network to intelligently
    blend white balance, gamma correction, and histogram equalization
    via learned confidence maps.
    
    Usage:
        restorer = WaterNetFullRestorer()
        restored_bgr = restorer(bgr_image)
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = WaterNet()
        self._loaded = False
        self._weights_path = weights_path

    def _ensure_loaded(self):
        """Lazy-load weights on first call."""
        if self._loaded:
            return

        try:
            if self._weights_path:
                state_dict = torch.load(self._weights_path, map_location=self.device)
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    WATERNET_WEIGHTS_URL,
                    progress=True,
                    check_hash=True,
                    map_location=self.device,
                )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("Water-Net weights loaded successfully (Gated Fusion Network).")
        except Exception as e:
            logger.error(f"Failed to load Water-Net weights: {e}")
            self._loaded = True  # Prevent retry loops

    @torch.no_grad()
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Restore a single underwater image using the full Water-Net pipeline.
        
        Args:
            image: BGR uint8 numpy array (H, W, 3).
            
        Returns:
            Restored BGR uint8 numpy array (H, W, 3).
        """
        self._ensure_loaded()

        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate the 3 transformed inputs
        wb, gc, he = waternet_preprocess(rgb)

        # Convert to tensors [0, 1], NCHW
        def to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0)
            return t.to(self.device)

        rgb_t = to_tensor(rgb)
        wb_t = to_tensor(wb)
        he_t = to_tensor(he)
        gc_t = to_tensor(gc)

        # Forward pass through the Gated Fusion Network
        out_t = self.model(rgb_t, wb_t, he_t, gc_t)

        # Convert back to numpy BGR
        out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
