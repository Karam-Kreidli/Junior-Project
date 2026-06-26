"""WaterNet spectral restoration (split from data_factory). #14."""
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


# Pretrained weights, in resolution order. The repo-local copy is the
# permanent, version-controlled home; the hub URL is the online fallback.
_WATERNET_REPO_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "weights", "waternet.pt"
)
_WATERNET_WEIGHTS_URL = (
    "https://www.dropbox.com/s/j8ida1d86hy5tm4/"
    "waternet_exported_state_dict-daa0ee.pt?dl=1"
)


# --- WaterNet preprocessing: the 3 transformed inputs ------------------------

def _wn_white_balance(im_rgb: np.ndarray) -> np.ndarray:
    """Simplest Color Balance white balance. HWC uint8 RGB in/out."""
    R = np.sum(im_rgb[:, :, 0])
    G = np.sum(im_rgb[:, :, 1])
    B = np.sum(im_rgb[:, :, 2])
    maxpix = max(R, G, B)
    ratio = np.array([maxpix / R, maxpix / G, maxpix / B])
    satLevel = 0.005 * ratio

    m, n, p = im_rgb.shape
    flat = np.zeros((p, m * n))
    for i in range(p):
        flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))

    wb = np.zeros(flat.shape)
    for ch in range(p):
        q = [satLevel[ch], 1 - satLevel[ch]]
        tiles = np.quantile(flat[ch, :], q)
        temp = flat[ch, :].copy()
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        bottom, top = temp.min(), temp.max()
        wb[ch, :] = (temp - bottom) * 255 / (top - bottom) if top - bottom > 0 else temp

    out = np.zeros(im_rgb.shape)
    for i in range(p):
        out[:, :, i] = np.reshape(wb[i, :], (m, n))
    return out.astype(np.uint8)


def _wn_gamma(im: np.ndarray) -> np.ndarray:
    """Gamma correction (gamma=0.7, brightens shadows)."""
    gc = np.power(im / 255.0, 0.7)
    return np.clip(255 * gc, 0, 255).astype(np.uint8)


def _wn_histeq(im_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --- WaterNet architecture (Li et al. 2019) ---------------------------------

class _WNConfidenceMapGenerator(nn.Module):
    """Generates 3 confidence maps for gated fusion of the refined inputs."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 128, 7, padding="same")
        self.conv2 = nn.Conv2d(128, 128, 5, padding="same")
        self.conv3 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 64, 1, padding="same")
        self.conv5 = nn.Conv2d(64, 64, 7, padding="same")
        self.conv6 = nn.Conv2d(64, 64, 5, padding="same")
        self.conv7 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv8 = nn.Conv2d(64, 3, 3, padding="same")

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4,
                     self.conv5, self.conv6, self.conv7):
            out = F.relu(conv(out))
        out = torch.sigmoid(self.conv8(out))
        return torch.split(out, [1, 1, 1], dim=1)


class _WNRefiner(nn.Module):
    """Refines one transformed input against the original frame."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, 7, padding="same")
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.conv3 = nn.Conv2d(32, 3, 3, padding="same")

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return F.relu(self.conv3(out))


class _WaterNet(nn.Module):
    """Gated Fusion Network. Inputs: raw + white-balance + hist-eq + gamma."""

    def __init__(self):
        super().__init__()
        self.cmg = _WNConfidenceMapGenerator()
        self.wb_refiner = _WNRefiner()
        self.ce_refiner = _WNRefiner()
        self.gc_refiner = _WNRefiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)
        return (
            self.wb_refiner(x, wb) * wb_cm
            + self.ce_refiner(x, ce) * ce_cm
            + self.gc_refiner(x, gc) * gc_cm
        )


class WaterNetRestorer(nn.Module):
    """
    Underwater image restoration using pretrained Water-Net.

    Performs three parallel operations fused via gated network:
        1. White Balancing — removes global blue-green color cast
        2. Gamma Correction — reveals shadow details (reef ledges)
        3. Local Enhancement — sharpens edges for DINOv2 patch extraction

    Reference: Li et al. (2019), "An Underwater Image Enhancement Benchmark
    Dataset and Beyond," IEEE TIP.

    Ecological note:
        Restoring the red channel is critical in the Gulf of Oman where
        selective absorption at depth renders species like Ehrenberg's
        Snapper (Lutjanus ehrenbergii) as grey silhouettes. This module
        recovers the diagnostic yellow lateral stripe and black opercular
        spot essential for fine-grained taxonomic separation.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: Explicit path to a WaterNet state_dict. If None,
                the loader falls back to the repo-local copy, then torch.hub.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self._model: Optional[nn.Module] = None

    def _resolve_weights(self) -> Tuple[Optional[dict], str]:
        """
        Resolve the WaterNet state_dict, offline-first.

        Order: explicit checkpoint_path → repo-local weights/waternet.pt →
        torch.hub download. Returns (state_dict, source_description).
        Raises RuntimeError if every source fails — restoration must never
        silently degrade to passing raw frames through.
        """
        # 1. Explicit path
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return (torch.load(self.checkpoint_path, map_location="cpu"),
                    f"checkpoint_path ({self.checkpoint_path})")

        # 2. Repo-local copy — the permanent, version-controlled home
        repo_path = os.path.abspath(_WATERNET_REPO_WEIGHTS)
        if os.path.exists(repo_path):
            return (torch.load(repo_path, map_location="cpu"),
                    f"repo weights ({repo_path})")

        # 3. Online fallback — download from the published URL
        try:
            sd = torch.hub.load_state_dict_from_url(
                _WATERNET_WEIGHTS_URL, map_location="cpu", progress=True,
            )
            logger.warning(
                "WaterNet weights not found locally; downloaded from hub. "
                "Copy them to %s to enable offline runs.", repo_path,
            )
            return sd, "torch.hub download"
        except Exception as e:
            raise RuntimeError(
                f"WaterNet weights unavailable — no checkpoint_path, no "
                f"repo copy at {repo_path}, and hub download failed ({e}). "
                f"Place the WaterNet state_dict at {repo_path}."
            ) from e

    def _load_model(self):
        """Lazy-load the WaterNet network + weights on first use."""
        if self._model is not None:
            return

        state_dict, source = self._resolve_weights()
        model = _WaterNet()
        model.load_state_dict(state_dict)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model
        logger.info("WaterNet loaded from %s.", source)

    @torch.no_grad()
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Restore a single underwater image with the 4-input WaterNet.

        Args:
            image: BGR uint8 array (H, W, 3).

        Returns:
            Restored BGR uint8 array. On a per-frame numerical failure
            (e.g. a zero-variance solid-colour frame) the raw frame is
            returned — but a *missing model* raises, it never silent-passes.
        """
        self._load_model()
        device = next(self._model.parameters()).device

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # The 3 physically-derived inputs alongside the raw frame.
            wb = _wn_white_balance(rgb)
            gc = _wn_gamma(rgb)
            he = _wn_histeq(rgb)

            def to_tensor(arr):
                t = torch.from_numpy(arr.astype(np.float32) / 255.0)
                return t.permute(2, 0, 1).unsqueeze(0).to(device)

            out = self._model(
                to_tensor(rgb), to_tensor(wb), to_tensor(he), to_tensor(gc)
            )
            restored = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored_rgb = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            # Per-frame numerical breakdown (solid colour → NaNs). Safe to
            # pass the raw frame here — the model itself loaded fine.
            logger.debug(f"WaterNet frame breakdown ({e}); using raw frame.")
            return image

        return cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)


