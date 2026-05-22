"""
Detector ABC + concrete YOLO and RF-DETR implementations.

The wrapper is intentionally narrow: it returns numpy arrays in a single
canonical format so call sites don't care which model is running. Any
backend-specific quirks (Ultralytics' BGR-vs-RGB handling, RF-DETR's PIL
input, tensor → numpy conversion, device placement) are hidden here.

Public API:
    Detections          dataclass: xyxy (K,4) float64, conf (K,) float64,
                        cls (K,) int64. Empty if no detections.
    Detector            abstract base — implementations supply .predict().
    YOLODetector        wraps ultralytics.YOLO
    RFDETRDetector      wraps rfdetr.RFDETR{Medium,Small,Nano}
    build_detector()    factory dispatched by string `backend`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("bioreef.detection")


# =============================================================================
# Canonical output type
# =============================================================================

@dataclass
class Detections:
    """
    Per-frame detection result, canonicalised across backends.

    Fields:
        xyxy:  (K, 4) float64 — corner coords in pixel space, in the
               source image's coordinate system (no rescale).
        conf:  (K,) float64   — detection scores in [0, 1].
        cls:   (K,) int64     — 0-indexed class IDs. For single-class
                                detectors (fish-only) every entry is 0.
    """
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float64),
            conf=np.empty((0,), dtype=np.float64),
            cls=np.empty((0,), dtype=np.int64),
        )

    @property
    def xywh(self) -> np.ndarray:
        """(K, 4) in COCO [x, y, w, h] convention."""
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.float64)
        out = self.xyxy.copy()
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return out


# Type alias for "thing you can feed predict()". We accept BGR ndarray
# (OpenCV's default) or a PIL.Image; each backend handles the conversion.
ImageInput = Union[np.ndarray, "PIL.Image.Image"]  # noqa: F821


# =============================================================================
# Abstract base
# =============================================================================

class Detector(ABC):
    """Abstract detector. Subclasses wrap a concrete backend."""

    #: Display name — used in logs only.
    backend: str = "abstract"

    @abstractmethod
    def predict(self, image: ImageInput, conf: float = 0.05) -> Detections:
        """
        Run detection on one image. Returns canonical Detections.

        Args:
            image: BGR uint8 ndarray (H, W, 3) or PIL.Image (any mode).
            conf:  Confidence threshold. Detections below this score are
                   dropped at source — so the caller never has to filter.
        """
        ...


# =============================================================================
# YOLO (Ultralytics)
# =============================================================================

class YOLODetector(Detector):
    """
    Wrapper around `ultralytics.YOLO`. Used by legacy scripts and as a
    fallback for old `.pt` checkpoints.

    Note: ultralytics accepts BGR ndarrays directly, so no colorspace
    conversion is needed here.
    """

    backend = "yolo"

    def __init__(
        self,
        weights: str,
        imgsz: int = 960,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO  # local import — keeps import cheap
        logger.info("loading YOLO from %s (imgsz=%d)", weights, imgsz)
        self._model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device  # None = let ultralytics decide
        self.names = list(self._model.names.values())

    def predict(self, image: ImageInput, conf: float = 0.05) -> Detections:
        res = self._model.predict(
            image,
            conf=conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        cnf = boxes.conf.cpu().numpy().astype(np.float64)
        cls = boxes.cls.cpu().numpy().astype(np.int64)
        return Detections(xyxy=xyxy, conf=cnf, cls=cls)


# =============================================================================
# RF-DETR (Community Fish Detector, the production detector — issue #6)
# =============================================================================

class RFDETRDetector(Detector):
    """
    Wrapper around the `rfdetr` PyPI package.

    The CFD weights ship in three sizes. Pass `model_size` to pick:
        "medium" → RFDETRMedium (best AP, default), resolution 1024.
        "small"  → RFDETRSmall, resolution 1024.
        "nano"   → RFDETRNano, resolution 640.

    The package consumes PIL images; this wrapper handles BGR-ndarray →
    PIL conversion so call sites don't need to.
    """

    backend = "rfdetr"

    def __init__(
        self,
        weights: str,
        model_size: str = "medium",
        resolution: Optional[int] = None,
        device: Optional[str] = None,
    ):
        from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRNano
        sizes = {"medium": RFDETRMedium, "small": RFDETRSmall, "nano": RFDETRNano}
        if model_size not in sizes:
            raise ValueError(f"model_size must be one of {sorted(sizes)}; got {model_size!r}")
        cls_obj = sizes[model_size]
        # Nano defaults to 640, the others to 1024 — match if not overridden.
        if resolution is None:
            resolution = 640 if model_size == "nano" else 1024
        logger.info("loading %s from %s (resolution=%d)",
                    cls_obj.__name__, weights, resolution)
        self._model = cls_obj(pretrain_weights=weights, resolution=resolution)
        self.resolution = resolution
        self.model_size = model_size
        self.device = device
        # CFD is single-class.
        self.names = ["fish"]

    def predict(self, image: ImageInput, conf: float = 0.05) -> Detections:
        # Accept BGR ndarray or PIL; normalize to PIL.RGB for rfdetr.
        if isinstance(image, np.ndarray):
            from PIL import Image
            # Assume BGR (OpenCV convention) — swap to RGB.
            rgb = image[:, :, ::-1] if image.ndim == 3 else image
            pil = Image.fromarray(rgb)
        else:
            pil = image.convert("RGB") if hasattr(image, "convert") else image

        dets = self._model.predict(pil, threshold=conf)
        n = len(dets.xyxy) if hasattr(dets, "xyxy") else 0
        if n == 0:
            return Detections.empty()

        xyxy = np.asarray(dets.xyxy, dtype=np.float64).reshape(-1, 4)
        cnf = (np.asarray(dets.confidence, dtype=np.float64)
               if dets.confidence is not None else np.ones(n, dtype=np.float64))
        cls = (np.asarray(dets.class_id, dtype=np.int64)
               if dets.class_id is not None else np.zeros(n, dtype=np.int64))
        return Detections(xyxy=xyxy, conf=cnf, cls=cls)


# =============================================================================
# Factory
# =============================================================================

# Where the production RF-DETR weights live in the repo (committed file).
DEFAULT_RFDETR_WEIGHTS = "weights/rfdetr_medium_cfd.pth"


def build_detector(
    backend: str,
    weights: Optional[str] = None,
    *,
    model_size: str = "medium",
    resolution: Optional[int] = None,
    imgsz: int = 960,
    device: Optional[str] = None,
) -> Detector:
    """
    Construct a detector by name.

    Args:
        backend: "rfdetr" (production default per #6) or "yolo" (legacy).
        weights: Checkpoint path. If None and backend is "rfdetr", falls
                 back to the repo's committed RF-DETR Medium weights
                 (DEFAULT_RFDETR_WEIGHTS). For "yolo" the path is required.
        model_size:  RF-DETR variant ("medium" / "small" / "nano"). Ignored for yolo.
        resolution:  RF-DETR inference resolution. Ignored for yolo.
        imgsz:       YOLO inference imgsz. Ignored for rfdetr.
        device:      "cuda" / "cpu" / None (let the backend decide).

    Returns:
        A concrete Detector instance.
    """
    backend = backend.lower()
    if backend == "rfdetr":
        if weights is None:
            weights = DEFAULT_RFDETR_WEIGHTS
        return RFDETRDetector(
            weights=weights, model_size=model_size,
            resolution=resolution, device=device,
        )
    if backend == "yolo":
        if weights is None:
            raise ValueError("yolo backend requires --weights (no built-in default)")
        return YOLODetector(weights=weights, imgsz=imgsz, device=device)
    raise ValueError(f"unknown backend: {backend!r} — choose 'rfdetr' or 'yolo'")
