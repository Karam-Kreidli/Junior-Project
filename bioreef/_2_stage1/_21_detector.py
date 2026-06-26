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

logger = logging.getLogger("bioreef._2_stage1")


# =============================================================================
# Canonical output type
# =============================================================================

@dataclass
class Detections:
    """Per-frame result, canonical across backends: xyxy (K,4) float64 pixel
    corners, conf (K,) in [0,1], cls (K,) int64 (0 for single-class fish)."""
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
        """Detect on one image (BGR ndarray or PIL) -> canonical Detections.
        Detections below `conf` are dropped at source."""
        ...


# =============================================================================
# YOLO (Ultralytics)
# =============================================================================

class YOLODetector(Detector):
    """Wraps `ultralytics.YOLO` (legacy / old .pt checkpoints). Ultralytics
    takes BGR ndarrays directly — no colorspace conversion needed."""

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
    """Wraps the `rfdetr` package (production detector, #6). model_size picks
    medium/small (res 1024) or nano (res 640). Handles BGR-ndarray -> PIL.RGB."""

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
    """Build a detector by backend: "rfdetr" (#6 default; weights default to the
    committed CFD medium) or "yolo" (legacy, weights required)."""
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
