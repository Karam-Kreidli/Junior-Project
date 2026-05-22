"""
BioReef.ai — Detection (Backend-Agnostic Detector Wrapper)
============================================================
Single source of truth for "give me fish boxes from this frame," regardless
of which detector model is actually running underneath.

Production-path scripts (`infer_stage1.py`, `demo_video.py`,
`eval_pipeline.py`, `run_detector.py`) all import `build_detector()` from
this module and consume a uniform `(xyxy, conf, cls)` tuple — never touch
ultralytics or rfdetr directly.

Supported backends:
    rfdetr  — pretrained Community Fish Detector (RF-DETR Medium/Small/Nano)
              via the `rfdetr` PyPI package. Production default per #6.
    yolo    — legacy Ultralytics YOLO (kept for back-compat with old
              checkpoints and for the YOLO-specific eval scripts).

Returned arrays are numpy, on CPU. The wrapper handles tensor → numpy
conversion so call sites never see torch types leak through.
"""

from .detector import (
    Detector,
    YOLODetector,
    RFDETRDetector,
    build_detector,
    Detections,
)

__all__ = [
    "Detector",
    "YOLODetector",
    "RFDETRDetector",
    "build_detector",
    "Detections",
]
