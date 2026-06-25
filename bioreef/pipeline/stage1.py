"""
Stage 1 inference primitives — detection + per-detection embedding extraction.

Extracted from infer_stage1.py so both the inference CLI (infer_stage1) and the
demo overlay (demo_video) import these from the library instead of one script
importing from another. Bodies are verbatim.

    detect_frame(detector, frame_bgr, conf)        -> (bboxes, confs, class_ids)
    extract_embeddings(backbone, mceam, head, ...)  -> (embeddings, reid, logits)
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from bioreef.detection import Detector
from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester


# =============================================================================
# Detection (backend-agnostic — bioreef.detection)
# =============================================================================

def detect_frame(
    detector: Detector,
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run detection on a single frame via the bioreef.detection wrapper.

    Args:
        detector:       Any bioreef.detection.Detector (YOLO or RF-DETR).
        frame_bgr:      Original frame (BGR, full resolution).
        conf_threshold: Minimum detection confidence.

    Returns:
        bboxes:       (K, 4) array of [x, y, w, h] in pixels.
        confidences:  (K,) array of scores.
        class_ids:    (K,) array of predicted class indices.
    """
    dets = detector.predict(frame_bgr, conf=conf_threshold)
    return dets.xywh, dets.conf, dets.cls


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    backbone: ViTBackbone,
    mceam: MCEAM,
    head: nn.Module,
    harvester: ContextHarvester,
    frame_bgr: np.ndarray,
    bboxes: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-detection embeddings + species logits for a frame.

    Returns two embeddings and the classifier logits per fish, each
    serving a distinct downstream role:
      - MCEAM-fused (256-D): habitat-aware z_context for the classifier
        head and the Stage 3 tracklet.
      - DINOv3 ROI [CLS] (768-D): the raw, domain-general backbone token
        used by the Stage 2 tracker for Re-ID association (issue #1).
        MCEAM deliberately collapses same-species individuals, so it is
        the wrong descriptor for individual Re-ID; the frozen DINOv3
        token is not species-collapsed and generalizes cross-domain.
      - Species logits (C-D): the per-frame classifier output (issue #2).
        Stored as Stage 1's species *prior* — Stage 3 / track-level
        aggregation (W4) marginalize it up the taxonomy for the
        genus/family hierarchical fallback. Without it, downstream has
        no probability vector to aggregate or back off on.

    Args:
        backbone:  Frozen ViT backbone.
        mceam:     Trained MCEAM fusion module.
        head:      Trained nn.Linear(256, C) species classifier head.
        harvester: ContextHarvester for 4-stream cropping.
        frame_bgr: Original frame (BGR, full resolution).
        bboxes:    (K, 4) array of [x, y, w, h] in pixels.
        device:    CUDA/CPU device.

    Returns:
        (embeddings, reid, logits) where
          embeddings: (K, 256) MCEAM fused embeddings,
          reid:       (K, D)   raw DINOv3 ROI [CLS] tokens (D = backbone dim),
          logits:     (K, C)   per-species classifier logits.
    """
    K = len(bboxes)
    if K == 0:
        D = getattr(backbone, "embed_dim", 768)
        C = head.out_features
        return (
            np.empty((0, 256), dtype=np.float64),
            np.empty((0, D), dtype=np.float64),
            np.empty((0, C), dtype=np.float16),
        )

    # Harvest 4-stream crops for each detection
    stream_lists = defaultdict(list)

    for bbox in bboxes:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Clamp to valid dimensions
        w = max(w, 1)
        h = max(h, 1)
        crops = harvester.harvest(frame_bgr, (x, y, w, h))
        for stream_name, tensor in crops.items():
            stream_lists[stream_name].append(tensor)

    # Stack into batched tensors: (K, 3, 224, 224)
    streams_batched = {
        name: torch.stack(tensors).to(device)
        for name, tensors in stream_lists.items()
    }

    # Run backbone on each stream -> (cls_token, patch_tokens) per stream
    backbone_features = backbone(streams_batched)

    # Run MCEAM -> fused embedding (256-D) + raw ROI [CLS] (768-D)
    mceam_out = mceam(backbone_features)
    fused = mceam_out["embedding"]                       # (K, 256), on device
    embeddings = fused.float().cpu().numpy()
    reid = mceam_out["roi_cls"].float().cpu().numpy()

    # Run the species head on the fused embedding -> per-class logits.
    # This is Stage 1's per-frame species prior (issue #2).
    logits = head(fused).detach().float().cpu().numpy()  # (K, C)

    return (
        embeddings.astype(np.float64),
        reid.astype(np.float64),
        logits.astype(np.float16),
    )
