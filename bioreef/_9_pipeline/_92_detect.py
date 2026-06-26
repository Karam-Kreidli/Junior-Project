"""
Stage 1 inference — detection + per-detection embedding/logit extraction.

    detect_frame(detector, frame_bgr, conf)        -> (bboxes, confs, class_ids)
    extract_embeddings(backbone, mceam, head, ...)  -> (embeddings, reid, logits)
    run_stage1(frames, models, cfg)                 -> Stage1Output   (one clip)

run_stage1 is the per-clip detection loop lifted from infer_stage1.process_video,
but it returns an in-memory Stage1Output instead of writing a .npz directly, so
the pipeline can pass it straight to Stage 2 (the CLI still .save()s it). The
arrays/dtypes are identical to the old .npz.
"""

import logging
from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

from bioreef._2_stage1 import Detector
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._9_pipeline.io import Stage1Output

logger = logging.getLogger("bioreef._9_pipeline.stage1")


# =============================================================================
# Detection (backend-agnostic — bioreef._2_stage1)
# =============================================================================

def detect_frame(
    detector: Detector,
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect one BGR frame -> (bboxes [x,y,w,h], confidences, class_ids)."""
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
    Per-detection embeddings + logits -> (embeddings, reid, logits). Three
    outputs with distinct roles:
      - embeddings (K,256): MCEAM-fused z_context -> classifier head + Stage 3.
      - reid (K,D): raw DINOv3 ROI [CLS] for Stage-2 Re-ID (#1) — MCEAM collapses
        same-species individuals, so it's wrong for Re-ID; the frozen token isn't.
      - logits (K,C): per-frame species prior (#2) that Stage 3 / W4 aggregation
        marginalizes up the taxonomy for the genus/family fallback.
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


# =============================================================================
# Per-clip Stage 1 run
# =============================================================================

def run_stage1(
    frames: Iterable[Tuple[int, np.ndarray]],
    models,
    cfg,
    video_id: str = "",
) -> Stage1Output:
    """Detect + extract embeddings/logits over one clip's frames (an iterable of
    (frame_id, bgr)) -> Stage1Output (same arrays/dtypes as the old .npz)."""
    num_classes = models.num_classes
    all_frame_ids, all_bboxes, all_confidences = [], [], []
    all_embeddings, all_reid, all_logits, all_class_ids = [], [], [], []

    for frame_num, frame_bgr in frames:
        if frame_bgr is None:
            continue
        if models.waternet is not None:
            frame_bgr = models.waternet(frame_bgr)

        bboxes, confidences, class_ids = detect_frame(
            models.detector, frame_bgr, cfg.conf_threshold,
        )
        if len(bboxes) == 0:
            continue

        embeddings, reid, logits = extract_embeddings(
            models.backbone, models.mceam, models.head, models.harvester,
            frame_bgr, bboxes, models.device,
        )

        n_dets = len(bboxes)
        all_frame_ids.append(np.full(n_dets, frame_num, dtype=np.int64))
        all_bboxes.append(bboxes)
        all_confidences.append(confidences)
        all_embeddings.append(embeddings)
        all_reid.append(reid)
        all_logits.append(logits)
        all_class_ids.append(class_ids)

    if all_frame_ids:
        out = Stage1Output(
            video_id=video_id,
            frame_ids=np.concatenate(all_frame_ids),
            bboxes=np.concatenate(all_bboxes),
            confidences=np.concatenate(all_confidences),
            embeddings=np.concatenate(all_embeddings),
            reid_embeddings=np.concatenate(all_reid),
            logits=np.concatenate(all_logits),
            class_ids=np.concatenate(all_class_ids),
        )
    else:
        D = getattr(models.backbone, "embed_dim", 768)
        out = Stage1Output(
            video_id=video_id,
            frame_ids=np.empty(0, dtype=np.int64),
            bboxes=np.empty((0, 4), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float64),
            embeddings=np.empty((0, 256), dtype=np.float64),
            reid_embeddings=np.empty((0, D), dtype=np.float64),
            logits=np.empty((0, num_classes), dtype=np.float16),
            class_ids=np.empty(0, dtype=np.int64),
        )

    total = len(out)
    logger.info(f"  {video_id}: {total} detections")
    return out
