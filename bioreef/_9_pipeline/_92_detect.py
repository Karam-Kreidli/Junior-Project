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
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from bioreef._2_stage1 import Detector
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._3_stage2 import SUTTracker
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
    logit_prior: Optional[np.ndarray] = None,
    logit_tau: float = 0.0,
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
    dtype = next(head.parameters()).dtype     # bf16 when load_models ran fast_inference
    streams_batched = {
        name: torch.stack(tensors).to(device, dtype)
        for name, tensors in stream_lists.items()
    }

    # The full_frame stream is the same image for every detection: run it once
    # and broadcast, instead of K identical backbone passes (~1/4 of the cost).
    # ViT is per-sample (LayerNorm, in-sample attention), so this is exact.
    if K > 1 and "full_frame" in streams_batched:
        streams_batched["full_frame"] = streams_batched["full_frame"][:1]

    # Run backbone on each stream -> (cls_token, patch_tokens) per stream
    backbone_features = backbone(streams_batched)
    if K > 1 and "full_frame" in backbone_features:
        cls_tok, patch_tok = backbone_features["full_frame"]
        backbone_features["full_frame"] = (cls_tok.expand(K, *cls_tok.shape[1:]),
                                           patch_tok.expand(K, *patch_tok.shape[1:]))

    # Run MCEAM -> fused embedding (256-D) + raw ROI [CLS] (768-D)
    mceam_out = mceam(backbone_features)
    fused = mceam_out["embedding"]                       # (K, 256), on device
    embeddings = fused.float().cpu().numpy()
    reid = mceam_out["roi_cls"].float().cpu().numpy()

    # Run the species head on the fused embedding -> per-class logits.
    # This is Stage 1's per-frame species prior (issue #2).
    logits = head(fused).detach().float().cpu().numpy()  # (K, C)

    # Post-hoc logit adjustment (Menon 2020 / Balanced Softmax): subtract a
    # per-class constant, tau * log(prior), so rare classes are not out-competed
    # by frequent ones. Applied to the RAW logits, before the float16 cast, so
    # Stage-2's tracklet softmax sees the adjusted values. tau=0 or no prior is
    # an exact no-op. This is a decision-rule change only -- the trained weights
    # are untouched, which is why tau stays tunable at inference.
    if logit_prior is not None and logit_tau:
        logits = logits - logit_tau * np.log(logit_prior + 1e-12)

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

    # Classify per TRACK, not per box (cfg.classify_every > 1): a motion-only SU-T
    # pass with Stage 2's settings picks the boxes worth classifying -- each
    # confirmed track when it appears, then every N frames. Unclassified boxes
    # keep NaN embeddings/logits, which Stage 2 treats as "not classified"; its
    # verdicts average over the classified frames. classify_every=1 classifies
    # every box (the old behaviour).
    stride = max(1, int(getattr(cfg, "classify_every", 1)))
    tracker = (SUTTracker(track_thresh=cfg.track_thresh, iou_threshold=cfg.iou_threshold,
                          max_age=cfg.max_age, min_hits=cfg.min_hits) if stride > 1 else None)
    last_classified: Dict[int, int] = {}
    started = False
    D = getattr(models.backbone, "embed_dim", 768)

    for frame_num, frame_bgr in frames:
        if frame_bgr is None:
            continue
        if models.waternet is not None:
            frame_bgr = models.waternet(frame_bgr)

        bboxes, confidences, class_ids = detect_frame(
            models.detector, frame_bgr, cfg.conf_threshold,
        )
        # start at the first frame with fish, like Stage 2, so both passes see the same frames
        started = started or len(bboxes) > 0
        if tracker is not None and started:
            active = tracker.update(bboxes, confidences, None, frame_id=frame_num)
        if len(bboxes) == 0:
            continue

        if tracker is None:
            todo = np.arange(len(bboxes))
        else:
            due = [t for t in active
                   if frame_num - last_classified.get(t.track_id, -10**9) >= stride]
            for t in due:
                last_classified[t.track_id] = frame_num
            # t.bbox is the exact input row the track matched this frame
            todo = np.array([int(np.flatnonzero((bboxes == t.bbox).all(axis=1))[0]) for t in due],
                            dtype=np.int64)

        n_dets = len(bboxes)
        embeddings = np.full((n_dets, 256), np.nan)
        reid = np.full((n_dets, D), np.nan)
        logits = np.full((n_dets, num_classes), np.nan, dtype=np.float16)
        if len(todo):
            embeddings[todo], reid[todo], logits[todo] = extract_embeddings(
                models.backbone, models.mceam, models.head, models.harvester,
                frame_bgr, bboxes[todo], models.device,
                logit_prior=getattr(models, "logit_prior", None),
                logit_tau=getattr(models, "logit_tau", 0.0),
            )

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
