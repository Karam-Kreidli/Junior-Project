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
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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

CLASSIFY_BATCH = 64     # boxes per D6a pass (ViT-H+, bf16: comfortable on the L40S)
CLASSIFY_FRAMES = 16    # frames held in memory at most while waiting for a pass


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
    One frame's boxes; extract_embeddings_multi does several frames in one pass.
    """
    return extract_embeddings_multi(backbone, mceam, head, harvester, [frame_bgr], [bboxes],
                                    device, logit_prior, logit_tau)


@torch.no_grad()
def extract_embeddings_multi(
    backbone: ViTBackbone,
    mceam: MCEAM,
    head: nn.Module,
    harvester: ContextHarvester,
    frames_bgr: List[np.ndarray],
    bboxes_per_frame: List[np.ndarray],
    device: torch.device,
    logit_prior: Optional[np.ndarray] = None,
    logit_tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """extract_embeddings for boxes from several frames in ONE backbone pass, rows in
    frame order. Batching across frames is what keeps the GPU busy: one frame usually
    has only a couple of boxes due for classification."""
    K = sum(len(b) for b in bboxes_per_frame)
    if K == 0:
        D = getattr(backbone, "embed_dim", 768)
        C = head.out_features
        return (
            np.empty((0, 256), dtype=np.float64),
            np.empty((0, D), dtype=np.float64),
            np.empty((0, C), dtype=np.float16),
        )

    # Harvest 4-stream crops for each detection. The full_frame stream is the same
    # image for every box of a frame: keep one per frame and map it back to the
    # boxes after the backbone, instead of identical passes (~1/4 of the cost).
    # ViT is per-sample (LayerNorm, in-sample attention), so this is exact.
    stream_lists = defaultdict(list)
    frame_of_box = []
    for fi, (frame_bgr, bboxes) in enumerate(zip(frames_bgr, bboxes_per_frame)):
        for bi, bbox in enumerate(bboxes):
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Clamp to valid dimensions
            w = max(w, 1)
            h = max(h, 1)
            crops = harvester.harvest(frame_bgr, (x, y, w, h))
            for stream_name, tensor in crops.items():
                if stream_name != "full_frame" or bi == 0:
                    stream_lists[stream_name].append(tensor)
            frame_of_box.append(len(stream_lists["full_frame"]) - 1 if "full_frame" in crops else fi)

    # Stack into batched tensors: (K, 3, 224, 224); full_frame is (frames, 3, 224, 224)
    dtype = next(head.parameters()).dtype     # bf16 when load_models ran fast_inference
    streams_batched = {
        name: torch.stack(tensors).to(device, dtype)
        for name, tensors in stream_lists.items()
    }

    # Run backbone on each stream -> (cls_token, patch_tokens) per stream
    backbone_features = backbone(streams_batched)
    if "full_frame" in backbone_features:
        idx = torch.as_tensor(frame_of_box, device=device)
        cls_tok, patch_tok = backbone_features["full_frame"]
        backbone_features["full_frame"] = (cls_tok[idx], patch_tok[idx])

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
    # seconds per part (read = waiting on the frame iterator: decode + caller's work)
    secs = dict.fromkeys(("read", "detect", "track", "classify"), 0.0)
    n_classified = 0

    # Boxes due for classification wait here and go through D6a together (up to
    # CLASSIFY_BATCH boxes or CLASSIFY_FRAMES frames): the backbone pass is the same
    # per box, but one frame at a time left the GPU mostly idle.
    pending = []          # (index into the all_* lists, rows to fill, frame, boxes)

    def flush():
        nonlocal n_classified
        if not pending:
            return
        t0 = time.perf_counter()
        e, r, lg = extract_embeddings_multi(
            models.backbone, models.mceam, models.head, models.harvester,
            [p[2] for p in pending], [p[3] for p in pending], models.device,
            logit_prior=getattr(models, "logit_prior", None),
            logit_tau=getattr(models, "logit_tau", 0.0),
        )
        k = 0
        for li, rows, _, _ in pending:
            n = len(rows)
            all_embeddings[li][rows], all_reid[li][rows], all_logits[li][rows] = e[k:k + n], r[k:k + n], lg[k:k + n]
            k += n
        n_classified += k
        pending.clear()
        secs["classify"] += time.perf_counter() - t0

    t = time.perf_counter()

    for frame_num, frame_bgr in frames:
        secs["read"] += time.perf_counter() - t
        if frame_bgr is None:
            t = time.perf_counter()
            continue
        if models.waternet is not None:
            frame_bgr = models.waternet(frame_bgr)

        t = time.perf_counter()
        bboxes, confidences, class_ids = detect_frame(
            models.detector, frame_bgr, cfg.conf_threshold,
        )
        secs["detect"] += time.perf_counter() - t
        t = time.perf_counter()
        # start at the first frame with fish, like Stage 2, so both passes see the same frames
        started = started or len(bboxes) > 0
        if tracker is not None and started:
            active = tracker.update(bboxes, confidences, None, frame_id=frame_num)
        secs["track"] += time.perf_counter() - t
        if len(bboxes) == 0:
            t = time.perf_counter()
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
            pending.append((len(all_frame_ids), todo, frame_bgr, bboxes[todo]))

        all_frame_ids.append(np.full(n_dets, frame_num, dtype=np.int64))
        all_bboxes.append(bboxes)
        all_confidences.append(confidences)
        all_embeddings.append(embeddings)
        all_reid.append(reid)
        all_logits.append(logits)
        all_class_ids.append(class_ids)
        if sum(len(p[1]) for p in pending) >= CLASSIFY_BATCH or len(pending) >= CLASSIFY_FRAMES:
            flush()
        t = time.perf_counter()
    flush()

    print("Stage 1 time: " + ", ".join(f"{k} {v:.1f}s" for k, v in secs.items())
          + f"  ({n_classified} of {sum(map(len, all_bboxes))} boxes classified)", flush=True)

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
