"""
Stage 2 — tracking + hierarchical aggregation, as a callable.

    run_stage2(stage1_out, models, cfg) -> Stage2Output

Builds a BoTSORT tracker from cfg, feeds it the Stage-1 detections (in memory,
no .npz round-trip), extracts tracklets, and — if the species taxonomy is
available — runs the #5 hierarchical aggregation to attach verdicts. Returns an
in-memory Stage2Output (tracklets [+ verdicts]).

Runs in "no-frames" mode (CMC disabled): the offline pipeline tracks from
precomputed detections without re-decoding frames, matching track_stage2's
--no_frames batch path. The detection-association logic, tracker params, and
aggregation are unchanged from the script.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from bioreef.tracking import BoTSORTTracker, TrackletWriter
from bioreef.tracking.track import Track
from bioreef.pipeline.io import Stage1Output, Stage2Output
from bioreef.pipeline.aggregation import (
    build_taxonomy_for_aggregation, aggregate_video_verdicts,
)

logger = logging.getLogger("bioreef.pipeline.stage2")


class _NullFrameSource:
    """Yields blank frames for no-frames tracking (CMC disabled)."""
    def __init__(self, frame_ids: List[int]):
        self.frame_ids = list(frame_ids)

    def __iter__(self):
        for fid in self.frame_ids:
            yield fid, np.zeros((1, 1, 3), dtype=np.uint8)


def _stage1_to_per_frame(s1: Stage1Output) -> Dict[int, Dict[str, np.ndarray]]:
    """Group a flat Stage1Output into the per-frame dict the tracker consumes —
    the same structure load_detections_npz builds, but from memory."""
    per_frame: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"bboxes": [], "confidences": [],
                 "embeddings": [], "reid_embeddings": [], "logits": []}
    )
    has_logits = s1.logits is not None and getattr(s1.logits, "size", 0) > 0
    for i in range(len(s1)):
        fid = int(s1.frame_ids[i])
        per_frame[fid]["bboxes"].append(s1.bboxes[i])
        per_frame[fid]["confidences"].append(s1.confidences[i])
        per_frame[fid]["embeddings"].append(s1.embeddings[i])
        per_frame[fid]["reid_embeddings"].append(s1.reid_embeddings[i])
        if has_logits:
            per_frame[fid]["logits"].append(s1.logits[i])

    result = {}
    for fid, arrays in per_frame.items():
        result[fid] = {
            "bboxes": np.array(arrays["bboxes"], dtype=np.float64),
            "confidences": np.array(arrays["confidences"], dtype=np.float64),
            "embeddings": np.array(arrays["embeddings"], dtype=np.float64),
            "reid_embeddings": np.array(arrays["reid_embeddings"],
                                        dtype=np.float64),
        }
        if has_logits:
            result[fid]["logits"] = np.array(arrays["logits"], dtype=np.float32)
    return result


def run_stage2(stage1_out: Stage1Output, models, cfg) -> Stage2Output:
    """
    Track one clip's Stage-1 detections and aggregate verdicts.

    Args:
        stage1_out: Stage1Output (in memory) for one clip.
        models:     Models — used for idx_to_sp (species names for verdicts).
        cfg:        InferenceConfig — tracker + aggregation knobs.

    Returns:
        Stage2Output(tracklets, verdicts|None).
    """
    detections = _stage1_to_per_frame(stage1_out)
    frame_ids = sorted(detections.keys())

    Track.reset_id_counter()
    tracker = BoTSORTTracker(
        high_thresh=cfg.high_thresh,
        low_thresh=cfg.low_thresh,
        max_lost_age=cfg.max_lost_age,
        iou_threshold=cfg.iou_threshold,
        appearance_threshold=cfg.appearance_threshold,
        ema_alpha=cfg.ema_alpha,
        enable_cmc=False,                 # no-frames mode
    )
    writer = TrackletWriter(
        min_length=cfg.min_tracklet_len,
        max_length=cfg.max_tracklet_len,
    )

    src = _NullFrameSource(frame_ids)
    for fid, frame in src:
        dets = detections.get(fid)
        if dets is None:
            tracker.update(bboxes=np.empty((0, 4)), confidences=np.empty(0),
                           embeddings=None, frame=frame)
        else:
            tracker.update(
                bboxes=dets["bboxes"], confidences=dets["confidences"],
                embeddings=dets["embeddings"], frame=frame,
                reid_embeddings=dets.get("reid_embeddings"),
                logits=dets.get("logits"),
            )

    tracklets = writer.extract_tracklets(tracker.get_all_tracks())
    logger.info("  %s: %d tracklets from tracking",
                stage1_out.video_id, len(tracklets))

    verdicts = _maybe_aggregate(tracklets, models, cfg)
    return Stage2Output(video_id=stage1_out.video_id,
                        tracklets=tracklets, verdicts=verdicts)


def _maybe_aggregate(tracklets, models, cfg):
    """Run #5 hierarchical aggregation if the taxonomy can be built from the
    species mapping + CSV. Returns verdict dicts or None (gracefully skipped)."""
    idx_to_sp = getattr(models, "idx_to_sp", None)
    if not idx_to_sp:
        return None
    taxonomy = build_taxonomy_for_aggregation(idx_to_sp, cfg.csv_path)
    if not taxonomy:
        logger.warning("  taxonomy unavailable (csv=%s) — no verdicts",
                       cfg.csv_path)
        return None
    return aggregate_video_verdicts(
        tracklets, taxonomy, idx_to_sp,
        cfg.species_thresh, cfg.genus_thresh, cfg.family_thresh,
    )
