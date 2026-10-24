"""
Stage 2 — tracking + hierarchical aggregation, as a callable.

    run_stage2(stage1_out, models, cfg) -> Stage2Output

Builds an SU-T tracker from cfg, feeds it the Stage-1 detections (in memory,
no .npz round-trip), extracts tracklets, and — if the species taxonomy is
available — runs the #5 hierarchical aggregation to attach verdicts. Returns an
in-memory Stage2Output (tracklets [+ verdicts]).

Runs in "no-frames" mode: SU-T needs only boxes, so the offline pipeline tracks
from precomputed detections without re-decoding frames, matching track_stage2's
--no_frames batch path. Every frame from the first to the last detection is fed,
including empty ones, so the motion model sees real time.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from bioreef._3_stage2 import SUTTracker, Tracklet, TrackletWriter, stitch_tracks
from bioreef._9_pipeline.io import Stage1Output, Stage2Output
from bioreef._9_pipeline.aggregation import (
    build_taxonomy_for_aggregation, aggregate_video_verdicts,
)

logger = logging.getLogger("bioreef._9_pipeline.stage2")


from bioreef._9_pipeline.frame_sources import NullFrameSource


def track_single_video(detections, frame_source, tracker, tracklet_writer,
                       video_id: str = "", stitch_max_gap: int = 0):
    """Run the tracker over one clip's detections against any frame source
    (video/dir/null), then (if stitch_max_gap > 0) join broken tracks offline.
    Returns (tracklets, stats). Shared by run_stage2 and the track_stage2 CLI."""
    total_detections = 0
    for frame_idx, frame in frame_source:
        dets = detections.get(frame_idx)
        if dets is None:
            tracker.update(bboxes=np.empty((0, 4)), confidences=np.empty(0),
                           embeddings=None, frame=frame, frame_id=frame_idx)
        else:
            tracker.update(
                bboxes=dets["bboxes"], confidences=dets["confidences"],
                embeddings=dets["embeddings"], frame=frame,
                reid_embeddings=dets.get("reid_embeddings"),
                logits=dets.get("logits"), frame_id=frame_idx,
            )
            total_detections += len(dets["bboxes"])

    all_tracks = tracker.get_all_tracks()
    if stitch_max_gap > 0:
        all_tracks = stitch_tracks(all_tracks, max_gap=stitch_max_gap)
    tracklets = tracklet_writer.extract_tracklets(all_tracks)
    stats = {
        "video_id": video_id,
        "total_detections": total_detections,
        "total_tracks": len(all_tracks),
        "tracklets_exported": len(tracklets),
        "tracklet_lengths": [t.length for t in tracklets],
    }
    return tracklets, stats, all_tracks


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
    """Track one clip's Stage-1 detections + aggregate verdicts ->
    Stage2Output(tracklets, verdicts|None). no-frames mode."""
    detections = _stage1_to_per_frame(stage1_out)
    # every frame in the span, not just those with detections: empty frames still advance the motion model
    frame_ids = list(range(min(detections), max(detections) + 1)) if detections else []

    tracker = SUTTracker(track_thresh=cfg.track_thresh, iou_threshold=cfg.iou_threshold,
                         max_age=cfg.max_age, min_hits=cfg.min_hits)
    writer = TrackletWriter(
        min_length=cfg.min_tracklet_len,
        max_length=cfg.max_tracklet_len,
    )

    # no-frames tracking via the shared general loop
    tracklets, _stats, all_tracks = track_single_video(
        detections, NullFrameSource(frame_ids), tracker, writer,
        video_id=stage1_out.video_id, stitch_max_gap=cfg.stitch_max_gap,
    )
    logger.info("  %s: %d tracklets from tracking",
                stage1_out.video_id, len(tracklets))

    verdicts = _maybe_aggregate(tracklets, models, cfg)

    # One verdict per FISH (track), over every frame it was classified on. The
    # tracklet verdicts above are per 16-30 frame window (Stage 3's input) and skip
    # tracks shorter than 16 classified frames, so they can't be what gets counted.
    tracks = [{"track_id": t.track_id,
               "frames": sorted((int(f), b) for f, b, *_ in t.frame_history)}
              for t in all_tracks]
    track_verdicts = _maybe_aggregate(
        [Tracklet(track_id=t.track_id, frames=[h for h in t.frame_history if h[3] is not None])
         for t in all_tracks], models, cfg)
    return Stage2Output(video_id=stage1_out.video_id, tracklets=tracklets, verdicts=verdicts,
                        tracks=tracks, track_verdicts=track_verdicts)


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
