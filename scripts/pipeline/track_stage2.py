"""
BioReef.ai — Stage 2 Tracking
===============================
Processes a video file (or directory of frames) through the BoTSORT tracker,
producing Spatiotemporal Tracklets for Stage 3 classification.

Pipeline per frame:
    1. Load frame + detections (bboxes, confidences, embeddings)
    2. CMC estimates camera motion between frames
    3. Kalman Filter predicts track positions
    4. Cascaded matching: high-conf IoU -> low-conf rescue -> EMA Re-ID
    5. Hungarian Algorithm solves each cost matrix globally
    6. Tracks accumulate frame histories

After all frames are processed:
    - Tracks with 16-30 matched frames become Tracklets
    - Tracklets are saved (.npz) for Stage 3 input
    - HOTA evaluation runs if ground-truth annotations are provided

Usage:
    # Single video:
    python track_stage2.py --frames_dir path/to/frames/ --detections path/to/dets.npz

    # Batch mode (process all .npz files from infer_stage1.py output):
    python track_stage2.py --no_frames --detections_dir outputs/detections/

    # With HOTA evaluation:
    python track_stage2.py --no_frames --detections_dir outputs/detections/ \
        --gt_tracks path/to/gt_tracks.json

Input format (detections .npz):
    Precomputed Stage 1 output per frame, stored as a NumPy archive:
        - frame_ids:    (N,) array of frame numbers
        - bboxes:       (N, 4) array of [x, y, w, h]
        - confidences:  (N,) array of confidence scores
        - embeddings:   (N, 256) array of MCEAM embeddings
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path. This script lives in scripts/pipeline/, so the
# repo root (where bioreef/ lives) is two levels up.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from bioreef._3_stage2 import BoTSORTTracker, TrackletWriter
from bioreef._3_stage2._32_track import Track
from bioreef._3_stage2._35_tracklet import Tracklet
from bioreef._4_eval._41_hota_evaluator import HOTAEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.stage2")

# Frame filename pattern (matches infer_stage1.py output convention)
FRAME_PATTERN = re.compile(r"^(.+\.avi)\.(\d+)\.png$")


# =============================================================================
# Detection Loading
# =============================================================================

from bioreef._9_pipeline.frame_sources import (  # noqa: E402
    load_detections_npz, load_detections_csv,
    VideoFrameSource, DirectoryFrameSource, NullFrameSource,
    load_gt_tracks, tracklets_to_hota_format,
)


# =============================================================================
# Frame Sources
# =============================================================================

# (frame sources re-exported from bioreef._9_pipeline.frame_sources below)


# =============================================================================
# Hierarchical-fallback aggregation (issue #5)
# =============================================================================

# Aggregation helpers live in the library (bioreef._9_pipeline.aggregation) so
# the in-process run_stage2 and this CLI share one copy. Re-exported for any
# `from track_stage2 import build_taxonomy_for_aggregation` callers.
from bioreef._9_pipeline.aggregation import (  # noqa: E402,F401
    build_taxonomy_for_aggregation,
    aggregate_video_verdicts,
)


# =============================================================================
# Core Tracking Loop
# =============================================================================

# track_single_video moved to bioreef._9_pipeline._93_track (shared with run_stage2).
from bioreef._9_pipeline._93_track import track_single_video  # noqa: E402


# =============================================================================
# HOTA Evaluation Helpers
# =============================================================================

# (HOTA adapters re-exported from bioreef._9_pipeline.frame_sources below)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioReef.ai Stage 2 — BoTSORT Tracking"
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--video", type=str, help="Path to video file.")
    input_group.add_argument("--frames_dir", type=str,
                             help="Directory of frame images.")
    input_group.add_argument("--no_frames", action="store_true",
                             help="Run without frame images (CMC disabled). "
                                  "Useful when frames are not available locally.")

    # Detection source
    det_group = parser.add_mutually_exclusive_group(required=True)
    det_group.add_argument("--detections", type=str,
                           help="Path to a single detections .npz file.")
    det_group.add_argument("--detections_dir", type=str,
                           help="Directory of .npz files (batch mode — "
                                "one file per video from infer_stage1.py).")
    det_group.add_argument("--from_csv", type=str,
                           help="Generate mock detections from CSV "
                                "(for testing before detection head exists).")

    # Tracker parameters
    parser.add_argument("--high_thresh", type=float, default=0.6)
    parser.add_argument("--low_thresh", type=float, default=0.1)
    parser.add_argument("--max_lost_age", type=int, default=30)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--appearance_threshold", type=float, default=0.4)
    parser.add_argument("--ema_alpha", type=float, default=0.9)
    parser.add_argument("--no_cmc", action="store_true",
                         help="Disable Camera Motion Compensation.")

    # Tracklet parameters
    parser.add_argument("--min_tracklet_len", type=int, default=16)
    parser.add_argument("--max_tracklet_len", type=int, default=30)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/tracklets")

    # Hierarchical-fallback aggregation (issue #5)
    parser.add_argument("--csv_path", type=str,
                         default="data_oz/metadata/frame_metadata_subset.csv",
                         help="Metadata CSV (species/genus/family) for the "
                              "taxonomy used by hierarchical aggregation. "
                              "Defaults to the recovered 256-class subset "
                              "matching bioreef_stage1.pt (#24). The full "
                              "307-species frame_metadata.csv mis-sizes the "
                              "taxonomy maps and crashes aggregation against "
                              "the 256-class head.")
    parser.add_argument("--species_thresh", type=float, default=0.50,
                         help="Min aggregated prob to commit to a species.")
    parser.add_argument("--genus_thresh", type=float, default=0.60,
                         help="Min aggregated prob to commit to a genus.")
    parser.add_argument("--family_thresh", type=float, default=0.70,
                         help="Min aggregated prob to commit to a family.")

    # For CSV mode
    parser.add_argument("--img_dir", type=str, default="",
                         help="Image directory (required with --from_csv "
                              "unless --no_frames is set).")

    # HOTA evaluation
    parser.add_argument("--gt_tracks", type=str, default=None,
                         help="Path to ground-truth tracking JSON for HOTA "
                              "evaluation. Format: {video_id: {frame_id: "
                              "[{track_id, bbox}]}}.")

    args = parser.parse_args()

    # Validate: must specify a frame source unless --no_frames
    if not args.no_frames and args.video is None and args.frames_dir is None:
        if not args.detections_dir:
            parser.error(
                "One of --video, --frames_dir, or --no_frames is required."
            )
        else:
            # Batch mode without frames defaults to no_frames
            args.no_frames = True
            args.no_cmc = True

    # Force no CMC in no_frames mode
    if args.no_frames:
        args.no_cmc = True

    # =========================================================================
    # Tracker config (shared across batch runs)
    # =========================================================================
    def make_tracker():
        Track.reset_id_counter()
        return BoTSORTTracker(
            high_thresh=args.high_thresh,
            low_thresh=args.low_thresh,
            max_lost_age=args.max_lost_age,
            iou_threshold=args.iou_threshold,
            appearance_threshold=args.appearance_threshold,
            ema_alpha=args.ema_alpha,
            enable_cmc=not args.no_cmc,
        )

    tracklet_writer = TrackletWriter(
        min_length=args.min_tracklet_len,
        max_length=args.max_tracklet_len,
        output_dir=args.output_dir,
    )

    # ---- Hierarchical-fallback aggregation setup (issue #5) ----------------
    # idx_to_sp comes from species_mapping.npz (written by infer_stage1.py
    # alongside the detection archives); taxonomy maps come from the CSV.
    # If either is unavailable, aggregation is skipped — tracking still runs.
    idx_to_sp: Dict[int, str] = {}
    mapping_npz = None
    if args.detections_dir:
        cand = os.path.join(args.detections_dir, "species_mapping.npz")
        if os.path.exists(cand):
            mapping_npz = cand
    elif args.detections:
        cand = os.path.join(os.path.dirname(args.detections),
                            "species_mapping.npz")
        if os.path.exists(cand):
            mapping_npz = cand
    if mapping_npz:
        m = np.load(mapping_npz, allow_pickle=True)
        idx_to_sp = {int(k): v for k, v in m["idx_to_sp"].item().items()}
        logger.info(f"Loaded species mapping: {len(idx_to_sp)} species")

    taxonomy = build_taxonomy_for_aggregation(idx_to_sp, args.csv_path)
    if taxonomy:
        logger.info(
            f"Hierarchical aggregation enabled — "
            f"{taxonomy['num_genera']} genera, {taxonomy['num_families']} "
            f"families (thresholds sp={args.species_thresh} "
            f"ge={args.genus_thresh} fa={args.family_thresh})"
        )
    else:
        logger.warning(
            "Hierarchical aggregation disabled — species mapping or CSV "
            "taxonomy unavailable. Tracklets are still produced."
        )

    # Load GT tracks for HOTA evaluation (if provided)
    gt_data = None
    if args.gt_tracks:
        gt_data = load_gt_tracks(args.gt_tracks)
        logger.info(f"Loaded GT tracks for {len(gt_data)} videos")

    hota_evaluator = HOTAEvaluator(output_dir=args.output_dir)

    # =========================================================================
    # BATCH MODE: process all .npz files in a directory
    # =========================================================================
    if args.detections_dir:
        npz_files = sorted(glob.glob(os.path.join(args.detections_dir, "*.npz")))
        # Exclude species_mapping.npz from infer_stage1.py
        npz_files = [f for f in npz_files
                     if os.path.basename(f) != "species_mapping.npz"]

        if not npz_files:
            logger.error(f"No .npz files found in: {args.detections_dir}")
            return

        logger.info("=" * 60)
        logger.info("BioReef.ai — Stage 2 Tracking (Batch Mode)")
        logger.info(f"  Detection dir  : {args.detections_dir}")
        logger.info(f"  Videos         : {len(npz_files)}")
        logger.info(f"  CMC            : {'enabled' if not args.no_cmc else 'disabled'}")
        logger.info(f"  Tracklet range : [{args.min_tracklet_len}, {args.max_tracklet_len}]")
        logger.info(f"  Output dir     : {args.output_dir}")
        logger.info("=" * 60)

        grand_total_dets = 0
        grand_total_tracklets = 0
        all_video_tracklets: Dict[str, List[Tracklet]] = {}

        for npz_path in npz_files:
            video_id = os.path.splitext(os.path.basename(npz_path))[0]
            logger.info(f"Tracking: {video_id}")

            detections = load_detections_npz(npz_path)
            if not detections:
                logger.warning(f"  No detections in {npz_path}, skipping.")
                continue

            # Build frame source from detection frame IDs
            sorted_frame_ids = sorted(detections.keys())
            frame_source = NullFrameSource(sorted_frame_ids)

            tracker = make_tracker()
            tracklets, stats = track_single_video(
                detections, frame_source, tracker, tracklet_writer, video_id,
            )

            # Save per-video tracklets
            if tracklets:
                tracklet_writer.save(tracklets, filename=f"{video_id}.npz")

                # Hierarchical-fallback species verdicts (issue #5)
                if taxonomy:
                    verdicts = aggregate_video_verdicts(
                        tracklets, taxonomy, idx_to_sp,
                        args.species_thresh, args.genus_thresh,
                        args.family_thresh,
                    )
                    vpath = os.path.join(args.output_dir,
                                         f"{video_id}_verdicts.json")
                    with open(vpath, "w", encoding="utf-8") as f:
                        json.dump(verdicts, f, indent=2)
                    levels = Counter(v["level"] for v in verdicts)
                    logger.info(
                        f"  Verdicts: {dict(levels)} -> {vpath}"
                    )

            all_video_tracklets[video_id] = tracklets
            grand_total_dets += stats["total_detections"]
            grand_total_tracklets += stats["tracklets_exported"]

            logger.info(
                f"  {video_id}: {stats['total_detections']} dets, "
                f"{stats['total_tracks']} tracks, "
                f"{stats['tracklets_exported']} tracklets"
            )

            # HOTA evaluation for this video (if GT available)
            if gt_data and video_id in gt_data:
                pred_hota = tracklets_to_hota_format(tracklets)
                hota_evaluator.evaluate_sequence(
                    video_id, gt_data[video_id], pred_hota,
                )

        # Batch summary
        logger.info("=" * 60)
        logger.info("Stage 2 Batch Summary")
        logger.info("=" * 60)
        logger.info(f"  Videos processed       : {len(npz_files)}")
        logger.info(f"  Total detections       : {grand_total_dets}")
        logger.info(f"  Total tracklets        : {grand_total_tracklets}")
        logger.info(f"  Output directory       : {args.output_dir}")
        logger.info("=" * 60)

        # Aggregate HOTA results
        if gt_data:
            agg = hota_evaluator.aggregate()
            hota_evaluator.save_results()
            logger.info(f"  HOTA (aggregate)       : {agg.get('HOTA_mean', 0):.4f}")
            logger.info(f"  DetA (aggregate)       : {agg.get('DetA_mean', 0):.4f}")
            logger.info(f"  AssA (aggregate)       : {agg.get('AssA_mean', 0):.4f}")

        return

    # =========================================================================
    # SINGLE VIDEO MODE (existing behavior)
    # =========================================================================
    frame_paths = None

    if args.detections:
        logger.info(f"Loading detections from: {args.detections}")
        detections = load_detections_npz(args.detections)
    else:
        if not args.no_frames and not args.img_dir:
            parser.error(
                "--img_dir is required when using --from_csv without --no_frames"
            )
        img_dir = args.img_dir if args.img_dir else ""
        logger.info(f"Generating detections from CSV: {args.from_csv}")
        detections, frame_paths = load_detections_csv(args.from_csv, img_dir)

    # Set up frame source
    if args.no_frames:
        sorted_frame_ids = sorted(detections.keys()) if detections else []
        frame_source = NullFrameSource(sorted_frame_ids)
        logger.info(
            f"No-frames mode: {len(sorted_frame_ids)} frames, CMC disabled."
        )
    elif args.video:
        frame_source = VideoFrameSource(args.video)
        logger.info(
            f"Video: {args.video} | "
            f"{frame_source.total_frames} frames @ {frame_source.fps:.1f} fps"
        )
    elif frame_paths is not None:
        frame_source = DirectoryFrameSource(args.frames_dir or "", paths=frame_paths)
        logger.info(f"Frames from CSV: {len(frame_source)} frames")
    else:
        frame_source = DirectoryFrameSource(args.frames_dir)
        logger.info(f"Frames dir: {args.frames_dir} | {len(frame_source)} frames")

    # Run tracking
    tracker = make_tracker()
    tracklets, stats = track_single_video(
        detections, frame_source, tracker, tracklet_writer,
        video_id="single",
    )

    if tracklets:
        npz_path = tracklet_writer.save(tracklets)
        logger.info(f"Tracklets saved to: {npz_path}")

        # Hierarchical-fallback species verdicts (issue #5)
        if taxonomy:
            verdicts = aggregate_video_verdicts(
                tracklets, taxonomy, idx_to_sp,
                args.species_thresh, args.genus_thresh, args.family_thresh,
            )
            vpath = os.path.join(args.output_dir, "verdicts.json")
            with open(vpath, "w", encoding="utf-8") as f:
                json.dump(verdicts, f, indent=2)
            levels = Counter(v["level"] for v in verdicts)
            logger.info(f"Verdicts: {dict(levels)} -> {vpath}")
    else:
        logger.warning("No tracklets met the minimum length requirement.")

    # HOTA evaluation (single video)
    if gt_data:
        video_id = list(gt_data.keys())[0] if len(gt_data) == 1 else "single"
        if video_id in gt_data:
            pred_hota = tracklets_to_hota_format(tracklets)
            result = hota_evaluator.evaluate_sequence(
                video_id, gt_data[video_id], pred_hota,
            )
            hota_evaluator.save_results()

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 2 Tracking Summary")
    logger.info("=" * 60)
    logger.info(f"  Total detections processed : {stats['total_detections']}")
    logger.info(f"  Total tracks created       : {stats['total_tracks']}")
    logger.info(f"  Tracklets exported         : {stats['tracklets_exported']}")
    logger.info(
        f"  Tracklet lengths           : "
        f"{stats['tracklet_lengths'][:10]}"
        f"{'...' if len(stats['tracklet_lengths']) > 10 else ''}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
