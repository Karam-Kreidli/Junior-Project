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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioreef.tracking import BoTSORTTracker, TrackletWriter
from bioreef.tracking.track import Track
from bioreef.tracking.tracklet import Tracklet
from bioreef.evaluation.hota_evaluator import HOTAEvaluator

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

def load_detections_npz(
    path: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed detections from a .npz archive.

    Returns:
        Dict mapping frame_id -> {
            'bboxes':      (K, 4) array,
            'confidences': (K,) array,
            'embeddings':  (K, 256) array,
        }
    """
    data = np.load(path, allow_pickle=True)
    frame_ids = data["frame_ids"]
    bboxes = data["bboxes"]
    confidences = data["confidences"]
    embeddings = data["embeddings"]

    per_frame: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"bboxes": [], "confidences": [], "embeddings": []}
    )

    for i in range(len(frame_ids)):
        fid = int(frame_ids[i])
        per_frame[fid]["bboxes"].append(bboxes[i])
        per_frame[fid]["confidences"].append(confidences[i])
        per_frame[fid]["embeddings"].append(embeddings[i])

    result = {}
    for fid, arrays in per_frame.items():
        result[fid] = {
            "bboxes": np.array(arrays["bboxes"], dtype=np.float64),
            "confidences": np.array(arrays["confidences"], dtype=np.float64),
            "embeddings": np.array(arrays["embeddings"], dtype=np.float64),
        }

    return result


def load_detections_csv(
    csv_path: str,
    img_dir: str,
    embedding_dim: int = 256,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[str]]:
    """
    Create mock detections from the frame_metadata CSV (for testing Stage 2
    before the detection head exists). Embeddings are zero-vectors since
    we don't have real Stage 1 output yet.

    Returns:
        (detections_dict, frame_paths): Per-frame detections and ordered
        list of frame file paths.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required_cols = {"file_name", "x0", "y0", "x1", "y1"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. "
            f"Found: {set(df.columns)}"
        )

    # Group by file_name to get per-frame detections
    grouped = df.groupby("file_name")
    frame_names = sorted(grouped.groups.keys())

    detections = {}
    frame_paths = []

    for frame_idx, fname in enumerate(frame_names):
        group = grouped.get_group(fname)
        bboxes = []

        for _, row in group.iterrows():
            x0, y0, x1, y1 = (
                int(row["x0"]), int(row["y0"]),
                int(row["x1"]), int(row["y1"]),
            )
            w = x1 - x0
            h = y1 - y0
            bboxes.append([x0, y0, w, h])

        n = len(bboxes)
        detections[frame_idx] = {
            "bboxes": np.array(bboxes, dtype=np.float64),
            "confidences": np.ones(n, dtype=np.float64),
            "embeddings": np.zeros((n, embedding_dim), dtype=np.float64),
        }

        frame_paths.append(os.path.join(img_dir, fname))

    return detections, frame_paths


# =============================================================================
# Frame Sources
# =============================================================================

class VideoFrameSource:
    """Read frames from a video file."""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        idx = self._frame_idx
        self._frame_idx += 1
        return idx, frame

    def __len__(self):
        return self.total_frames

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


class DirectoryFrameSource:
    """Read frames from a directory of images (sorted by filename)."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, frames_dir: str, paths: Optional[List[str]] = None):
        if paths is not None:
            self.paths = paths
        else:
            self.paths = sorted(
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if os.path.splitext(f)[1].lower() in self.EXTENSIONS
            )
        self.total_frames = len(self.paths)
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        if self._idx >= len(self.paths):
            raise StopIteration
        path = self.paths[self._idx]
        frame = cv2.imread(path)
        if frame is None:
            logger.warning(f"Could not read frame: {path}")
            self._idx += 1
            return self.__next__()
        idx = self._idx
        self._idx += 1
        return idx, frame

    def __len__(self):
        return self.total_frames


class NullFrameSource:
    """
    Synthetic frame source for --no_frames mode.

    Yields blank (1x1) frames so the tracker can run without actual images.
    CMC should be disabled when using this source.
    """

    def __init__(self, frame_ids: List[int]):
        """
        Args:
            frame_ids: Sorted list of frame IDs to iterate over.
        """
        self.frame_ids = frame_ids
        self.total_frames = len(frame_ids)
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        if self._idx >= self.total_frames:
            raise StopIteration
        fid = self.frame_ids[self._idx]
        self._idx += 1
        return fid, np.zeros((1, 1, 3), dtype=np.uint8)

    def __len__(self):
        return self.total_frames


# =============================================================================
# Core Tracking Loop
# =============================================================================

def track_single_video(
    detections: Dict[int, Dict[str, np.ndarray]],
    frame_source,
    tracker: BoTSORTTracker,
    tracklet_writer: TrackletWriter,
    video_id: str = "",
) -> Tuple[List[Tracklet], Dict]:
    """
    Run the tracking cascade on one video's detections.

    Args:
        detections:      Per-frame detections from infer_stage1.py.
        frame_source:    Iterator yielding (frame_id, frame_array).
        tracker:         Fresh BoTSORTTracker instance.
        tracklet_writer: TrackletWriter for extracting tracklets.
        video_id:        Identifier for logging.

    Returns:
        (tracklets, stats): List of Tracklet objects and summary dict.
    """
    total_detections = 0

    for frame_idx, frame in frame_source:
        dets = detections.get(frame_idx)

        if dets is None:
            tracker.update(
                bboxes=np.empty((0, 4)),
                confidences=np.empty(0),
                embeddings=None,
                frame=frame,
            )
        else:
            tracker.update(
                bboxes=dets["bboxes"],
                confidences=dets["confidences"],
                embeddings=dets["embeddings"],
                frame=frame,
            )
            total_detections += len(dets["bboxes"])

        if frame_idx % 100 == 0:
            n_active = len(tracker.active_tracks)
            n_lost = len(tracker.lost_tracks)
            logger.debug(
                f"[{video_id}] Frame {frame_idx:5d} | "
                f"Active: {n_active:3d} | Lost: {n_lost:3d}"
            )

    all_tracks = tracker.get_all_tracks()
    tracklets = tracklet_writer.extract_tracklets(all_tracks)

    stats = {
        "video_id": video_id,
        "total_detections": total_detections,
        "total_tracks": len(all_tracks),
        "tracklets_exported": len(tracklets),
        "tracklet_lengths": [t.length for t in tracklets],
    }

    return tracklets, stats


# =============================================================================
# HOTA Evaluation Helpers
# =============================================================================

def load_gt_tracks(gt_path: str) -> Dict[str, Dict]:
    """
    Load ground-truth tracking annotations.

    Expected JSON format:
        {
            "video_id": {
                "frame_id": [
                    {"track_id": int, "bbox": [x, y, w, h]},
                    ...
                ],
                ...
            },
            ...
        }

    Args:
        gt_path: Path to ground-truth JSON file.

    Returns:
        Dict mapping video_id -> {frame_id: [annotations]}.
    """
    with open(gt_path) as f:
        raw = json.load(f)

    # Convert string frame_id keys to int
    gt_data = {}
    for video_id, frames in raw.items():
        gt_data[video_id] = {
            int(fid): anns for fid, anns in frames.items()
        }

    return gt_data


def tracklets_to_hota_format(
    tracklets: List[Tracklet],
) -> Dict[int, List[Dict]]:
    """
    Convert tracklets to the per-frame format HOTA expects.

    Returns:
        {frame_id: [{'track_id': int, 'bbox': [x,y,w,h]}, ...]}
    """
    result: Dict[int, List[Dict]] = defaultdict(list)
    for tracklet in tracklets:
        for frame_id, bbox, _embedding in tracklet.frames:
            result[frame_id].append({
                "track_id": tracklet.track_id,
                "bbox": bbox.tolist(),
            })
    return dict(result)


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
