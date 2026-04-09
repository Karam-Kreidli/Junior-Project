"""
BioReef.ai — Stage 2 Tracking
===============================
Processes a video file (or directory of frames) through the BoTSORT tracker,
producing Spatiotemporal Tracklets for Stage 3 classification.

Pipeline per frame:
    1. Load frame + detections (bboxes, confidences, embeddings)
    2. CMC estimates camera motion between frames
    3. Kalman Filter predicts track positions
    4. Cascaded matching: high-conf IoU → low-conf rescue → EMA Re-ID
    5. Hungarian Algorithm solves each cost matrix globally
    6. Tracks accumulate frame histories

After all frames are processed:
    - Tracks with 16–30 matched frames become Tracklets
    - Tracklets are saved (.npz) for Stage 3 input
    - HOTA evaluation runs if ground-truth annotations are provided

Usage:
    python track_stage2.py --video path/to/video.mp4 --detections path/to/dets.npz
    python track_stage2.py --frames_dir path/to/frames/ --detections path/to/dets.npz

    # No-frames mode (CMC disabled, useful when frame images are unavailable):
    python track_stage2.py --no_frames --from_csv frame_metadata.csv

Input format (detections .npz):
    Precomputed Stage 1 output per frame, stored as a NumPy archive:
        - frame_ids:    (N,) array of frame numbers
        - bboxes:       (N, 4) array of [x, y, w, h]
        - confidences:  (N,) array of confidence scores
        - embeddings:   (N, 256) array of MCEAM embeddings

    Once the detection head is built into Stage 1, this file will be
    generated automatically. For now, it can be created from CSV annotations
    using the --from_csv flag.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioreef.tracking import BoTSORTTracker, TrackletWriter
from bioreef.evaluation.hota_evaluator import HOTAEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.stage2")


# =============================================================================
# Detection Loading
# =============================================================================

def load_detections_npz(
    path: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed detections from a .npz archive.

    Returns:
        Dict mapping frame_id → {
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

    Yields blank (1×1) frames so the tracker can run without actual images.
    CMC should be disabled when using this source.
    """

    def __init__(self, num_frames: int):
        self.total_frames = num_frames
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        if self._idx >= self.total_frames:
            raise StopIteration
        idx = self._idx
        self._idx += 1
        # 1×1 blank frame — CMC will find no features and return None (safe)
        return idx, np.zeros((1, 1, 3), dtype=np.uint8)

    def __len__(self):
        return self.total_frames


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
                           help="Path to precomputed detections (.npz).")
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

    args = parser.parse_args()

    # Validate: must specify a frame source unless --no_frames
    if not args.no_frames and args.video is None and args.frames_dir is None:
        parser.error(
            "One of --video, --frames_dir, or --no_frames is required."
        )

    # =========================================================================
    # Load detections
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

    # =========================================================================
    # Set up frame source
    # =========================================================================
    if args.no_frames:
        # Headless mode: blank frames, CMC automatically disabled
        num_frames = max(detections.keys()) + 1 if detections else 0
        frame_source = NullFrameSource(num_frames)
        args.no_cmc = True
        logger.info(
            f"No-frames mode: {num_frames} synthetic frames, CMC disabled."
        )
    elif args.video:
        frame_source = VideoFrameSource(args.video)
        logger.info(
            f"Video: {args.video} | "
            f"{frame_source.total_frames} frames @ {frame_source.fps:.1f} fps"
        )
    elif frame_paths is not None:
        # CSV mode: use the frame paths extracted from the CSV
        frame_source = DirectoryFrameSource(args.frames_dir or "", paths=frame_paths)
        logger.info(f"Frames from CSV: {len(frame_source)} frames")
    else:
        frame_source = DirectoryFrameSource(args.frames_dir)
        logger.info(f"Frames dir: {args.frames_dir} | {len(frame_source)} frames")

    # =========================================================================
    # Initialize tracker
    # =========================================================================
    tracker = BoTSORTTracker(
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

    # =========================================================================
    # Run tracking loop
    # =========================================================================
    total_detections = 0
    total_tracks_seen = 0

    for frame_idx, frame in frame_source:
        dets = detections.get(frame_idx)

        if dets is None:
            # No detections this frame — still run tracker for prediction
            confirmed = tracker.update(
                bboxes=np.empty((0, 4)),
                confidences=np.empty(0),
                embeddings=None,
                frame=frame,
            )
        else:
            confirmed = tracker.update(
                bboxes=dets["bboxes"],
                confidences=dets["confidences"],
                embeddings=dets["embeddings"],
                frame=frame,
            )
            total_detections += len(dets["bboxes"])

        # Progress logging every 100 frames
        if frame_idx % 100 == 0:
            n_active = len(tracker.active_tracks)
            n_lost = len(tracker.lost_tracks)
            logger.info(
                f"Frame {frame_idx:5d} | "
                f"Active: {n_active:3d} | Lost: {n_lost:3d} | "
                f"Confirmed: {len(confirmed):3d}"
            )

    # =========================================================================
    # Extract and save tracklets
    # =========================================================================
    all_tracks = tracker.get_all_tracks()
    total_tracks_seen = len(all_tracks)

    tracklets = tracklet_writer.extract_tracklets(all_tracks)

    if tracklets:
        npz_path = tracklet_writer.save(tracklets)
        logger.info(f"Tracklets saved to: {npz_path}")
    else:
        logger.warning("No tracklets met the minimum length requirement.")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 2 Tracking Summary")
    logger.info("=" * 60)
    logger.info(f"  Total detections processed : {total_detections}")
    logger.info(f"  Total tracks created       : {total_tracks_seen}")
    logger.info(f"  Tracklets exported         : {len(tracklets)}")
    logger.info(
        f"  Tracklet lengths           : "
        f"{[t.length for t in tracklets[:10]]}"
        f"{'...' if len(tracklets) > 10 else ''}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
