"""
Stage 2 I/O plumbing — frame sources, detection loaders, HOTA adapters.

Extracted from track_stage2.py so the CLI and the in-process run_stage2 share
one copy (no duplication, no script-imports-script). Bodies verbatim.

    VideoFrameSource / DirectoryFrameSource / NullFrameSource
    load_detections_npz(path) / load_detections_csv(csv, img_dir)
    load_gt_tracks(path) / tracklets_to_hota_format(tracklets)
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bioreef._9_pipeline.frame_sources")


# =============================================================================
# Detection loaders
# =============================================================================

def load_detections_npz(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load precomputed detections from a .npz archive.

    Returns frame_id -> {bboxes, confidences, embeddings, reid_embeddings,
    [logits]}. Backward compatible: archives without 'reid_embeddings' fall
    back to 'embeddings'; archives without 'logits' omit that field.
    """
    data = np.load(path, allow_pickle=True)
    frame_ids = data["frame_ids"]
    bboxes = data["bboxes"]
    confidences = data["confidences"]
    embeddings = data["embeddings"]
    reid_embeddings = data["reid_embeddings"] if "reid_embeddings" in data \
        else embeddings
    logits = data["logits"] if "logits" in data else None

    per_frame: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"bboxes": [], "confidences": [],
                 "embeddings": [], "reid_embeddings": [], "logits": []}
    )

    for i in range(len(frame_ids)):
        fid = int(frame_ids[i])
        per_frame[fid]["bboxes"].append(bboxes[i])
        per_frame[fid]["confidences"].append(confidences[i])
        per_frame[fid]["embeddings"].append(embeddings[i])
        per_frame[fid]["reid_embeddings"].append(reid_embeddings[i])
        if logits is not None:
            per_frame[fid]["logits"].append(logits[i])

    result = {}
    for fid, arrays in per_frame.items():
        result[fid] = {
            "bboxes": np.array(arrays["bboxes"], dtype=np.float64),
            "confidences": np.array(arrays["confidences"], dtype=np.float64),
            "embeddings": np.array(arrays["embeddings"], dtype=np.float64),
            "reid_embeddings": np.array(
                arrays["reid_embeddings"], dtype=np.float64
            ),
        }
        if logits is not None:
            result[fid]["logits"] = np.array(
                arrays["logits"], dtype=np.float32
            )

    return result


def load_detections_csv(
    csv_path: str,
    img_dir: str,
    embedding_dim: int = 256,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[str]]:
    """
    Mock detections from the frame_metadata CSV (for testing Stage 2 before the
    detection head exists). Embeddings are zero-vectors. Returns
    (detections_dict, ordered frame_paths).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required_cols = {"file_name", "x0", "y0", "x1", "y1"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. "
            f"Found: {set(df.columns)}"
        )

    grouped = df.groupby("file_name")
    frame_names = sorted(grouped.groups.keys())

    detections = {}
    frame_paths = []
    for frame_idx, fname in enumerate(frame_names):
        group = grouped.get_group(fname)
        bboxes = []
        for _, row in group.iterrows():
            x0, y0, x1, y1 = (int(row["x0"]), int(row["y0"]),
                              int(row["x1"]), int(row["y1"]))
            bboxes.append([x0, y0, x1 - x0, y1 - y0])
        n = len(bboxes)
        detections[frame_idx] = {
            "bboxes": np.array(bboxes, dtype=np.float64),
            "confidences": np.ones(n, dtype=np.float64),
            "embeddings": np.zeros((n, embedding_dim), dtype=np.float64),
        }
        frame_paths.append(os.path.join(img_dir, fname))
    return detections, frame_paths


# =============================================================================
# Frame sources
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
    """Read frames from a directory of images. frame_id parsed from filename
    (*.avi.NNN.png), matching infer_stage1's IDs; sequential fallback."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    _FRAME_RE = re.compile(r"\.avi\.(\d+)\.")

    def __init__(self, frames_dir: str, paths: Optional[List[str]] = None):
        if paths is not None:
            raw_paths = list(paths)
        else:
            raw_paths = [
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if os.path.splitext(f)[1].lower() in self.EXTENSIONS
            ]

        indexed: List[Tuple[int, str]] = []
        for i, p in enumerate(raw_paths):
            m = self._FRAME_RE.search(os.path.basename(p))
            fid = int(m.group(1)) if m else i
            indexed.append((fid, p))
        indexed.sort(key=lambda x: x[0])

        self.frame_ids = [fid for fid, _ in indexed]
        self.paths = [p for _, p in indexed]
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
        fid = self.frame_ids[self._idx]
        self._idx += 1
        return fid, frame

    def __len__(self):
        return self.total_frames


class NullFrameSource:
    """Synthetic frame source for no-frames mode — yields blank (1x1) frames so
    the tracker runs without images. CMC must be disabled with this source."""

    def __init__(self, frame_ids: List[int]):
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
# HOTA adapters
# =============================================================================

def load_gt_tracks(gt_path: str) -> Dict[str, Dict]:
    """Load ground-truth tracking JSON: {video_id: {frame_id: [{track_id,
    bbox}]}}. String frame_id keys are converted to int."""
    with open(gt_path) as f:
        raw = json.load(f)
    gt_data = {}
    for video_id, frames in raw.items():
        gt_data[video_id] = {int(fid): anns for fid, anns in frames.items()}
    return gt_data


def tracklets_to_hota_format(tracklets: List) -> Dict[int, List[Dict]]:
    """Convert tracklets to HOTA's per-frame format:
    {frame_id: [{'track_id', 'bbox'}, ...]}."""
    result: Dict[int, List[Dict]] = defaultdict(list)
    for tracklet in tracklets:
        for frame in tracklet.frames:
            # Tracklet.frames entries are (frame_id, bbox, embedding, logits);
            # index defensively so this survives the tuple arity (originally
            # 3-tuple unpacking — would crash on the current 4-tuple).
            frame_id, bbox = frame[0], frame[1]
            result[frame_id].append({
                "track_id": tracklet.track_id,
                "bbox": bbox.tolist(),
            })
    return dict(result)
