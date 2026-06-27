"""
Shared visualization / tracklet-IO helpers.

Small utilities used by more than one CLI script (demo_video, visualize_tracklets,
tracklets_to_cvat). Kept here so the scripts share one copy instead of each
carrying its own near-identical version.
"""

import colorsys
import os
from typing import List, Tuple

import numpy as np


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic vivid BGR color from a track ID (golden-ratio hue)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def load_tracklets(npz_path: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Read a TrackletWriter .npz -> [(track_id, frame_ids(T,), bboxes(T,4 xywh))].

    frame_ids are int, bboxes are float [x, y, w, h] — the Tracklet convention
    used throughout the pipeline.
    """
    if not os.path.exists(npz_path):
        raise SystemExit(f"tracklets file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    out = []
    for tid, fids, bxs in zip(data["track_ids"], data["frame_ids"], data["bboxes"]):
        out.append((int(tid), np.asarray(fids, dtype=int),
                    np.asarray(bxs, dtype=float)))
    return out
