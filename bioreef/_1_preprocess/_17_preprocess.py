"""
Shared preprocessing — the one place both the training and inference pipelines
turn raw video into frames ready for the model.

    prepare_frames(video, cfg) -> Frames
        Extract every frame of `video` to PNGs named "<video_id>.NNNNNN.png"
        (the convention Stage 1's frame discovery groups on), optionally
        WaterNet-restore them, and return a Frames handle.

Design notes:
- The detector runs on RAW frames (#14); restoration here is OFF by default and
  is meant only for human-eye uses (e.g. CVAT display). cfg.apply_waternet
  controls it.
- Filenames carry the video extension in the id (e.g. "clip01.mp4.000012.png")
  so two clips with the same stem but different extensions never collide.
- Extraction is skipped if the output dir already holds >= the reported frame
  count (resumable / idempotent), matching prelabel_clips' behaviour.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2

from bioreef._9_pipeline.io import Frames

logger = logging.getLogger("bioreef._1_preprocess._17_preprocess")


def extract_frames(
    video: str,
    out_dir: str,
    video_id: str,
) -> list:
    """Extract every frame of `video` to out_dir as <video_id>.NNNNNN.png.
    Returns the sorted list of (frame_index, path). Idempotent: if the dir
    already holds >= the reported frame count, reuses what's there."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith(video_id + ".") and f.endswith(".png")
    )
    if total > 0 and len(existing) >= total:
        cap.release()
        logger.info("frames: %d already extracted, reusing", len(existing))
        return [(_frame_index(f), os.path.join(out_dir, f)) for f in existing]

    out = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        path = os.path.join(out_dir, f"{video_id}.{n:06d}.png")
        cv2.imwrite(path, frame)
        out.append((n, path))
        n += 1
    cap.release()
    logger.info("frames: extracted %d", n)
    return out


def _frame_index(fname: str) -> int:
    """Parse NNNNNN from '<video_id>.NNNNNN.png'."""
    stem = fname[:-4] if fname.endswith(".png") else fname
    return int(stem.rsplit(".", 1)[-1])


def restore_frame_dir(in_dir: str, out_dir: str, skip_existing: bool = True):
    """WaterNet-restore every PNG in in_dir into out_dir, preserving filenames
    (so frame indices stay aligned). Used when cfg.apply_waternet is set."""
    from bioreef._1_preprocess._11_restoration import WaterNetRestorer
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".png"))
    restorer = WaterNetRestorer()
    restorer._load_model()
    for f in files:
        dst = os.path.join(out_dir, f)
        if skip_existing and os.path.exists(dst):
            continue
        img = cv2.imread(os.path.join(in_dir, f))
        if img is None:
            continue
        cv2.imwrite(dst, restorer(img))
    logger.info("restored %d frames -> %s", len(files), out_dir)


def prepare_frames(
    video: str,
    cfg,
    frames_root: Optional[str] = None,
) -> Frames:
    """Turn a video into a Frames handle for Stage 1. frames_root defaults to a
    '<clip>_frames' dir next to the video; if cfg.apply_waternet, paths point at
    the restored copies (same indices)."""
    if not os.path.exists(video):
        raise SystemExit(f"video not found: {video}")

    clip_dir = os.path.dirname(video)
    clip_base = os.path.splitext(os.path.basename(video))[0]
    video_id = getattr(cfg, "video_id", None) or os.path.basename(video)

    frames_dir = frames_root or os.path.join(clip_dir, f"{clip_base}_frames")
    indexed = extract_frames(video, frames_dir, video_id)

    if getattr(cfg, "apply_waternet", False):
        restored_dir = os.path.join(clip_dir, f"{clip_base}_frames_restored")
        restore_frame_dir(frames_dir, restored_dir, skip_existing=True)
        # Re-point paths at the restored copies (same filenames/indices).
        indexed = [(idx, os.path.join(restored_dir, os.path.basename(p)))
                   for idx, p in indexed]

    indexed.sort(key=lambda x: x[0])
    return Frames(
        video_id=video_id,
        frame_ids=[i for i, _ in indexed],
        paths=[p for _, p in indexed],
    )
