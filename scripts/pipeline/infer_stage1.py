"""
BioReef.ai — Stage 1 Inference Pipeline
=========================================
Runs the single-class detector (RF-DETR per #6) + MCEAM models on a directory of frames,
producing per-video .npz archives for Stage 2 tracking input.

Pipeline per frame:
    1. Run class-agnostic detector (RF-DETR or legacy YOLO) -> fish bounding boxes + confidences
    2. Filter detections by confidence threshold
    3. For each detection, ContextHarvester generates 4 streams (ROI/Social/Habitat/Full)
    4. Run backbone + MCEAM on the 4 streams -> 256-dim embedding per detection
    5. Run the species head on the fused embedding -> per-class logits
    6. Accumulate (frame_id, bboxes, confidences, embeddings, logits)

The species mapping (idx -> species name) is derived from the training CSV,
using the same deterministic filtering as train_stage1.py so it matches the
MCEAM checkpoint. The detector's class output is always 0 ("fish") and is
not used for species identification.

Output per video (.npz):
    frame_ids:       (N,) int array of frame numbers
    bboxes:          (N, 4) float array of [x, y, w, h] in pixels
    confidences:     (N,) float array of detection confidence scores
    embeddings:      (N, 256) float array of MCEAM fused embeddings
    reid_embeddings: (N, D)  float array of raw DINOv3 ROI [CLS] tokens
    logits:          (N, C)  float16 array of per-species classifier logits
                     — Stage 1's per-frame species prior. Stage 3 / track-
                     level aggregation (W4) marginalize this up the taxonomy
                     for the genus/family hierarchical fallback. Logits, not
                     softmax, so downstream can re-temperature losslessly.
    class_ids:       (N,) int array (all zeros — fish class from detector)

Usage:
    python infer_stage1.py \\
        --frames_dir data_oz/frames_waternet_1 data_oz/frames_waternet_2 \\
        --detection_ckpt runs/detect/trainX/weights/best.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --csv_path data_oz/metadata/frame_metadata_subset.csv \\
        --output_dir outputs/detections
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._2_stage1 import Detector, build_detector

from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._11_restoration import WaterNetRestorer
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._9_pipeline.config import InferenceConfig, DEFAULT_CONFIG_PATH
from bioreef._9_pipeline.models import load_models
from bioreef._9_pipeline.io import Frames
from bioreef._9_pipeline._92_detect import run_stage1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.infer")


# Species-mapping helpers now live in the library; re-exported here so existing
# `from infer_stage1 import resolve_species_mapping` imports keep working.
from bioreef._1_preprocess._15_dataset_split import (   # noqa: E402,F401
    build_species_mapping,
    resolve_species_mapping,
)


# Frame filename pattern: {video_id}.{frame_number}.png
# video_id is everything up to the final ".<frame>.png". The video_id may
# itself carry an extension (OzFish frames are "<name>.avi.<n>.png"; Khorfakkan
# pre-labeling writes "<clip>.mp4.<n>.png") — both group correctly because the
# capture is greedy up to the last numeric ".<n>.png" suffix.
FRAME_PATTERN = re.compile(r"^(.+)\.(\d+)\.png$")


# =============================================================================
# Frame Discovery & Grouping
# =============================================================================

def discover_videos(
    frames_dirs: List[str],
    video_id: Optional[str] = None,
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Scan multiple directories for frame images and group by video ID.

    Frames may be split across several directories (e.g. separate drives).
    If the same frame appears in multiple directories, the first occurrence
    wins (directories are searched in order).

    Filename format: {video_id}.{frame_number}.png
        e.g. A000001_L.avi.5107.png -> video_id=A000001_L.avi, frame=5107

    Args:
        frames_dirs: List of directories containing frame images.
        video_id:    If set, only return frames for this video.

    Returns:
        Dict mapping video_id -> sorted list of (frame_number, file_path).
    """
    videos: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    seen: set = set()  # (video_id, frame_num) to avoid duplicates

    for frames_dir in frames_dirs:
        if not os.path.isdir(frames_dir):
            logger.warning(f"Frames directory not found, skipping: {frames_dir}")
            continue

        for fname in os.listdir(frames_dir):
            match = FRAME_PATTERN.match(fname)
            if not match:
                continue

            vid_id = match.group(1)
            frame_num = int(match.group(2))

            if video_id is not None and vid_id != video_id:
                continue

            key = (vid_id, frame_num)
            if key in seen:
                continue
            seen.add(key)

            videos[vid_id].append((frame_num, os.path.join(frames_dir, fname)))

    # Sort each video's frames by frame number (temporal order)
    for vid_id in videos:
        videos[vid_id].sort(key=lambda x: x[0])

    return dict(videos)


# Detection + embedding-extraction primitives now live in the library so the
# demo overlay can import them without importing this CLI script. Re-exported
# here for backward compat (`from infer_stage1 import detect_frame`).
from bioreef._9_pipeline._92_detect import (   # noqa: E402,F401
    detect_frame,
    extract_embeddings,
)




# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioReef.ai Stage 1 Inference — Detection + Embedding. "
                    "All settings come from the config file (see config.yaml); "
                    "the only argument is its path."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help=f"Pipeline config YAML. Default: {DEFAULT_CONFIG_PATH}")
    args = parser.parse_args()

    cfg = InferenceConfig.from_yaml(args.config)
    if not cfg.frames_dir:
        logger.error("inference.frames_dir is empty in %s — Stage 1 reads "
                     "extracted frame dirs.", args.config)
        return

    # Discover + group frames (filename pattern -> video_id).
    videos = discover_videos(cfg.frames_dir, cfg.video_id)
    if not videos:
        logger.error("No frames found. Check inference.frames_dir / video_id "
                     "and the frame filename pattern.")
        return
    total_frames = sum(len(f) for f in videos.values())
    logger.info(f"Found {len(videos)} videos, {total_frames} frames total")

    # Load every model once (backbone/detector/mceam/head/harvester/waternet).
    models = load_models(cfg)

    logger.info("=" * 60)
    logger.info("BioReef.ai — Stage 1 Inference")
    logger.info(f"  Device        : {models.device}")
    logger.info(f"  Conf threshold: {cfg.conf_threshold}")
    logger.info(f"  Videos        : {len(videos)}")
    logger.info(f"  Output dir    : {cfg.output_dir}")
    logger.info("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for vid_id in tqdm(sorted(videos.keys()), desc="Videos"):
        frame_list = videos[vid_id]            # [(frame_num, path), ...]
        frame_obj = Frames(
            video_id=vid_id,
            frame_ids=[fn for fn, _ in frame_list],
            paths=[p for _, p in frame_list],
        )
        out = run_stage1(frame_obj.iter_frames(), models, cfg, video_id=vid_id)
        safe_name = vid_id.replace(".avi", "").replace(".", "_")
        out.save(os.path.join(cfg.output_dir, f"{safe_name}.npz"))
        logger.info(f"  {vid_id}: {len(out)} detections -> "
                    f"{safe_name}.npz")

    logger.info("=" * 60)
    logger.info("Inference Complete")
    logger.info(f"  Output directory : {cfg.output_dir}")
    logger.info("=" * 60)

    # Save species mapping alongside detections (Stage 2 reads this).
    mapping_path = os.path.join(cfg.output_dir, "species_mapping.npz")
    np.savez_compressed(
        mapping_path,
        sp_to_idx={v: k for k, v in models.idx_to_sp.items()},
        idx_to_sp=models.idx_to_sp,
    )
    logger.info(f"Species mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
