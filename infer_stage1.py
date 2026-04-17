"""
BioReef.ai — Stage 1 Inference Pipeline
=========================================
Runs the trained YOLO detector + MCEAM models on a directory of frames,
producing per-video .npz archives for Stage 2 tracking input.

Pipeline per frame:
    1. Run YOLOv11 detector -> bounding boxes, confidences, class IDs
    2. Filter detections by confidence threshold (handled by YOLO)
    3. For each detection, ContextHarvester generates 4 streams (ROI/Social/Habitat/Full)
    4. Run backbone + MCEAM on the 4 streams -> 256-dim embedding per detection
    5. Accumulate (frame_id, bboxes, confidences, embeddings)

Output per video (.npz):
    frame_ids:    (N,) int array of frame numbers
    bboxes:       (N, 4) float array of [x, y, w, h] in pixels
    confidences:  (N,) float array of detection confidence scores
    embeddings:   (N, 256) float array of MCEAM fused embeddings
    class_ids:    (N,) int array of predicted species indices

Usage:
    python infer_stage1.py \\
        --frames_dir data_oz/frames_waternet_1 data_oz/frames_waternet_2 \\
        --detection_ckpt runs/detect/trainX/weights/best.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --output_dir outputs/detections

    # Process a specific video only:
    python infer_stage1.py \\
        --frames_dir data_oz/frames_waternet_1 data_oz/frames_waternet_2 \\
        --detection_ckpt runs/detect/trainX/weights/best.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --video_id A000001_L.avi
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
from tqdm import tqdm
from ultralytics import YOLO

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.infer")

# Frame filename pattern: {video_id}.{frame_number}.png
FRAME_PATTERN = re.compile(r"^(.+\.avi)\.(\d+)\.png$")


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


# =============================================================================
# Detection (YOLO)
# =============================================================================

def detect_frame(
    yolo: YOLO,
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run YOLOv11 detection on a single frame.

    Args:
        yolo:           Loaded Ultralytics YOLO model.
        frame_bgr:      Original frame (BGR, full resolution).
        conf_threshold: Minimum detection confidence.

    Returns:
        bboxes:       (K, 4) array of [x, y, w, h] in pixels.
        confidences:  (K,) array of scores.
        class_ids:    (K,) array of predicted class indices.
    """
    results = yolo(frame_bgr, conf=conf_threshold, verbose=False)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    xyxy = boxes.xyxy.cpu().numpy()
    x = xyxy[:, 0]
    y = xyxy[:, 1]
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]

    bboxes = np.stack([x, y, w, h], axis=1).astype(np.float64)
    confidences = boxes.conf.cpu().numpy().astype(np.float64)
    class_ids = boxes.cls.cpu().numpy().astype(np.int64)

    return bboxes, confidences, class_ids


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    backbone: ViTBackbone,
    mceam: MCEAM,
    harvester: ContextHarvester,
    frame_bgr: np.ndarray,
    bboxes: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Extract MCEAM embeddings for each detected fish in a frame.

    Args:
        backbone:  Frozen ViT backbone.
        mceam:     Trained MCEAM fusion module.
        harvester: ContextHarvester for 4-stream cropping.
        frame_bgr: Original frame (BGR, full resolution).
        bboxes:    (K, 4) array of [x, y, w, h] in pixels.
        device:    CUDA/CPU device.

    Returns:
        (K, 256) array of MCEAM fused embeddings.
    """
    K = len(bboxes)
    if K == 0:
        return np.empty((0, 256), dtype=np.float64)

    # Harvest 4-stream crops for each detection
    stream_lists = defaultdict(list)

    for bbox in bboxes:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Clamp to valid dimensions
        w = max(w, 1)
        h = max(h, 1)
        crops = harvester.harvest(frame_bgr, (x, y, w, h))
        for stream_name, tensor in crops.items():
            stream_lists[stream_name].append(tensor)

    # Stack into batched tensors: (K, 3, 224, 224)
    streams_batched = {
        name: torch.stack(tensors).to(device)
        for name, tensors in stream_lists.items()
    }

    # Run backbone on each stream -> (cls_token, patch_tokens) per stream
    backbone_features = backbone(streams_batched)

    # Run MCEAM -> fused embeddings
    mceam_out = mceam(backbone_features)
    embeddings = mceam_out["embedding"].cpu().numpy()

    return embeddings.astype(np.float64)


# =============================================================================
# Per-Video Processing
# =============================================================================

def process_video(
    video_id: str,
    frames: List[Tuple[int, str]],
    backbone: ViTBackbone,
    yolo: YOLO,
    mceam: MCEAM,
    harvester: ContextHarvester,
    device: torch.device,
    conf_threshold: float,
    output_dir: str,
    waternet: Optional[WaterNetRestorer] = None,
) -> str:
    """
    Process all frames of one video through detection + embedding extraction.

    Args:
        video_id:   Video identifier (e.g. 'A000001_L.avi').
        frames:     Sorted list of (frame_number, file_path).
        backbone, yolo, mceam: Loaded models.
        harvester:  ContextHarvester instance.
        device:     CUDA/CPU device.
        conf_threshold: Detection confidence threshold.
        output_dir: Directory for output .npz files.
        waternet:   If provided, apply WaterNet restoration to each frame.

    Returns:
        Path to the saved .npz file.
    """
    all_frame_ids = []
    all_bboxes = []
    all_confidences = []
    all_embeddings = []
    all_class_ids = []

    pbar = tqdm(frames, desc=f"  {video_id}", leave=False)
    for frame_num, frame_path in pbar:
        frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning(f"Could not read: {frame_path}")
            continue

        # WaterNet inline restoration (if enabled)
        if waternet is not None:
            frame_bgr = waternet(frame_bgr)

        # Step 1-3: Detect fish in the frame
        bboxes, confidences, class_ids = detect_frame(
            yolo, frame_bgr, conf_threshold,
        )

        if len(bboxes) == 0:
            continue

        # Step 4-5: Extract MCEAM embeddings for each detection
        embeddings = extract_embeddings(
            backbone, mceam, harvester, frame_bgr, bboxes, device,
        )

        # Accumulate
        n_dets = len(bboxes)
        all_frame_ids.append(np.full(n_dets, frame_num, dtype=np.int64))
        all_bboxes.append(bboxes)
        all_confidences.append(confidences)
        all_embeddings.append(embeddings)
        all_class_ids.append(class_ids)

        pbar.set_postfix(dets=n_dets)

    # Save per-video .npz
    os.makedirs(output_dir, exist_ok=True)
    safe_name = video_id.replace(".avi", "").replace(".", "_")
    npz_path = os.path.join(output_dir, f"{safe_name}.npz")

    if all_frame_ids:
        np.savez_compressed(
            npz_path,
            frame_ids=np.concatenate(all_frame_ids),
            bboxes=np.concatenate(all_bboxes),
            confidences=np.concatenate(all_confidences),
            embeddings=np.concatenate(all_embeddings),
            class_ids=np.concatenate(all_class_ids),
        )
    else:
        # Empty archive for videos with no detections
        np.savez_compressed(
            npz_path,
            frame_ids=np.empty(0, dtype=np.int64),
            bboxes=np.empty((0, 4), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float64),
            embeddings=np.empty((0, 256), dtype=np.float64),
            class_ids=np.empty(0, dtype=np.int64),
        )

    total = sum(len(b) for b in all_bboxes)
    logger.info(
        f"  {video_id}: {len(frames)} frames, {total} detections -> {npz_path}"
    )
    return npz_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioReef.ai Stage 1 Inference — Detection + Embedding"
    )
    parser.add_argument("--frames_dir", type=str, nargs="+", required=True,
                        help="One or more directories containing frame images.")
    parser.add_argument("--detection_ckpt", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Path to YOLO detection checkpoint (best.pt).")
    parser.add_argument("--stage1_ckpt", type=str,
                        default="bioreef_stage1.pt",
                        help="Path to trained Stage 1 (MCEAM) checkpoint.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/detections",
                        help="Directory for per-video .npz output files.")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Process only this video ID (e.g. A000001_L.avi).")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Minimum detection confidence.")
    parser.add_argument("--apply_waternet", action="store_true",
                        help="Apply WaterNet restoration to each frame before detection.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available).")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # =========================================================================
    # Discover videos
    # =========================================================================
    videos = discover_videos(args.frames_dir, args.video_id)  # frames_dir is a list
    if not videos:
        logger.error("No frames found. Check --frames_dir and filename pattern.")
        return

    total_frames = sum(len(f) for f in videos.values())
    logger.info(f"Scanning {len(args.frames_dir)} directories...")
    logger.info(f"Found {len(videos)} videos, {total_frames} frames total")

    # =========================================================================
    # Load models
    # =========================================================================
    logger.info("Loading backbone...")
    backbone = ViTBackbone(freeze=True).to(device)
    backbone.eval()

    # YOLO detection model
    logger.info(f"Loading YOLO detector: {args.detection_ckpt}")
    yolo = YOLO(args.detection_ckpt)
    sp_to_idx = {name: idx for idx, name in yolo.names.items()}
    idx_to_sp = yolo.names
    num_classes = len(yolo.names)
    logger.info(f"  YOLO: {num_classes} classes")

    # Stage 1 MCEAM model
    logger.info(f"Loading Stage 1 model: {args.stage1_ckpt}")
    s1_ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)

    mceam = MCEAM(
        embed_dim=backbone.embed_dim,
        num_context_levels=3,
        output_dim=256,
        num_heads=8,
    ).to(device)
    mceam.load_state_dict(s1_ckpt["mceam"])
    mceam.eval()
    logger.info("  MCEAM loaded")

    # =========================================================================
    # WaterNet (optional inline restoration)
    # =========================================================================
    waternet = None
    if args.apply_waternet:
        logger.info("Loading WaterNet for inline restoration...")
        waternet = WaterNetRestorer()
        # Trigger lazy load now so any errors surface early
        waternet._load_model()

    # =========================================================================
    # Context Harvester
    # =========================================================================
    harvester = ContextHarvester()

    # =========================================================================
    # Process each video
    # =========================================================================
    logger.info("=" * 60)
    logger.info("BioReef.ai — Stage 1 Inference")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Conf threshold: {args.conf_threshold}")
    logger.info(f"  Videos        : {len(videos)}")
    logger.info(f"  Total frames  : {total_frames}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info("=" * 60)

    all_npz_paths = []

    for vid_id in tqdm(sorted(videos.keys()), desc="Videos"):
        npz_path = process_video(
            video_id=vid_id,
            frames=videos[vid_id],
            backbone=backbone,
            yolo=yolo,
            mceam=mceam,
            harvester=harvester,
            device=device,
            conf_threshold=args.conf_threshold,
            output_dir=args.output_dir,
            waternet=waternet,
        )
        all_npz_paths.append(npz_path)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Inference Complete")
    logger.info(f"  Videos processed : {len(all_npz_paths)}")
    logger.info(f"  Output directory : {args.output_dir}")
    logger.info("=" * 60)

    # Save species mapping alongside detections for reference
    mapping_path = os.path.join(args.output_dir, "species_mapping.npz")
    np.savez_compressed(
        mapping_path,
        sp_to_idx=sp_to_idx,
        idx_to_sp=idx_to_sp,
    )
    logger.info(f"Species mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
