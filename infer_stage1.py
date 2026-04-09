"""
BioReef.ai — Stage 1 Inference Pipeline
=========================================
Runs the trained detection + MCEAM models on a directory of frames,
producing per-video .npz archives for Stage 2 tracking input.

Pipeline per frame:
    1. Resize frame to 512x512, extract patch tokens (frozen backbone)
    2. Run BioReefDetector -> bounding boxes + class logits
    3. Filter detections by confidence threshold
    4. For each detection, ContextHarvester generates 4 streams (ROI/Social/Habitat/Full)
    5. Run backbone + MCEAM on the 4 streams -> 256-dim embedding per detection
    6. Accumulate (frame_id, bboxes, confidences, embeddings)

Output per video (.npz):
    frame_ids:    (N,) int array of frame numbers
    bboxes:       (N, 4) float array of [x, y, w, h] in pixels
    confidences:  (N,) float array of detection confidence scores
    embeddings:   (N, 256) float array of MCEAM fused embeddings
    class_ids:    (N,) int array of predicted species indices

Usage:
    python infer_stage1.py \\
        --frames_dir /media/openuae/UUI/frames_waternet \\
            data/frames_waternet_1 data/frames_waternet_2 \\
        --detection_ckpt bioreef_detection.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --output_dir outputs/detections

    # Process a specific video only:
    python infer_stage1.py \\
        --frames_dir /media/openuae/UUI/frames_waternet \\
            data/frames_waternet_1 data/frames_waternet_2 \\
        --detection_ckpt bioreef_detection.pt \\
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
import torch.nn.functional as F
from tqdm import tqdm

from bioreef.models.backbone import ViTBackbone
from bioreef.models.detector import BioReefDetector
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.infer")

# ImageNet normalization (must match detection training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
# Detection Post-Processing
# =============================================================================

def postprocess_detections(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    orig_h: int,
    orig_w: int,
    conf_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw detector output to pixel-space detections.

    Args:
        pred_logits: (num_queries, num_classes+1) logits.
        pred_boxes:  (num_queries, 4) normalized [cx, cy, w, h].
        orig_h:      Original frame height.
        orig_w:      Original frame width.
        conf_threshold: Minimum confidence for a valid detection.

    Returns:
        bboxes:       (K, 4) array of [x, y, w, h] in pixels.
        confidences:  (K,) array of scores.
        class_ids:    (K,) array of predicted class indices.
    """
    # Softmax over all classes (including background at last index)
    probs = F.softmax(pred_logits, dim=-1)

    # Confidence = max probability over non-background classes
    # Background is the last class (index num_classes)
    foreground_probs = probs[:, :-1]
    confidences, class_ids = foreground_probs.max(dim=-1)

    # Filter by confidence
    mask = confidences >= conf_threshold
    confidences = confidences[mask].cpu().numpy()
    class_ids = class_ids[mask].cpu().numpy()
    boxes = pred_boxes[mask].cpu().numpy()

    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    # Convert normalized cxcywh -> pixel xywh (top-left corner + size)
    cx = boxes[:, 0] * orig_w
    cy = boxes[:, 1] * orig_h
    w = boxes[:, 2] * orig_w
    h = boxes[:, 3] * orig_h

    x = cx - w / 2.0
    y = cy - h / 2.0

    bboxes = np.stack([x, y, w, h], axis=1).astype(np.float64)

    return bboxes, confidences.astype(np.float64), class_ids.astype(np.int64)


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
# Per-Frame Detection
# =============================================================================

@torch.no_grad()
def detect_frame(
    backbone: ViTBackbone,
    detector: BioReefDetector,
    frame_bgr: np.ndarray,
    input_size: int,
    device: torch.device,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run detection on a single frame.

    Args:
        backbone:   Frozen ViT backbone.
        detector:   Trained BioReefDetector.
        frame_bgr:  Original frame (BGR, full resolution).
        input_size:  Detection resolution (512).
        device:     CUDA/CPU device.
        conf_threshold: Minimum detection confidence.

    Returns:
        bboxes, confidences, class_ids (in pixel coordinates).
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    # Preprocess for detection: resize + normalize
    img = cv2.resize(frame_bgr, (input_size, input_size),
                     interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Extract patch tokens and run detector
    patch_tokens = backbone.extract_patch_tokens(img_tensor)
    outputs = detector(patch_tokens, targets=None)

    # Post-process: first image in batch
    bboxes, confidences, class_ids = postprocess_detections(
        outputs["pred_logits"][0],
        outputs["pred_boxes"][0],
        orig_h, orig_w,
        conf_threshold,
    )

    return bboxes, confidences, class_ids


# =============================================================================
# Per-Video Processing
# =============================================================================

def process_video(
    video_id: str,
    frames: List[Tuple[int, str]],
    backbone: ViTBackbone,
    detector: BioReefDetector,
    mceam: MCEAM,
    harvester: ContextHarvester,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
    output_dir: str,
) -> str:
    """
    Process all frames of one video through detection + embedding extraction.

    Args:
        video_id:   Video identifier (e.g. 'A000001_L.avi').
        frames:     Sorted list of (frame_number, file_path).
        backbone, detector, mceam: Loaded models.
        harvester:  ContextHarvester instance.
        device:     CUDA/CPU device.
        input_size: Detection resolution.
        conf_threshold: Detection confidence threshold.
        output_dir: Directory for output .npz files.

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

        # Step 1-3: Detect fish in the frame
        bboxes, confidences, class_ids = detect_frame(
            backbone, detector, frame_bgr, input_size, device, conf_threshold,
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
                        default="bioreef_detection.pt",
                        help="Path to trained detection checkpoint.")
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
    parser.add_argument("--input_size", type=int, default=512,
                        help="Detection input resolution.")
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

    # Detection model
    logger.info(f"Loading detection model: {args.detection_ckpt}")
    det_ckpt = torch.load(args.detection_ckpt, map_location=device, weights_only=False)
    det_args = det_ckpt["args"]
    num_classes = det_ckpt["num_classes"]
    sp_to_idx = det_ckpt["sp_to_idx"]
    idx_to_sp = {v: k for k, v in sp_to_idx.items()}

    detector = BioReefDetector(
        backbone_dim=backbone.embed_dim,
        hidden_dim=det_args["hidden_dim"],
        num_queries=det_args["num_queries"],
        num_classes=num_classes,
        num_decoder_layers=det_args["num_decoder_layers"],
        num_fdr_bins=det_args["num_fdr_bins"],
    ).to(device)
    detector.load_state_dict(det_ckpt["detector"])
    detector.eval()
    logger.info(f"  Detector: {num_classes} classes, {det_args['num_queries']} queries")

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
    # Context Harvester
    # =========================================================================
    harvester = ContextHarvester()

    # =========================================================================
    # Process each video
    # =========================================================================
    logger.info("=" * 60)
    logger.info("BioReef.ai — Stage 1 Inference")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Detection res : {args.input_size}x{args.input_size}")
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
            detector=detector,
            mceam=mceam,
            harvester=harvester,
            device=device,
            input_size=args.input_size,
            conf_threshold=args.conf_threshold,
            output_dir=args.output_dir,
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
