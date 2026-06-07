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
from bioreef.detection import Detector, build_detector

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester, WaterNetRestorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.infer")


def build_species_mapping(csv_path: str, min_samples: int = 20) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build (species -> idx) mapping from the CSV using the SAME filtering
    as train_stage1.py so class indices align with the MCEAM checkpoint.
    """
    import pandas as pd
    from collections import Counter

    # The CSV only supplies the index→species-name mapping. If it is absent
    # (e.g. running the demo on a non-OzFish video without the training
    # metadata), return an empty mapping — callers fill unmapped indices
    # with placeholder names. The classifier itself still runs: its class
    # count comes from the checkpoint, not this CSV.
    if not os.path.exists(csv_path):
        logger.warning(
            "Species CSV not found (%s); species names unavailable — "
            "predictions will show placeholder labels.", csv_path,
        )
        return {}, {}

    df = pd.read_csv(csv_path).dropna(subset=['species'])
    sp_counter = Counter(df['species'].tolist())
    kept_species = sorted(sp for sp, cnt in sp_counter.items() if cnt >= min_samples)
    sp_to_idx = {sp: i for i, sp in enumerate(kept_species)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}
    return sp_to_idx, idx_to_sp


def resolve_species_mapping(ckpt: dict, csv_path: str, min_samples: int = 20) -> Dict[int, str]:
    """Resolve the species index→name mapping for a Stage 1 checkpoint.

    Authoritative source is the checkpoint itself: training (train_stage1.py)
    now saves `idx_to_sp` so the class indices are self-describing. For older
    checkpoints that lack it, fall back to re-deriving from the CSV — but that
    is only correct if the CSV matches the exact training image set, so a
    warning is emitted.

    Args:
        ckpt:        Loaded Stage 1 checkpoint dict.
        csv_path:    Frame metadata CSV (fallback only).
        min_samples: Species sample threshold (fallback only).

    Returns:
        idx_to_sp mapping {class_idx: species_name}. Keys are ints.
    """
    stored = ckpt.get("idx_to_sp")
    if stored:
        # torch.save/​load may turn int keys into str — normalize back to int.
        return {int(k): v for k, v in stored.items()}

    logger.warning(
        "Checkpoint has no embedded species mapping; re-deriving from %s. "
        "This is only correct if the CSV matches the training image set.",
        csv_path,
    )
    _, idx_to_sp = build_species_mapping(csv_path, min_samples)
    return idx_to_sp


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


# =============================================================================
# Detection (backend-agnostic — bioreef.detection)
# =============================================================================

def detect_frame(
    detector: Detector,
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run detection on a single frame via the bioreef.detection wrapper.

    Args:
        detector:       Any bioreef.detection.Detector (YOLO or RF-DETR).
        frame_bgr:      Original frame (BGR, full resolution).
        conf_threshold: Minimum detection confidence.

    Returns:
        bboxes:       (K, 4) array of [x, y, w, h] in pixels.
        confidences:  (K,) array of scores.
        class_ids:    (K,) array of predicted class indices.
    """
    dets = detector.predict(frame_bgr, conf=conf_threshold)
    return dets.xywh, dets.conf, dets.cls


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    backbone: ViTBackbone,
    mceam: MCEAM,
    head: nn.Module,
    harvester: ContextHarvester,
    frame_bgr: np.ndarray,
    bboxes: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-detection embeddings + species logits for a frame.

    Returns two embeddings and the classifier logits per fish, each
    serving a distinct downstream role:
      - MCEAM-fused (256-D): habitat-aware z_context for the classifier
        head and the Stage 3 tracklet.
      - DINOv3 ROI [CLS] (768-D): the raw, domain-general backbone token
        used by the Stage 2 tracker for Re-ID association (issue #1).
        MCEAM deliberately collapses same-species individuals, so it is
        the wrong descriptor for individual Re-ID; the frozen DINOv3
        token is not species-collapsed and generalizes cross-domain.
      - Species logits (C-D): the per-frame classifier output (issue #2).
        Stored as Stage 1's species *prior* — Stage 3 / track-level
        aggregation (W4) marginalize it up the taxonomy for the
        genus/family hierarchical fallback. Without it, downstream has
        no probability vector to aggregate or back off on.

    Args:
        backbone:  Frozen ViT backbone.
        mceam:     Trained MCEAM fusion module.
        head:      Trained nn.Linear(256, C) species classifier head.
        harvester: ContextHarvester for 4-stream cropping.
        frame_bgr: Original frame (BGR, full resolution).
        bboxes:    (K, 4) array of [x, y, w, h] in pixels.
        device:    CUDA/CPU device.

    Returns:
        (embeddings, reid, logits) where
          embeddings: (K, 256) MCEAM fused embeddings,
          reid:       (K, D)   raw DINOv3 ROI [CLS] tokens (D = backbone dim),
          logits:     (K, C)   per-species classifier logits.
    """
    K = len(bboxes)
    if K == 0:
        D = getattr(backbone, "embed_dim", 768)
        C = head.out_features
        return (
            np.empty((0, 256), dtype=np.float64),
            np.empty((0, D), dtype=np.float64),
            np.empty((0, C), dtype=np.float16),
        )

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

    # Run MCEAM -> fused embedding (256-D) + raw ROI [CLS] (768-D)
    mceam_out = mceam(backbone_features)
    fused = mceam_out["embedding"]                       # (K, 256), on device
    embeddings = fused.float().cpu().numpy()
    reid = mceam_out["roi_cls"].float().cpu().numpy()

    # Run the species head on the fused embedding -> per-class logits.
    # This is Stage 1's per-frame species prior (issue #2).
    logits = head(fused).detach().float().cpu().numpy()  # (K, C)

    return (
        embeddings.astype(np.float64),
        reid.astype(np.float64),
        logits.astype(np.float16),
    )


# =============================================================================
# Per-Video Processing
# =============================================================================

def process_video(
    video_id: str,
    frames: List[Tuple[int, str]],
    backbone: ViTBackbone,
    detector: Detector,
    mceam: MCEAM,
    head: nn.Module,
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
        backbone, detector, mceam, head: Loaded models.
        harvester:  ContextHarvester instance.
        device:     CUDA/CPU device.
        conf_threshold: Detection confidence threshold.
        output_dir: Directory for output .npz files.
        waternet:   If provided, apply WaterNet restoration to each frame.

    Returns:
        Path to the saved .npz file.
    """
    num_classes = head.out_features
    all_frame_ids = []
    all_bboxes = []
    all_confidences = []
    all_embeddings = []
    all_reid = []
    all_logits = []
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
            detector, frame_bgr, conf_threshold,
        )

        if len(bboxes) == 0:
            continue

        # Step 4-6: Extract fused (Stage 3) + Re-ID (Stage 2) embeddings
        #           and species logits (Stage 1 prior)
        embeddings, reid, logits = extract_embeddings(
            backbone, mceam, head, harvester, frame_bgr, bboxes, device,
        )

        # Accumulate
        n_dets = len(bboxes)
        all_frame_ids.append(np.full(n_dets, frame_num, dtype=np.int64))
        all_bboxes.append(bboxes)
        all_confidences.append(confidences)
        all_embeddings.append(embeddings)
        all_reid.append(reid)
        all_logits.append(logits)
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
            embeddings=np.concatenate(all_embeddings),       # 256-D fused → Stage 3
            reid_embeddings=np.concatenate(all_reid),        # 768-D DINOv3 → Stage 2 Re-ID
            logits=np.concatenate(all_logits),               # (N, C) species prior → Stage 3 / W4
            class_ids=np.concatenate(all_class_ids),
        )
    else:
        # Empty archive for videos with no detections
        D = getattr(backbone, "embed_dim", 768)
        np.savez_compressed(
            npz_path,
            frame_ids=np.empty(0, dtype=np.int64),
            bboxes=np.empty((0, 4), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float64),
            embeddings=np.empty((0, 256), dtype=np.float64),
            reid_embeddings=np.empty((0, D), dtype=np.float64),
            logits=np.empty((0, num_classes), dtype=np.float16),
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
    parser.add_argument("--detector_backend", type=str, default="rfdetr",
                        choices=["rfdetr", "yolo"],
                        help="Detector backend (production: rfdetr per #6).")
    parser.add_argument("--detection_ckpt", type=str, default=None,
                        help="Detector checkpoint. Defaults to "
                             "weights/rfdetr_medium_cfd.pth for rfdetr; "
                             "required for yolo.")
    parser.add_argument("--rfdetr_size", type=str, default="medium",
                        choices=["medium", "small", "nano"],
                        help="RF-DETR variant (ignored for yolo). Default: medium.")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO inference imgsz (ignored for rfdetr).")
    parser.add_argument("--stage1_ckpt", type=str,
                        default="bioreef_stage1.pt",
                        help="Path to trained Stage 1 (MCEAM) checkpoint.")
    parser.add_argument("--csv_path", type=str,
                        default="data_oz/metadata/frame_metadata_subset.csv",
                        help="Frame metadata CSV (used to derive MCEAM species "
                             "mapping when the checkpoint lacks an embedded "
                             "one). Defaults to the recovered 256-class subset "
                             "matching bioreef_stage1.pt (recover_species_"
                             "mapping.py / #24); the full 307-species CSV does "
                             "NOT align with this checkpoint's head.")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="Species sample threshold (must match train_stage1.py).")
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

    # Detector (RF-DETR per #6 by default; backend-agnostic wrapper).
    detector = build_detector(
        args.detector_backend,
        weights=args.detection_ckpt,  # None -> backend default
        model_size=args.rfdetr_size,
        imgsz=args.imgsz,
        device=args.device,
    )
    logger.info(
        f"  Detector classes: {detector.names} (class-agnostic — fish only)"
    )

    # Stage 1 MCEAM model
    logger.info(f"Loading Stage 1 model: {args.stage1_ckpt}")
    s1_ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)

    # Species mapping — authoritative source is the checkpoint (train_stage1.py
    # embeds idx_to_sp); falls back to the CSV for older checkpoints.
    # num_classes comes from the head weights, the single source of truth.
    num_classes = s1_ckpt["head"]["weight"].shape[0]
    idx_to_sp = resolve_species_mapping(s1_ckpt, args.csv_path, args.min_samples)
    logger.info(f"  Head classes: {num_classes}  |  species mapping entries: "
                f"{len(idx_to_sp)}")

    # Guard the #24 footgun: the CSV-fallback mapping must line up with the
    # head, or every species name downstream is wrong AND the per-class logits
    # can't be marginalized up the taxonomy (Stage 2 aggregation crashes on
    # the length mismatch). This happens when the checkpoint was trained on a
    # species split that the current CSV + min_samples no longer reproduces
    # (the embedded-mapping fix only protects checkpoints saved after it).
    if idx_to_sp and len(idx_to_sp) != num_classes:
        logger.error(
            "SPECIES MAPPING MISMATCH (#24): head has %d classes but the "
            "CSV-derived mapping has %d species (csv=%s, min_samples=%d). "
            "This checkpoint's class indices cannot be mapped to species "
            "names from this CSV. Boxes + embeddings + Re-ID are unaffected, "
            "but species/genus/family verdicts will be WRONG. Saving a "
            "placeholder mapping so downstream species output is obviously "
            "unusable rather than silently mislabeled.",
            num_classes, len(idx_to_sp), args.csv_path, args.min_samples,
        )
        idx_to_sp = {i: f"__unmapped_{i}__" for i in range(num_classes)}

    mceam = MCEAM(
        embed_dim=backbone.embed_dim,
        num_context_levels=3,
        output_dim=256,
        num_heads=8,
    ).to(device)
    mceam.load_state_dict(s1_ckpt["mceam"])
    mceam.eval()
    logger.info("  MCEAM loaded")

    # Species classifier head — a standalone nn.Linear(256, C), saved
    # under the 'head' key by train_stage1.py (EMA weights). Running it
    # here lets the .npz carry Stage 1's per-frame species prior (issue #2).
    head = nn.Linear(256, num_classes).to(device)
    head.load_state_dict(s1_ckpt["head"])
    head.eval()
    logger.info(f"  Head loaded   : Linear(256, {num_classes})")

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
            detector=detector,
            mceam=mceam,
            head=head,
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
        sp_to_idx={v: k for k, v in idx_to_sp.items()},
        idx_to_sp=idx_to_sp,
    )
    logger.info(f"Species mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
