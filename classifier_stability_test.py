"""
Test classifier prediction stability across raw vs WaterNet-restored inputs.

The question this answers: if we drop WaterNet from the production inference
pipeline (because it hurts the detector by 5.6 pp recall), how much does
the classifier's behavior change as a consequence?

The classifier was trained exclusively on WaterNet-restored OzFish frames.
If we feed it raw Khorfakkan crops in production, that's a train/inference
distribution shift. This script measures the magnitude of that shift
WITHOUT requiring species labels (which we don't have for Khorfakkan).

Strategy:
    For every GT box in the given clips, run the FULL Stage 1 classifier
    (ContextHarvester 4-stream → backbone → MCEAM → head) twice:
      (a) using the raw source frame, and
      (b) using the WaterNet-restored source frame.
    Compare the predictions on each crop:
      - Top-1 agreement rate (do the two inputs predict the same species?)
      - Top-5 agreement rate (does raw's top-1 land in restored's top-5?)
      - Softmax cosine similarity (how similar are the full distributions?)

If predictions are highly stable (e.g. >90% Top-1 agreement) → MCEAM is
robust to the distribution shift; dropping WaterNet is safe.
If predictions diverge wildly (e.g. <60% agreement) → MCEAM is sensitive
to the input distribution; retraining on raw OzFish is justified.

This is the CHEAP test that gates the EXPENSIVE retrain decision.

Usage:
    # Default: test on all three labeled Khorfakkan clips
    python classifier_stability_test.py

    # Override clip list
    python classifier_stability_test.py --clip <dir> [--clip <dir> ...]

    # Each clip dir is expected to contain:
    #   <clip>/annotations/instances_default.json
    #   <clip>/images/        (or images/default/)
    #   <clip>/images_restored/ (created if missing via restore_frames.py logic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bioreef.data.data_factory import (
    ContextHarvester,
    WaterNetRestorer,
)
from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from infer_stage1 import resolve_species_mapping


# Default test clips — the three labeled Khorfakkan artifacts on the VM.
DEFAULT_CLIPS = [
    "annotations",
    "Khorfakkan/10-59-21_DEEP_TREKKER_SD_585_784_frames",
    "Khorfakkan/10-59-21_DEEP_TREKKER_SD_13260_13459_frames",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", action="append", default=None,
                   help=f"Clip directory (repeatable). Defaults to all three "
                        f"labeled Khorfakkan clips: {DEFAULT_CLIPS}")
    p.add_argument("--stage1_ckpt", default="models/bioreef_stage1.pt",
                   help="Stage 1 classifier checkpoint. Default: models/bioreef_stage1.pt")
    p.add_argument("--csv_path", default="frame_metadata.csv",
                   help="OzFish CSV (for species mapping fallback).")
    p.add_argument("--min_samples", type=int, default=20,
                   help="Species min-samples threshold for mapping fallback.")
    p.add_argument("--max_per_clip", type=int, default=None,
                   help="Cap the number of GT boxes tested per clip "
                        "(for a faster sanity run). Default: all.")
    p.add_argument("--device", default=None,
                   help="cuda or cpu. Default: auto.")
    return p.parse_args()


def find_images_dir(clip_dir: str) -> str:
    """Locate the raw images dir — handles both `images/` and `images/default/`."""
    for candidate in (os.path.join(clip_dir, "images"),
                      os.path.join(clip_dir, "images", "default")):
        if os.path.isdir(candidate) and any(
            f.endswith(".png") for f in os.listdir(candidate)
        ):
            return candidate
    raise FileNotFoundError(f"no images dir found under {clip_dir}")


def restored_path_for(images_dir: str) -> str:
    """Pick the parallel '_restored' directory name for a given images dir."""
    # If images_dir is 'foo/images', restored is 'foo/images_restored'.
    # If images_dir is 'foo/images/default', restored is 'foo/images_restored/default'.
    if os.path.basename(images_dir) == "default":
        parent = os.path.dirname(images_dir)  # foo/images
        grand = os.path.dirname(parent)        # foo
        return os.path.join(grand, "images_restored", "default")
    return images_dir + "_restored"


def ensure_restored(images_dir: str, restored_dir: str,
                    restorer: WaterNetRestorer) -> int:
    """Restore any frames missing from restored_dir. Returns count restored."""
    os.makedirs(restored_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    todo = [f for f in files if not os.path.exists(os.path.join(restored_dir, f))]
    if not todo:
        return 0

    print(f"  restoring {len(todo)} frames -> {restored_dir}")
    t0 = time.time()
    for i, fname in enumerate(todo):
        raw = cv2.imread(os.path.join(images_dir, fname))
        if raw is None:
            print(f"    warn: could not read {fname}", file=sys.stderr)
            continue
        cv2.imwrite(os.path.join(restored_dir, fname), restorer(raw))
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    [{i+1}/{len(todo)}] {rate:.2f} img/s", end="\r")
    print()
    return len(todo)


def load_clip(clip_dir: str) -> Tuple[str, str, List[Dict], Dict[int, str]]:
    """
    Returns (images_dir, restored_dir, annotations, image_id->filename).
    """
    gt_path = os.path.join(clip_dir, "annotations", "instances_default.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"no GT JSON at {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    images_dir = find_images_dir(clip_dir)
    restored_dir = restored_path_for(images_dir)
    id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    annotations = gt["annotations"]
    return images_dir, restored_dir, annotations, id_to_file


@torch.no_grad()
def classify_crop(
    frame_bgr: np.ndarray, bbox_xywh: List[float],
    backbone: ViTBackbone, mceam: MCEAM, head: nn.Module,
    harvester: ContextHarvester, device: torch.device,
) -> np.ndarray:
    """
    Run the full Stage 1 4-stream classifier on one fish crop.
    Returns the (C,) softmax probabilities.
    """
    x, y, w, h = bbox_xywh
    crops = harvester.harvest(frame_bgr, (int(x), int(y),
                                          max(int(w), 1), max(int(h), 1)))
    # crops is a dict {stream_name: tensor(3, 224, 224)}; batch dim of 1
    batched = {name: t.unsqueeze(0).to(device) for name, t in crops.items()}
    feats = backbone(batched)
    mceam_out = mceam(feats)
    fused = mceam_out["embedding"]  # (1, 256)
    logits = head(fused)            # (1, C)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def main() -> int:
    args = parse_args()
    clips = args.clip if args.clip else DEFAULT_CLIPS

    # Validate clips exist before loading any model
    for c in clips:
        if not os.path.isdir(c):
            print(f"error: clip dir not found: {c}", file=sys.stderr)
            return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    # --- Load classifier components ---
    print("loading backbone (DINOv3 ViT-B/16)...")
    backbone = ViTBackbone(freeze=True).to(device).eval()

    print(f"loading Stage 1 checkpoint: {args.stage1_ckpt}")
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    num_classes = ckpt["head"]["weight"].shape[0]
    idx_to_sp = resolve_species_mapping(ckpt, args.csv_path, args.min_samples)
    print(f"  {num_classes} classes")

    mceam = MCEAM(embed_dim=backbone.embed_dim, num_context_levels=3,
                  output_dim=256, num_heads=8).to(device).eval()
    mceam.load_state_dict(ckpt["mceam"])
    head = nn.Linear(256, num_classes).to(device).eval()
    head.load_state_dict(ckpt["head"])

    harvester = ContextHarvester()

    print("loading WaterNet (for on-demand frame restoration)...")
    restorer = WaterNetRestorer()
    restorer._load_model()

    # --- Per-clip stability metrics ---
    per_clip_stats: Dict[str, Dict] = {}

    for clip_dir in clips:
        print()
        print(f"=== {clip_dir} ===")
        images_dir, restored_dir, annotations, id_to_file = load_clip(clip_dir)
        print(f"  images   : {images_dir}")
        print(f"  restored : {restored_dir}")
        print(f"  GT boxes : {len(annotations)}")

        # Make sure restored frames exist
        ensure_restored(images_dir, restored_dir, restorer)

        # Group annotations by image to avoid re-reading the same frame
        by_image: Dict[int, List[Dict]] = defaultdict(list)
        for a in annotations:
            by_image[a["image_id"]].append(a)

        if args.max_per_clip:
            # Sample boxes across images, not all from the first few
            all_boxes = sum(([a for a in v] for v in by_image.values()), [])
            all_boxes = all_boxes[:args.max_per_clip]
            wanted_ids = {a["image_id"] for a in all_boxes}
            by_image = {k: [a for a in v if a in all_boxes]
                        for k, v in by_image.items() if k in wanted_ids}

        # Stats accumulators for this clip
        top1_agree = 0
        top5_agree = 0
        cos_sims: List[float] = []
        total = 0

        n_images = len(by_image)
        t0 = time.time()
        for img_idx, (image_id, boxes) in enumerate(by_image.items()):
            fname = id_to_file.get(image_id)
            if not fname:
                continue
            raw_path = os.path.join(images_dir, os.path.basename(fname))
            rest_path = os.path.join(restored_dir, os.path.basename(fname))
            if not (os.path.exists(raw_path) and os.path.exists(rest_path)):
                continue

            raw_frame = cv2.imread(raw_path)
            rest_frame = cv2.imread(rest_path)
            if raw_frame is None or rest_frame is None:
                continue

            for a in boxes:
                probs_raw = classify_crop(raw_frame, a["bbox"],
                                          backbone, mceam, head,
                                          harvester, device)
                probs_rest = classify_crop(rest_frame, a["bbox"],
                                           backbone, mceam, head,
                                           harvester, device)

                top1_raw = int(probs_raw.argmax())
                top1_rest = int(probs_rest.argmax())
                if top1_raw == top1_rest:
                    top1_agree += 1

                top5_raw = set(np.argsort(probs_raw)[-5:].tolist())
                top5_rest = set(np.argsort(probs_rest)[-5:].tolist())
                if top1_raw in top5_rest and top1_rest in top5_raw:
                    top5_agree += 1

                # Cosine sim of full distributions
                a_norm = probs_raw / (np.linalg.norm(probs_raw) + 1e-12)
                b_norm = probs_rest / (np.linalg.norm(probs_rest) + 1e-12)
                cos_sims.append(float(np.dot(a_norm, b_norm)))

                total += 1

            if (img_idx + 1) % 10 == 0 or (img_idx + 1) == n_images:
                rate = total / max(time.time() - t0, 1e-9)
                print(f"  [{img_idx+1}/{n_images} images, "
                      f"{total} crops, {rate:.1f} crops/s]", end="\r")

        print()
        if total == 0:
            print(f"  no crops processed for {clip_dir}")
            continue

        clip_stats = {
            "n_crops": total,
            "top1_agreement": top1_agree / total,
            "top5_agreement": top5_agree / total,
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "median_cosine_sim": float(np.median(cos_sims)),
        }
        per_clip_stats[clip_dir] = clip_stats
        print(f"  Top-1 agreement: {clip_stats['top1_agreement']:.1%}")
        print(f"  Top-5 agreement: {clip_stats['top5_agreement']:.1%}")
        print(f"  mean cosine sim: {clip_stats['mean_cosine_sim']:.3f}")

    # --- Aggregate report ---
    print()
    print("=" * 72)
    print("CLASSIFIER STABILITY: RAW vs WATERNET-RESTORED INPUTS")
    print("=" * 72)
    if not per_clip_stats:
        print("  no clips produced results")
        return 1

    header = f"{'clip':<55s} {'n':>5s} {'Top-1':>7s} {'Top-5':>7s} {'cos':>6s}"
    print(header)
    print("-" * len(header))
    total_crops = 0
    weighted_top1 = 0.0
    weighted_top5 = 0.0
    weighted_cos = 0.0
    for clip, s in per_clip_stats.items():
        short = clip[-52:] if len(clip) > 52 else clip
        print(f"{short:<55s} {s['n_crops']:>5d} "
              f"{s['top1_agreement']*100:>6.1f}% "
              f"{s['top5_agreement']*100:>6.1f}% "
              f"{s['mean_cosine_sim']:>6.3f}")
        total_crops += s["n_crops"]
        weighted_top1 += s["top1_agreement"] * s["n_crops"]
        weighted_top5 += s["top5_agreement"] * s["n_crops"]
        weighted_cos += s["mean_cosine_sim"] * s["n_crops"]

    print("-" * len(header))
    overall_top1 = weighted_top1 / total_crops
    overall_top5 = weighted_top5 / total_crops
    overall_cos = weighted_cos / total_crops
    print(f"{'WEIGHTED OVERALL':<55s} {total_crops:>5d} "
          f"{overall_top1*100:>6.1f}% {overall_top5*100:>6.1f}% {overall_cos:>6.3f}")
    print()
    print("INTERPRETATION:")
    if overall_top1 > 0.90:
        print(f"  → Top-1 agreement {overall_top1:.0%}: classifier is HIGHLY STABLE")
        print(f"    across the WaterNet/raw boundary. Dropping WaterNet from")
        print(f"    inference is safe — MCEAM has learned features robust to")
        print(f"    the input distribution shift. The expensive retrain on raw")
        print(f"    OzFish is unlikely to change much.")
    elif overall_top1 > 0.70:
        print(f"  → Top-1 agreement {overall_top1:.0%}: classifier is MODERATELY")
        print(f"    stable. Disagreements affect a non-trivial share of crops")
        print(f"    but most stay within the top-5. The retrain on raw OzFish")
        print(f"    might modestly improve consistency in deployment; whether")
        print(f"    it's worth 4 days of GPU depends on how much classifier")
        print(f"    quality matters relative to other gains.")
    else:
        print(f"  → Top-1 agreement {overall_top1:.0%}: classifier is SENSITIVE")
        print(f"    to the WaterNet/raw boundary. Predictions change for a")
        print(f"    substantial share of crops. The retrain on raw OzFish is")
        print(f"    justified: the current restored-trained classifier is")
        print(f"    misaligned with what production would feed it on raw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
