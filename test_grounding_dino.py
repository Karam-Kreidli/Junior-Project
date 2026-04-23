"""
BioReef.ai — Grounding DINO Zero-Shot Fish Detection Test
===========================================================
Runs Grounding DINO (open-vocabulary detector, HuggingFace) on a sample of
OzFish frames with the prompt "fish" and compares against existing YOLO
labels. No training, no fine-tuning — pure zero-shot.

Purpose: verify whether an off-the-shelf detector, without any OzFish
training, catches the missed fish our trained detector (and the dataset
labels) both miss. If yes -> use it for pseudo-labeling. If no -> the
domain is genuinely hard.

Output:
    <out>/viz/*.png : annotated frames
        * green = existing YOLO GT boxes
        * red   = Grounding DINO detections (zero-shot, prompt="fish")
    <out>/counts.csv : per-frame [gt_count, gdino_count, overlap, gdino_extra]

Usage:
    python test_grounding_dino.py \\
        --dataset_dir datasets/ozfish \\
        --split train \\
        --model IDEA-Research/grounding-dino-tiny \\
        --prompt "a fish." \\
        --max_frames 10 \\
        --output_dir gdino_out
"""

import argparse
import csv
import logging
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gdino")


# =============================================================================
# Path + label helpers (duplicated from audit_labels.py so this is standalone)
# =============================================================================

def image_to_label_path(img_path: str) -> str:
    parts = img_path.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    stem, _ = os.path.splitext(parts[-1])
    parts[-1] = stem + ".txt"
    return os.sep.join(parts)


def load_yolo_labels(label_path: str, img_w: int, img_h: int):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _cls, cx, cy, bw, bh = map(float, parts[:5])
            x = (cx - bw / 2) * img_w
            y = (cy - bh / 2) * img_h
            boxes.append([x, y, bw * img_w, bh * img_h])
    return boxes


def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def count_matched(dets, gts, iou_thresh=0.5):
    """How many dets overlap a GT by >= iou_thresh (greedy)."""
    used_gt = set()
    matched = 0
    pairs = [(iou_xywh(d, g), di, gi) for di, d in enumerate(dets) for gi, g in enumerate(gts)]
    pairs.sort(reverse=True)
    used_det = set()
    for iou, di, gi in pairs:
        if iou < iou_thresh:
            break
        if di in used_det or gi in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gi)
        matched += 1
    return matched


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default="datasets/ozfish")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--prompt", type=str, default="a fish.")
    p.add_argument("--box_threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument("--max_frames", type=int, default=10)
    p.add_argument("--random_sample", action="store_true",
                   help="Pick frames randomly instead of the first N.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="gdino_out")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ---- Load Grounding DINO via HuggingFace transformers ----
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        raise SystemExit(
            "transformers not installed. Run: pip install transformers>=4.37"
        )

    logger.info(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    # ---- Select frames ----
    split_txt = os.path.join(args.dataset_dir, f"{args.split}.txt")
    with open(split_txt) as f:
        img_paths = [ln.strip() for ln in f if ln.strip()]

    if args.random_sample:
        random.seed(args.seed)
        img_paths = random.sample(img_paths, min(args.max_frames, len(img_paths)))
    else:
        img_paths = img_paths[:args.max_frames]

    logger.info(f"Testing on {len(img_paths)} frames with prompt: {args.prompt!r}")

    # ---- Output ----
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "counts.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "gt_count", "gdino_count", "matched", "gdino_extra", "gt_missing_from_gdino"])

    # ---- Run ----
    total_gt = total_gdino = total_matched = 0

    for img_path in tqdm(img_paths, desc="gdino"):
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        gt_boxes = load_yolo_labels(image_to_label_path(img_path), w, h)

        # Grounding DINO forward
        inputs = processor(images=pil, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )[0]

        # xyxy -> xywh
        gdino_boxes = []
        gdino_scores = []
        for box, score in zip(results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()):
            x1, y1, x2, y2 = box
            gdino_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            gdino_scores.append(float(score))

        matched = count_matched(gdino_boxes, gt_boxes, iou_thresh=0.5)
        gdino_extra = len(gdino_boxes) - matched
        gt_missing = len(gt_boxes) - matched

        total_gt += len(gt_boxes)
        total_gdino += len(gdino_boxes)
        total_matched += matched

        writer.writerow([
            os.path.basename(img_path), len(gt_boxes), len(gdino_boxes),
            matched, gdino_extra, gt_missing,
        ])

        # Visualize
        canvas = frame_bgr.copy()
        for g in gt_boxes:
            x, y, bw, bh = map(int, g)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        for b, s in zip(gdino_boxes, gdino_scores):
            x, y, bw, bh = map(int, b)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(canvas, f"{s:.2f}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(viz_dir, os.path.basename(img_path)), canvas)

    csv_f.close()

    print("\n" + "=" * 60)
    print(f"Frames tested     : {len(img_paths)}")
    print(f"Prompt            : {args.prompt!r}")
    print(f"box_threshold     : {args.box_threshold}")
    print(f"text_threshold    : {args.text_threshold}")
    print(f"Total GT boxes    : {total_gt}")
    print(f"Total GDINO boxes : {total_gdino}")
    print(f"GDINO matched GT  : {total_matched} / {total_gt} ({(total_matched / max(1, total_gt) * 100):.1f}%)")
    print(f"GDINO extra (not in GT) : {total_gdino - total_matched}")
    print(f"GT not found by GDINO   : {total_gt - total_matched}")
    print("=" * 60)
    print(f"Viz : {viz_dir}/")
    print(f"CSV : {csv_path}")
    print()
    print("What to look for in viz/:")
    print("  - RED boxes not overlapping GREEN = GDINO found fish the labels missed")
    print("  - If there are many clear fish with no red OR green box, GDINO is also missing them")


if __name__ == "__main__":
    main()
