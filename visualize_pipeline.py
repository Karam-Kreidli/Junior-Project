"""
BioReef.ai — Pipeline Visualization (for presentations / demos)
================================================================
Runs detector -> MCEAM on sampled test frames and produces annotated images
showing detections, predicted species, and correctness against original GT.

Color key:
    GREEN   : detected + correct species (Top-1 match)
    CYAN    : detected + correct genus (species wrong, genus right)
    ORANGE  : detected + wrong species (and wrong genus)
    RED     : GT fish that the detector missed entirely

Each detection is labeled with: `species (conf)` where conf is detector confidence.
GT fish (missed or matched) show the original species name.

Usage:
    python visualize_pipeline.py \\
        --detection_ckpt runs/detect/train4/weights/best.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --csv_path Junior-Project/frame_metadata.csv \\
        --split test \\
        --conf 0.15 \\
        --num_frames 30 \\
        --output_dir demo_viz
"""

import argparse
import logging
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester

from train_stage1 import split_dataset, get_taxonomy_tree, is_placeholder_species
from eval_pipeline import iou_xywh, greedy_match, classify_boxes


def display_species(name):
    """Cosmetic mapping of placeholder species labels to 'unidentified' for demo visuals."""
    if is_placeholder_species(name):
        return "unidentified"
    return name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("viz")


# =============================================================================
# Taxonomy
# =============================================================================

def build_taxonomy_lookup(csv_path):
    """species -> (genus, family). Used for genus-level correctness coloring."""
    import pandas as pd
    df = pd.read_csv(csv_path).dropna(subset=['species', 'genus', 'family'])
    return {row['species']: (row['genus'], row['family'])
            for _, row in df.iterrows()}


# =============================================================================
# Drawing
# =============================================================================

def draw_box(canvas, box_xywh, color, label, thickness=2, font_scale=0.6):
    x, y, w, h = map(int, box_xywh)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    # Label background for readability
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    lbl_y = max(y - 4, th + 4)
    cv2.rectangle(canvas, (x, lbl_y - th - 4), (x + tw + 4, lbl_y + baseline - 2), color, -1)
    cv2.putText(canvas, label, (x + 2, lbl_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_legend(canvas, conf):
    h, w = canvas.shape[:2]
    legend = [
        ((0, 200, 0),     "correct species"),
        ((255, 200, 0),   "correct genus / wrong species"),
        ((0, 140, 255),   "wrong genus"),
        ((0, 0, 255),     "missed by detector"),
        ((255, 0, 200),   "detected, no GT label (extra)"),
    ]
    box_h = 26
    pad = 8
    x0, y0 = pad, h - len(legend) * box_h - pad
    for i, (color, txt) in enumerate(legend):
        y = y0 + i * box_h
        cv2.rectangle(canvas, (x0, y), (x0 + 20, y + box_h - 6), color, -1)
        cv2.putText(canvas, txt, (x0 + 28, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0 + 28, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"conf={conf}", (pad, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"conf={conf}", (pad, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detection_ckpt", type=str, required=True)
    p.add_argument("--stage1_ckpt", type=str, default="bioreef_stage1.pt")
    p.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    p.add_argument("--img_dir", type=str, default="data_oz/frames_waternet_1")
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--num_frames", type=int, default=30,
                   help="How many frames to annotate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="demo_viz")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min_gt_per_frame", type=int, default=2,
                   help="Prefer frames with at least this many GT fish (more interesting visuals).")
    p.add_argument("--find_examples", action="store_true",
                   help="Scan ALL frames and pick the clearest examples of each category.")
    p.add_argument("--examples_per_category", type=int, default=5,
                   help="When --find_examples is set, how many best frames to keep per category.")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Split ---
    # filter_placeholders=False matches the current checkpoint.
    _, val_samples, test_samples, num_classes, idx_to_sp, _ = split_dataset(
        args.csv_path, args.img_dir, min_samples=args.min_samples,
        filter_placeholders=False,
    )
    sp_to_idx = {sp: i for i, sp in idx_to_sp.items()}
    eval_samples = val_samples if args.split == "val" else test_samples

    by_frame = defaultdict(list)
    for s in eval_samples:
        by_frame[s['img_path']].append({'bbox': s['bbox'], 'species': s['species']})

    all_frames = list(by_frame.items())

    if args.find_examples:
        # Scan every frame; selection happens after processing.
        chosen = all_frames
        logger.info(f"find_examples mode: scanning all {len(chosen)} frames")
    else:
        # Prefer frames with multiple GT fish for a more interesting demo
        rich_frames = [f for f in all_frames if len(f[1]) >= args.min_gt_per_frame]
        if len(rich_frames) < args.num_frames:
            rich_frames = all_frames
        rng = random.Random(args.seed)
        chosen = rng.sample(rich_frames, min(args.num_frames, len(rich_frames)))
        logger.info(f"Selected {len(chosen)} frames (>= {args.min_gt_per_frame} GT fish each)")

    # --- Load models ---
    logger.info(f"Loading YOLO: {args.detection_ckpt}")
    yolo = YOLO(args.detection_ckpt)

    backbone = ViTBackbone(freeze=True).to(device).eval()
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    mceam = MCEAM(
        embed_dim=backbone.embed_dim, num_context_levels=3, output_dim=256, num_heads=8,
    ).to(device).eval()
    mceam.load_state_dict(ckpt['mceam'])
    head = nn.Linear(256, num_classes).to(device).eval()
    head.load_state_dict(ckpt['head'])

    harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
    taxonomy = build_taxonomy_lookup(args.csv_path)

    # Track per-frame category counts for --find_examples mode.
    # stats[img_path] = {"correct_species": int, "correct_genus": int,
    #                    "wrong_genus": int, "missed_gt": int, "extra": int}
    stats = {}

    # --- Process ---
    for img_path, gt_list in tqdm(chosen, desc="render"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        gt_boxes = [g['bbox'] for g in gt_list]
        gt_species = [g['species'] for g in gt_list]

        res = yolo(frame, conf=args.conf, verbose=False)[0].boxes
        if len(res) == 0:
            det_boxes, det_conf = np.empty((0, 4)), np.empty(0)
        else:
            xyxy = res.xyxy.cpu().numpy()
            det_boxes = np.stack(
                [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]],
                axis=1,
            )
            det_conf = res.conf.cpu().numpy()

        pairs = greedy_match(det_boxes, det_conf, gt_boxes, iou_thresh=0.5)
        matched_gt_set = {gi for _, gi in pairs}
        matched_det_set = {di for di, _ in pairs}
        unmatched_det_idx = [di for di in range(len(det_boxes)) if di not in matched_det_set]

        # Classify ALL detections (matched + unmatched) in one batch
        preds = {}  # det_idx -> (species, score)
        all_det_idx = [di for di, _ in pairs] + unmatched_det_idx
        if all_det_idx:
            all_boxes = np.array([det_boxes[di] for di in all_det_idx])
            scores = classify_boxes(backbone, mceam, head, harvester, frame, all_boxes, device)
            top1_idx = scores.argmax(axis=1)
            for i, di in enumerate(all_det_idx):
                preds[di] = (idx_to_sp[int(top1_idx[i])], float(scores[i, top1_idx[i]]))

        canvas = frame.copy()
        cat_count = {"correct_species": 0, "correct_genus": 0,
                     "wrong_genus": 0, "missed_gt": 0, "extra": 0}

        # Draw order: bottom to top — extras first so verified boxes render on top.

        # 1. Extras (purely detector output, no GT) — draw first so they sit behind.
        for di in unmatched_det_idx:
            pred_sp, _ = preds[di]
            det_c = float(det_conf[di])
            draw_box(canvas, det_boxes[di], (255, 0, 200), f"extra: {display_species(pred_sp)} ({det_c:.2f})")
            cat_count["extra"] += 1

        # 2. Missed GT in red.
        for gi, g in enumerate(gt_boxes):
            if gi not in matched_gt_set:
                draw_box(canvas, g, (0, 0, 255), f"missed: {display_species(gt_species[gi])}")
                cat_count["missed_gt"] += 1

        # 3. Matched (verified) detections — drawn LAST so they appear on top.
        for di, gi in pairs:
            pred_sp, pred_score = preds[di]
            gt_sp = gt_species[gi]
            det_c = float(det_conf[di])
            pred_disp = display_species(pred_sp)
            gt_disp = display_species(gt_sp)

            if pred_sp == gt_sp:
                color = (0, 200, 0)  # green
                label = f"{pred_disp} ({det_c:.2f})"
                cat_count["correct_species"] += 1
            else:
                pred_gen = taxonomy.get(pred_sp, (None,))[0]
                gt_gen = taxonomy.get(gt_sp, (None,))[0]
                if pred_gen is not None and pred_gen == gt_gen:
                    color = (255, 200, 0)  # cyan-ish (BGR)
                    label = f"{pred_disp} -> {gt_disp} (same genus)"
                    cat_count["correct_genus"] += 1
                else:
                    color = (0, 140, 255)  # orange
                    label = f"{pred_disp} (GT: {gt_disp})"
                    cat_count["wrong_genus"] += 1

            draw_box(canvas, det_boxes[di], color, label)

        draw_legend(canvas, args.conf)

        out_name = os.path.basename(img_path)
        if args.find_examples:
            # Save every rendered frame to a staging dir and remember counts.
            staging = os.path.join(args.output_dir, "_all")
            os.makedirs(staging, exist_ok=True)
            cv2.imwrite(os.path.join(staging, out_name), canvas)
            stats[img_path] = (cat_count, os.path.join(staging, out_name))
        else:
            cv2.imwrite(os.path.join(args.output_dir, out_name), canvas)

    # --- Pick best examples per category ---
    if args.find_examples:
        for cat in ("correct_species", "correct_genus", "wrong_genus", "missed_gt", "extra"):
            cat_dir = os.path.join(args.output_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)
            ranked = sorted(
                stats.items(), key=lambda it: it[1][0][cat], reverse=True,
            )
            picked = 0
            for img_path, (counts, rendered) in ranked:
                if counts[cat] == 0:
                    break
                # Copy into category folder (prefix with count so sorting is by quality)
                out_path = os.path.join(cat_dir, f"{counts[cat]:03d}_{os.path.basename(img_path)}")
                import shutil
                shutil.copy2(rendered, out_path)
                picked += 1
                if picked >= args.examples_per_category:
                    break
            logger.info(f"  {cat}: saved {picked} examples to {cat_dir}")
        print(f"\nExamples organized into subfolders under {args.output_dir}/")
    else:
        print(f"\nWrote {len(chosen)} annotated frames to {args.output_dir}/")


if __name__ == "__main__":
    main()
