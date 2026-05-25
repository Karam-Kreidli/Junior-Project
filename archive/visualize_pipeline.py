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
from eval_pipeline import iou_xywh, classify_boxes


def display_species(name):
    """Cosmetic mapping of placeholder species labels to 'unidentified' for demo visuals."""
    if is_placeholder_species(name):
        return "unidentified"
    return name


def iomin_xywh(a, b):
    """Intersection over minimum area. Robust to large size mismatches between boxes
    (e.g., a tight detection fully inside a loose GT box returns 1.0)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(a[2] * a[3], b[2] * b[3])
    return inter / min_area if min_area > 0 else 0.0


def greedy_match_lenient(det_boxes, det_conf, gt_boxes, iou_thresh=0.5, iomin_thresh=0.7):
    """Like greedy_match, but a pair also counts as matched if intersection-over-min-area
    is high enough — handles the "tight detection inside loose GT" case where IoU
    underestimates the visual overlap.
    """
    used = set()
    matches = []
    order = np.argsort(-det_conf) if len(det_conf) else []
    for di in order:
        best_score = 0.0
        best_gt = -1
        for gi in range(len(gt_boxes)):
            if gi in used:
                continue
            iou = iou_xywh(det_boxes[di], gt_boxes[gi])
            iomin = iomin_xywh(det_boxes[di], gt_boxes[gi])
            score = max(iou / iou_thresh, iomin / iomin_thresh)  # 1.0 = passes either
            if score >= 1.0 and score > best_score:
                best_score = score
                best_gt = gi
        if best_gt >= 0:
            used.add(best_gt)
            matches.append((int(di), best_gt))
    return matches

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
    p.add_argument("--conf", type=float, nargs="+", default=[0.15],
                   help="Detector confidence threshold(s). Pass multiple values to generate "
                        "a side-by-side comparison: one subfolder per conf level, same frames.")
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
    p.add_argument("--max_box_frac", type=float, default=0.5,
                   help="Drop detections whose box area exceeds this fraction of the frame.")
    p.add_argument("--dedup_iomin", type=float, default=0.7,
                   help="Suppress an unmatched detection if it overlaps a matched one by IoMin >= this.")
    p.add_argument("--frames_dir", type=str, default=None,
                   help="If set: render every image in this directory at each --conf. "
                        "Skips test-split sampling. GT is matched by filename against the CSV split.")
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

    if args.frames_dir:
        # Re-render a specific set of images. Walks recursively, deduplicates by
        # filename. Matches each filename against the test split's GT (strips any
        # "NNN_" prefix added by the find_examples copy step).
        import re
        prefix_re = re.compile(r'^\d+_')
        by_basename = {os.path.basename(p): (p, gts) for p, gts in all_frames}
        seen_basenames = set()
        chosen = []
        skipped = 0
        for root, _dirs, files in os.walk(args.frames_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stripped = prefix_re.sub('', fname)
                if stripped in seen_basenames:
                    continue  # same image discovered in another category folder
                seen_basenames.add(stripped)
                if stripped in by_basename:
                    chosen.append(by_basename[stripped])
                else:
                    skipped += 1
        if not chosen:
            raise SystemExit(
                f"No images under {args.frames_dir} matched the test split. "
                f"Filenames must match the original test frames (with optional 'NNN_' prefix)."
            )
        logger.info(f"frames_dir mode: matched {len(chosen)} unique frames from {args.frames_dir} "
                    f"(skipped {skipped} non-matching files)")
    elif args.find_examples:
        chosen = all_frames
        logger.info(f"find_examples mode: scanning all {len(chosen)} frames")
    else:
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

    # --- Outer loop: one pass per confidence value ---
    multi_conf = len(args.conf) > 1

    for conf_val in args.conf:
        if multi_conf:
            conf_root = os.path.join(args.output_dir, f"conf_{conf_val}")
        else:
            conf_root = args.output_dir
        os.makedirs(conf_root, exist_ok=True)
        logger.info(f"\n=== Rendering at conf={conf_val} -> {conf_root}/ ===")

        # Track per-frame category counts for --find_examples mode.
        stats = {}

        for img_path, gt_list in tqdm(chosen, desc=f"render conf={conf_val}"):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            gt_boxes = [g['bbox'] for g in gt_list]
            gt_species = [g['species'] for g in gt_list]

            res = yolo(frame, conf=conf_val, verbose=False)[0].boxes
            if len(res) == 0:
                det_boxes, det_conf = np.empty((0, 4)), np.empty(0)
            else:
                xyxy = res.xyxy.cpu().numpy()
                det_boxes = np.stack(
                    [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]],
                    axis=1,
                )
                det_conf = res.conf.cpu().numpy()

            # Filter 1: drop detections covering too much of the frame.
            if len(det_boxes) > 0:
                h_img, w_img = frame.shape[:2]
                frame_area = h_img * w_img
                box_areas = det_boxes[:, 2] * det_boxes[:, 3]
                keep = box_areas / frame_area < args.max_box_frac
                det_boxes = det_boxes[keep]
                det_conf = det_conf[keep]

            pairs = greedy_match_lenient(det_boxes, det_conf, gt_boxes,
                                         iou_thresh=0.5, iomin_thresh=0.7)
            matched_gt_set = {gi for _, gi in pairs}
            matched_det_set = {di for di, _ in pairs}
            unmatched_det_idx = [di for di in range(len(det_boxes)) if di not in matched_det_set]

            # Filter 2: suppress unmatched detections that overlap a matched one.
            if pairs and unmatched_det_idx:
                matched_boxes_list = [det_boxes[di] for di, _ in pairs]
                unmatched_det_idx = [
                    di for di in unmatched_det_idx
                    if all(iomin_xywh(det_boxes[di], mb) < args.dedup_iomin
                           for mb in matched_boxes_list)
                ]

            # Filter 3: dedupe within the unmatched set.
            if len(unmatched_det_idx) > 1:
                unmatched_sorted = sorted(unmatched_det_idx, key=lambda di: -float(det_conf[di]))
                kept = []
                for di in unmatched_sorted:
                    if all(iomin_xywh(det_boxes[di], det_boxes[kj]) < args.dedup_iomin
                           for kj in kept):
                        kept.append(di)
                unmatched_det_idx = kept

            # Classify ALL detections (matched + unmatched) in one batch
            preds = {}
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

            # 1. Extras (drawn first so they sit behind verified boxes)
            for di in unmatched_det_idx:
                pred_sp, _ = preds[di]
                det_c = float(det_conf[di])
                draw_box(canvas, det_boxes[di], (255, 0, 200),
                         f"extra: {display_species(pred_sp)} ({det_c:.2f})")
                cat_count["extra"] += 1

            # 2. Missed GT in red.
            for gi, g in enumerate(gt_boxes):
                if gi not in matched_gt_set:
                    draw_box(canvas, g, (0, 0, 255), f"missed: {display_species(gt_species[gi])}")
                    cat_count["missed_gt"] += 1

            # 3. Verified matched detections last (top of z-order).
            for di, gi in pairs:
                pred_sp, pred_score = preds[di]
                gt_sp = gt_species[gi]
                det_c = float(det_conf[di])
                pred_disp = display_species(pred_sp)
                gt_disp = display_species(gt_sp)

                if pred_sp == gt_sp:
                    color = (0, 200, 0)
                    label = f"{pred_disp} ({det_c:.2f})"
                    cat_count["correct_species"] += 1
                else:
                    pred_gen = taxonomy.get(pred_sp, (None,))[0]
                    gt_gen = taxonomy.get(gt_sp, (None,))[0]
                    if pred_gen is not None and pred_gen == gt_gen:
                        color = (255, 200, 0)
                        label = f"{pred_disp} -> {gt_disp} (same genus)"
                        cat_count["correct_genus"] += 1
                    else:
                        color = (0, 140, 255)
                        label = f"{pred_disp} (GT: {gt_disp})"
                        cat_count["wrong_genus"] += 1

                draw_box(canvas, det_boxes[di], color, label)

            draw_legend(canvas, conf_val)

            out_name = os.path.basename(img_path)
            if args.find_examples:
                staging = os.path.join(conf_root, "_all")
                os.makedirs(staging, exist_ok=True)
                cv2.imwrite(os.path.join(staging, out_name), canvas)
                stats[img_path] = (cat_count, os.path.join(staging, out_name))
            else:
                cv2.imwrite(os.path.join(conf_root, out_name), canvas)

        # --- Pick best examples per category for THIS conf ---
        if args.find_examples:
            for cat in ("correct_species", "correct_genus", "wrong_genus", "missed_gt", "extra"):
                cat_dir = os.path.join(conf_root, cat)
                os.makedirs(cat_dir, exist_ok=True)
                ranked = sorted(stats.items(), key=lambda it: it[1][0][cat], reverse=True)
                picked = 0
                for img_path, (counts, rendered) in ranked:
                    if counts[cat] == 0:
                        break
                    out_path = os.path.join(cat_dir, f"{counts[cat]:03d}_{os.path.basename(img_path)}")
                    import shutil
                    shutil.copy2(rendered, out_path)
                    picked += 1
                    if picked >= args.examples_per_category:
                        break
                logger.info(f"  conf={conf_val} {cat}: saved {picked} examples to {cat_dir}")

    if args.find_examples:
        print(f"\nExamples organized into subfolders under {args.output_dir}/")
    else:
        total = len(chosen) * len(args.conf)
        print(f"\nWrote {total} annotated frames to {args.output_dir}/")


if __name__ == "__main__":
    main()
