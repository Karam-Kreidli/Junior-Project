"""
BioReef.ai — Detector Label Audit
===================================
Finds label disagreements between the current detector and existing YOLO
labels to guide manual cleanup.

For each image listed in <split>.txt:
    1. Run best.pt at HIGH confidence (default 0.7) — "what does the
       detector say with high confidence"
    2. Load the existing YOLO label file
    3. Match detections <-> labels via IoU >= 0.5
    4. Flag two categories:
        - MISSED_LABEL: high-conf detection with no matching GT box
                        -> probably a fish the annotator missed
        - NO_DETECTION: GT box with no matching detection
                        -> possibly bad label OR legitimate detector miss

Output:
    - <out>/audit_candidates.csv : one row per candidate with frame, bbox,
                                    category, confidence
    - <out>/summary.txt          : counts per category
    - <out>/viz/*.png            : rendered frames with overlays
        * green  = existing GT
        * red    = MISSED_LABEL (proposed addition)
        * yellow = NO_DETECTION (existing GT that detector did not find)

Usage:
    python audit_labels.py \\
        --detection_ckpt runs/detect/trainX/weights/best.pt \\
        --dataset_dir datasets/ozfish \\
        --split train \\
        --conf 0.7 \\
        --output_dir audit_out
"""

import argparse
import csv
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("audit")


# =============================================================================
# Path + label helpers
# =============================================================================

def image_to_label_path(img_path: str) -> str:
    """Ultralytics convention: swap last 'images' component with 'labels',
    swap extension with .txt."""
    parts = img_path.split(os.sep)
    # find last 'images' component
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    stem, _ = os.path.splitext(parts[-1])
    parts[-1] = stem + ".txt"
    return os.sep.join(parts)


def load_yolo_labels(label_path: str, img_w: int, img_h: int):
    """Returns list of [x, y, w, h] in absolute pixel coords."""
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


# =============================================================================
# Geometry
# =============================================================================

def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_pairs(dets, gts, iou_thresh=0.5):
    """Returns (matched_det_idx, matched_gt_idx) sets."""
    matched_det, matched_gt = set(), set()
    # Greedy, IoU-max
    pairs = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            pairs.append((iou_xywh(d, g), di, gi))
    pairs.sort(reverse=True)
    for iou, di, gi in pairs:
        if iou < iou_thresh:
            break
        if di in matched_det or gi in matched_gt:
            continue
        matched_det.add(di)
        matched_gt.add(gi)
    return matched_det, matched_gt


# =============================================================================
# Visualization
# =============================================================================

def draw_audit(frame, gt_boxes, missed_det_boxes, unmatched_gt_idx, output_path):
    """Render color-coded overlay for manual review."""
    canvas = frame.copy()
    # Existing GT (green) / unmatched GT (yellow)
    for gi, g in enumerate(gt_boxes):
        color = (0, 255, 255) if gi in unmatched_gt_idx else (0, 200, 0)
        x, y, w, h = map(int, g)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    # Missed labels (red)
    for d in missed_det_boxes:
        x, y, w, h, c = d
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(canvas, f"{c:.2f}", (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_path, canvas)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detection_ckpt", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, default="datasets/ozfish")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--conf", type=float, default=0.7,
                   help="Detector confidence above which an unmatched detection is flagged.")
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="audit_out")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Stop after this many frames (for quick dry-runs).")
    p.add_argument("--save_viz", action="store_true",
                   help="Save annotated images to <output>/viz/ (one per frame with a candidate).")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    split_txt = os.path.join(args.dataset_dir, f"{args.split}.txt")
    if not os.path.exists(split_txt):
        raise FileNotFoundError(f"Split list not found: {split_txt}")

    with open(split_txt) as f:
        img_paths = [ln.strip() for ln in f if ln.strip()]
    if args.max_frames:
        img_paths = img_paths[:args.max_frames]

    logger.info(f"Loaded {len(img_paths)} frames from {split_txt}")
    logger.info(f"Loading YOLO: {args.detection_ckpt}")
    yolo = YOLO(args.detection_ckpt)

    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "viz")
    if args.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "audit_candidates.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["frame_path", "category", "x", "y", "w", "h", "conf"])

    counts = defaultdict(int)
    frames_with_issues = 0

    for img_path in tqdm(img_paths, desc="auditing"):
        frame = cv2.imread(img_path)
        if frame is None:
            counts["unreadable"] += 1
            continue
        h_img, w_img = frame.shape[:2]

        gt_boxes = load_yolo_labels(image_to_label_path(img_path), w_img, h_img)

        res = yolo(frame, conf=args.conf, verbose=False)[0].boxes
        if len(res) == 0:
            det_boxes, det_conf = [], []
        else:
            xyxy = res.xyxy.cpu().numpy()
            det_boxes = np.stack(
                [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]],
                axis=1,
            ).tolist()
            det_conf = res.conf.cpu().numpy().tolist()

        matched_det, matched_gt = match_pairs(det_boxes, gt_boxes, args.iou_thresh)
        missed_det_idx = [i for i in range(len(det_boxes)) if i not in matched_det]
        unmatched_gt_idx = [i for i in range(len(gt_boxes)) if i not in matched_gt]

        frame_had_issue = False
        missed_for_viz = []

        for di in missed_det_idx:
            x, y, w, h = det_boxes[di]
            c = det_conf[di]
            writer.writerow([img_path, "MISSED_LABEL", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{c:.3f}"])
            missed_for_viz.append((x, y, w, h, c))
            counts["MISSED_LABEL"] += 1
            frame_had_issue = True

        for gi in unmatched_gt_idx:
            x, y, w, h = gt_boxes[gi]
            writer.writerow([img_path, "NO_DETECTION", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", ""])
            counts["NO_DETECTION"] += 1
            frame_had_issue = True

        if frame_had_issue:
            frames_with_issues += 1
            if args.save_viz:
                out_name = os.path.basename(img_path)
                draw_audit(frame, gt_boxes, missed_for_viz, set(unmatched_gt_idx),
                           os.path.join(viz_dir, out_name))

    csv_f.close()

    # Summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Frames audited     : {len(img_paths)}\n")
        f.write(f"Frames with issues : {frames_with_issues}\n")
        f.write(f"MISSED_LABEL       : {counts['MISSED_LABEL']}\n")
        f.write(f"NO_DETECTION       : {counts['NO_DETECTION']}\n")
        f.write(f"Unreadable frames  : {counts['unreadable']}\n")
        f.write(f"Detector conf used : {args.conf}\n")
        f.write(f"IoU threshold      : {args.iou_thresh}\n")

    print("\n" + "=" * 60)
    print(f"Frames audited     : {len(img_paths)}")
    print(f"Frames with issues : {frames_with_issues}")
    print(f"MISSED_LABEL       : {counts['MISSED_LABEL']}  (detector found, no GT)")
    print(f"NO_DETECTION       : {counts['NO_DETECTION']}  (GT exists, detector missed)")
    print("=" * 60)
    print(f"Candidates written : {csv_path}")
    print(f"Summary written    : {summary_path}")
    if args.save_viz:
        print(f"Visualizations     : {viz_dir}/")


if __name__ == "__main__":
    main()
