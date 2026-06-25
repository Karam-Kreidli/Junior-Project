"""
Render side-by-side overlays for visual detector comparison.

For each frame, draws:
    GROUND TRUTH boxes in green
    PRED A boxes — color-coded TP (cyan) / FP (red, with "!" marker)
    PRED B boxes — color-coded TP (cyan) / FP (red, with "!" marker)
    Missed GT (no matching prediction at IoU>=thresh) — green dashed-style

Output is one composite PNG per frame: [GT | Pred A | Pred B] panels.

Usage:
    python compare_overlays.py \\
        --pred outputs/rfdetr_cfd/predictions.json \\
        --pred outputs/yolo_baseline/predictions.json \\
        --label RF-DETR --label YOLO

    # Single backend (just GT vs one prediction set)
    python compare_overlays.py --pred outputs/rfdetr_cfd/predictions.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# BGR/RGB convention: PIL uses RGB
COLOR_GT = (40, 220, 60)     # green
COLOR_TP = (60, 200, 240)    # cyan
COLOR_FP = (240, 70, 50)     # red
COLOR_FN_OVERLAY = (40, 220, 60)  # green for the missed GT marker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", default="annotations/instances_default.json")
    p.add_argument("--images", default="annotations/images",
                   help="Directory of frame PNGs (must match GT file_names).")
    p.add_argument("--pred", action="append", required=True,
                   help="predictions.json (repeat for multiple backends).")
    p.add_argument("--label", action="append", default=[],
                   help="Display name for each --pred. Defaults to parent folder.")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for TP vs FP. Default: 0.5.")
    p.add_argument("--score_thresh", type=float, default=0.05,
                   help="Drop predictions below this score before drawing.")
    p.add_argument("--out_dir", default="outputs/compare",
                   help="Where to write composite overlay PNGs.")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Only render this many frames (for a quick look).")
    p.add_argument("--only_disagreements", action="store_true",
                   help="Only render frames where backends disagree "
                        "(different TP/FP/FN counts).")
    return p.parse_args()


def label_for(pred_path: str, override: str | None) -> str:
    return override or os.path.basename(os.path.dirname(os.path.abspath(pred_path))) or pred_path


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    union = aw[:, None] * ah[:, None] + bw[None, :] * bh[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_preds_to_gt(
    preds: List[dict], gt_boxes: np.ndarray, iou_thresh: float,
) -> Tuple[List[bool], List[bool]]:
    """
    Greedy matching, predictions sorted by score (high first).
    Returns (is_tp per prediction, was_matched per GT).
    """
    if len(preds) == 0:
        return [], [False] * len(gt_boxes)
    if len(gt_boxes) == 0:
        return [False] * len(preds), []

    pr_boxes = np.array([p["bbox"] for p in preds], dtype=float)
    iou = iou_xywh(pr_boxes, gt_boxes)

    is_tp = [False] * len(preds)
    gt_taken = [False] * len(gt_boxes)
    # Predictions are already passed in score-sorted order
    for pi in range(len(preds)):
        best_g = -1
        best_iou = iou_thresh
        for gi in range(len(gt_boxes)):
            if gt_taken[gi]:
                continue
            if iou[pi, gi] >= best_iou:
                best_iou = iou[pi, gi]
                best_g = gi
        if best_g >= 0:
            gt_taken[best_g] = True
            is_tp[pi] = True
    return is_tp, gt_taken


def draw_panel(
    img: Image.Image, title: str,
    gt_boxes: np.ndarray, gt_matched: List[bool],
    preds: List[dict] | None, is_tp: List[bool] | None,
    iou_thresh: float, draw_font: ImageFont.ImageFont,
) -> Image.Image:
    """
    Draw one panel: image + GT boxes (green) + optional pred boxes
    (cyan TP / red FP).  GT not matched gets a thicker stroke + "MISS" tag.
    """
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")

    # GT (green). Missed GT gets thicker stroke + label.
    for gi, box in enumerate(gt_boxes):
        x, y, w, h = box
        missed = preds is not None and not gt_matched[gi]
        width = 4 if missed else 2
        d.rectangle([x, y, x + w, y + h], outline=COLOR_GT, width=width)
        if missed:
            tag = "MISS"
            ts = d.textbbox((0, 0), tag, font=draw_font)
            tw, th = ts[2] - ts[0], ts[3] - ts[1]
            d.rectangle([x, max(y - th - 4, 0), x + tw + 6, max(y, th + 4)],
                        fill=COLOR_GT)
            d.text((x + 3, max(y - th - 2, 2)), tag, fill=(0, 0, 0), font=draw_font)

    # Predictions (cyan TP, red FP)
    if preds is not None:
        for pi, p in enumerate(preds):
            x, y, w, h = p["bbox"]
            tp = is_tp[pi]
            color = COLOR_TP if tp else COLOR_FP
            d.rectangle([x, y, x + w, y + h], outline=color, width=2)
            tag = f"{'TP' if tp else 'FP'} {p.get('score', 0):.2f}"
            ts = d.textbbox((0, 0), tag, font=draw_font)
            tw, th = ts[2] - ts[0], ts[3] - ts[1]
            d.rectangle([x, y + h, x + tw + 6, y + h + th + 4], fill=color)
            d.text((x + 3, y + h + 2), tag, fill=(0, 0, 0), font=draw_font)

    # Panel header
    hd = f"{title}"
    ts = d.textbbox((0, 0), hd, font=draw_font)
    tw, th = ts[2] - ts[0], ts[3] - ts[1]
    d.rectangle([0, 0, tw + 12, th + 8], fill=(0, 0, 0))
    d.text((6, 4), hd, fill=(255, 255, 255), font=draw_font)
    return out


def compose_panels(panels: List[Image.Image], scale: float = 0.5) -> Image.Image:
    """Side-by-side horizontal composite, optionally downscaled."""
    panels = [p.resize((int(p.width * scale), int(p.height * scale)),
                       Image.LANCZOS) for p in panels]
    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width
    return out


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.gt):
        print(f"error: GT not found: {args.gt}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.images):
        print(f"error: images dir not found: {args.images}", file=sys.stderr)
        return 2

    # Font — use a fallback path so it works on any OS
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # GT
    with open(args.gt, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    file_by_id: Dict[int, str] = {im["id"]: im["file_name"] for im in gt_data["images"]}
    gt_by_img: Dict[int, List[dict]] = {}
    for a in gt_data["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)

    # Predictions — one dict per backend
    labels = list(args.label) + [None] * (len(args.pred) - len(args.label))
    backends: List[Tuple[str, Dict[int, List[dict]]]] = []
    for pred_path, lbl in zip(args.pred, labels):
        if not os.path.exists(pred_path):
            print(f"error: predictions not found: {pred_path}", file=sys.stderr)
            return 2
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        by_img: Dict[int, List[dict]] = {}
        for p in preds:
            if p.get("score", 1.0) < args.score_thresh:
                continue
            by_img.setdefault(p["image_id"], []).append(p)
        # sort each frame by score
        for k in by_img:
            by_img[k].sort(key=lambda p: -p.get("score", 0))
        backends.append((label_for(pred_path, lbl), by_img))

    os.makedirs(args.out_dir, exist_ok=True)

    # Image IDs we care about: those with GT or any prediction
    image_ids = set(gt_by_img)
    for _, by_img in backends:
        image_ids |= set(by_img)
    image_ids = sorted(image_ids)

    rendered = 0
    skipped_disagree = 0
    for img_id in image_ids:
        if args.max_frames is not None and rendered >= args.max_frames:
            break
        fname = file_by_id.get(img_id)
        if not fname:
            continue
        img_path = os.path.join(args.images, fname)
        if not os.path.exists(img_path):
            print(f"  skip (image missing on disk): {fname}", file=sys.stderr)
            continue

        gt_boxes = np.array([a["bbox"] for a in gt_by_img.get(img_id, [])], dtype=float)

        # Per-backend matching
        per_backend = []
        signatures = []
        for name, by_img in backends:
            preds = by_img.get(img_id, [])
            is_tp, gt_matched = match_preds_to_gt(preds, gt_boxes, args.iou)
            tp = sum(is_tp); fp = len(is_tp) - tp; fn = sum(1 for m in gt_matched if not m)
            per_backend.append((name, preds, is_tp, gt_matched, tp, fp, fn))
            signatures.append((tp, fp, fn))

        if args.only_disagreements and len(signatures) >= 2 and len(set(signatures)) == 1:
            skipped_disagree += 1
            continue

        img = Image.open(img_path).convert("RGB")
        panels = [
            draw_panel(img, f"GT  ({len(gt_boxes)})", gt_boxes,
                       [True] * len(gt_boxes), None, None, args.iou, font)
        ]
        for (name, preds, is_tp, gt_matched, tp, fp, fn) in per_backend:
            title = f"{name}  TP={tp} FP={fp} FN={fn}"
            panels.append(draw_panel(img, title, gt_boxes, gt_matched,
                                     preds, is_tp, args.iou, font))

        composite = compose_panels(panels, scale=0.5)
        composite.save(os.path.join(args.out_dir, fname))
        rendered += 1

    print(f"rendered {rendered} composites -> {args.out_dir}")
    if args.only_disagreements:
        print(f"skipped {skipped_disagree} agreement frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
