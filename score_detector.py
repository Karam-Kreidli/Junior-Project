"""
Score a detector's COCO-format predictions against the GT.

Computes COCO mAP (mAP@.5:.95, mAP@.5, mAP@.75) plus precision / recall
and TP/FP/FN counts at an IoU threshold of your choice.

Usage:
    # Score one backend
    python score_detector.py --pred outputs/rfdetr_cfd/predictions.json
    python score_detector.py --pred outputs/yolo_baseline/predictions.json

    # Or score multiple at once for a side-by-side
    python score_detector.py \\
        --pred outputs/rfdetr_cfd/predictions.json \\
        --pred outputs/yolo_baseline/predictions.json \\
        --label rfdetr_cfd --label yolo_baseline

Notes:
    pycocotools requires predictions to reference image_ids that exist in the
    GT. run_detector.py already does this (loads the GT's file_name -> id
    mapping). If you ever produce predictions another way, the image_ids
    must match.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", default="annotations/instances_default.json",
                   help="COCO ground-truth JSON.")
    p.add_argument("--pred", action="append", required=True,
                   help="Path to predictions.json (repeat for multiple backends).")
    p.add_argument("--label", action="append", default=[],
                   help="Display name for each --pred (parallel to --pred). "
                        "Defaults to the parent folder name.")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for TP/FP/FN counting. Default: 0.5. "
                        "(mAP itself is reported across the standard sweep.)")
    p.add_argument("--score_thresh", type=float, default=0.05,
                   help="Score cutoff for TP/FP/FN counting. Predictions below "
                        "this score are dropped. Default: 0.05.")
    return p.parse_args()


def label_for(pred_path: str, override: str | None) -> str:
    if override:
        return override
    return os.path.basename(os.path.dirname(os.path.abspath(pred_path))) or pred_path


def coco_eval(gt_path: str, pred_path: str) -> Dict:
    """Run COCO mAP evaluation; return a dict of named stats."""
    gt = COCO(gt_path)
    pred = gt.loadRes(pred_path)
    ev = COCOeval(gt, pred, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # COCO's 12-element stats array — these are the canonical names.
    return {
        "mAP@.5:.95": ev.stats[0],
        "mAP@.5":     ev.stats[1],
        "mAP@.75":    ev.stats[2],
        "mAP_small":  ev.stats[3],
        "mAP_medium": ev.stats[4],
        "mAP_large":  ev.stats[5],
        "AR_max1":    ev.stats[6],
        "AR_max10":   ev.stats[7],
        "AR_max100":  ev.stats[8],
        "AR_small":   ev.stats[9],
        "AR_medium":  ev.stats[10],
        "AR_large":   ev.stats[11],
    }


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two box arrays in COCO xywh format. Returns (len(a), len(b))."""
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
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def count_tp_fp_fn(
    gt_path: str, pred_path: str, iou_thresh: float, score_thresh: float,
) -> Tuple[int, int, int, float, float, int, int]:
    """
    Greedy per-image TP/FP/FN at one IoU threshold, score-cut applied first.

    Returns:
        tp, fp, fn, precision, recall, n_gt, n_pred_kept
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # group GT and predictions by image_id
    gt_by_img: Dict[int, List[dict]] = {}
    for a in gt_data["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    pred_by_img: Dict[int, List[dict]] = {}
    for p in preds:
        if p.get("score", 1.0) < score_thresh:
            continue
        pred_by_img.setdefault(p["image_id"], []).append(p)

    tp = fp = fn = 0
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_pred = sum(len(v) for v in pred_by_img.values())

    # all image_ids that appear in either side — both GT-only and pred-only frames matter
    image_ids = set(gt_by_img) | set(pred_by_img)

    for img_id in image_ids:
        gt_boxes = np.array([a["bbox"] for a in gt_by_img.get(img_id, [])], dtype=float)
        pr_list = sorted(pred_by_img.get(img_id, []), key=lambda x: -x.get("score", 0))
        pr_boxes = np.array([p["bbox"] for p in pr_list], dtype=float)

        if len(pr_boxes) == 0:
            fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            fp += len(pr_boxes)
            continue

        iou = iou_xywh(pr_boxes, gt_boxes)  # (P, G)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)
        for p_idx in range(len(pr_boxes)):
            # best unmatched GT for this prediction
            best_g = -1
            best_iou = iou_thresh
            for g_idx in range(len(gt_boxes)):
                if gt_taken[g_idx]:
                    continue
                if iou[p_idx, g_idx] >= best_iou:
                    best_iou = iou[p_idx, g_idx]
                    best_g = g_idx
            if best_g >= 0:
                gt_taken[best_g] = True
                tp += 1
            else:
                fp += 1
        fn += int((~gt_taken).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return tp, fp, fn, precision, recall, n_gt, n_pred


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.gt):
        print(f"error: GT not found: {args.gt}", file=sys.stderr)
        return 2

    labels = list(args.label) + [None] * (len(args.pred) - len(args.label))
    rows: List[Tuple[str, Dict, Tuple]] = []

    for pred_path, lbl in zip(args.pred, labels):
        if not os.path.exists(pred_path):
            print(f"error: predictions not found: {pred_path}", file=sys.stderr)
            return 2
        name = label_for(pred_path, lbl)
        print(f"\n{'=' * 70}\n[{name}]  {pred_path}\n{'=' * 70}")

        coco_stats = coco_eval(args.gt, pred_path)
        tp_fp = count_tp_fp_fn(
            args.gt, pred_path, args.iou, args.score_thresh,
        )
        rows.append((name, coco_stats, tp_fp))

    # --- Side-by-side comparison table ----------------------------------
    if len(rows) >= 1:
        print(f"\n\n{'=' * 70}")
        print(f"SIDE-BY-SIDE   (TP/FP/FN at IoU>={args.iou}, score>={args.score_thresh})")
        print(f"{'=' * 70}")
        # Header
        col_w = 18
        header = f"{'metric':<22}" + "".join(f"{n[:col_w]:>{col_w}}" for n, _, _ in rows)
        print(header)
        print("-" * len(header))

        def print_row(key: str, fmt: str = "{:.4f}", from_=None):
            vals = []
            for _, coco_stats, tp_fp in rows:
                if from_ == "tp_fp":
                    tp, fp, fn, prec, rec, ngt, npred = tp_fp
                    v = {"TP": tp, "FP": fp, "FN": fn,
                         "precision": prec, "recall": rec,
                         "GT boxes": ngt, "pred kept": npred}[key]
                else:
                    v = coco_stats[key]
                if isinstance(v, int):
                    vals.append(f"{v:>{col_w}d}")
                else:
                    vals.append(f"{v:>{col_w}.4f}")
            print(f"{key:<22}" + "".join(vals))

        # COCO-style mAP block
        for k in ("mAP@.5:.95", "mAP@.5", "mAP@.75",
                  "mAP_small", "mAP_medium", "mAP_large",
                  "AR_max100"):
            print_row(k)
        # TP/FP/FN block
        print("-" * len(header))
        for k in ("GT boxes", "pred kept", "TP", "FP", "FN", "precision", "recall"):
            print_row(k, from_="tp_fp")

    return 0


if __name__ == "__main__":
    sys.exit(main())
