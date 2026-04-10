"""
BioReef.ai — Detection Evaluation
===================================
Evaluates a trained detection checkpoint on the val/test split,
computing mAP@0.5, mAP@0.75, and mAP@[0.5:0.95].

Usage:
    python eval_detection.py --ckpt bioreef_detection.pt

    python eval_detection.py \
        --ckpt bioreef_detection.pt \
        --csv_path data_oz/metadata/frame_metadata.csv \
        --split test \
        --conf_threshold 0.1
"""

import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.ops as tv_ops
from tqdm import tqdm

from bioreef.models.backbone import ViTBackbone
from bioreef.models.detector import BioReefDetector
from bioreef.data.detection_dataset import (
    load_detection_data,
    split_detection_frames,
    DetectionDataset,
    detection_collate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bioreef.eval_det")


# =============================================================================
# IoU Computation
# =============================================================================

def box_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU between predicted and ground truth boxes.

    Args:
        pred: (M, 4) boxes in [cx, cy, w, h] normalized format.
        gt:   (N, 4) boxes in [cx, cy, w, h] normalized format.

    Returns:
        (M, N) IoU matrix.
    """
    # Convert cxcywh -> xyxy
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    gt_x1 = gt[:, 0] - gt[:, 2] / 2
    gt_y1 = gt[:, 1] - gt[:, 3] / 2
    gt_x2 = gt[:, 0] + gt[:, 2] / 2
    gt_y2 = gt[:, 1] + gt[:, 3] / 2

    M, N = len(pred), len(gt)
    iou = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        xx1 = np.maximum(pred_x1[i], gt_x1)
        yy1 = np.maximum(pred_y1[i], gt_y1)
        xx2 = np.minimum(pred_x2[i], gt_x2)
        yy2 = np.minimum(pred_y2[i], gt_y2)

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_pred = (pred_x2[i] - pred_x1[i]) * (pred_y2[i] - pred_y1[i])
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union = area_pred + area_gt - inter

        iou[i] = np.where(union > 0, inter / union, 0.0)

    return iou


# =============================================================================
# Per-Class AP Computation
# =============================================================================

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the 101-point interpolation (COCO style)."""
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        precs_above = precisions[recalls >= t]
        ap += precs_above.max() if len(precs_above) > 0 else 0.0
    return ap / 101.0


def compute_recall(
    all_preds: List[Dict],
    all_gts: List[Dict],
    iou_threshold: float,
) -> Tuple[float, float]:
    """
    Compute detection recall: fraction of GT boxes matched by at least one prediction.

    A GT box is recalled if any prediction (regardless of class) has IoU >= iou_threshold.
    Also returns class-correct recall: GT box matched AND predicted class matches.

    Returns:
        (recall_any_class, recall_correct_class)
    """
    total_gt = 0
    recalled_any = 0
    recalled_correct = 0

    for pred, gt in zip(all_preds, all_gts):
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']

        total_gt += len(gt_labels)

        if len(gt_labels) == 0:
            continue
        if len(pred_boxes) == 0:
            continue

        ious = box_iou(pred_boxes, gt_boxes)  # (M, N)

        for g_idx in range(len(gt_labels)):
            best_iou = ious[:, g_idx].max() if len(ious) > 0 else 0.0
            if best_iou >= iou_threshold:
                recalled_any += 1
                # Check if the best-matching prediction has the correct class
                best_pred_idx = ious[:, g_idx].argmax()
                if pred_labels[best_pred_idx] == gt_labels[g_idx]:
                    recalled_correct += 1

    if total_gt == 0:
        return 0.0, 0.0

    return recalled_any / total_gt, recalled_correct / total_gt


def evaluate_per_class(
    all_preds: List[Dict],
    all_gts: List[Dict],
    iou_threshold: float,
    num_classes: int,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute per-class AP at a given IoU threshold.

    Args:
        all_preds: List of dicts per image: {'boxes': (M,4), 'scores': (M,), 'labels': (M,)}
        all_gts:   List of dicts per image: {'boxes': (N,4), 'labels': (N,)}
        iou_threshold: IoU threshold for a match.
        num_classes: Total number of foreground classes.

    Returns:
        (mAP, per_class_ap) where per_class_ap maps class_idx -> AP.
    """
    # Gather all predictions and GT per class
    class_preds = defaultdict(list)  # class -> [(score, img_idx, pred_idx)]
    class_gts = defaultdict(lambda: defaultdict(list))  # class -> img_idx -> [gt_idx]
    class_n_gt = defaultdict(int)

    for img_idx, (pred, gt) in enumerate(zip(all_preds, all_gts)):
        for p_idx in range(len(pred['labels'])):
            cls = pred['labels'][p_idx]
            score = pred['scores'][p_idx]
            class_preds[cls].append((score, img_idx, p_idx))

        for g_idx in range(len(gt['labels'])):
            cls = gt['labels'][g_idx]
            class_gts[cls][img_idx].append(g_idx)
            class_n_gt[cls] += 1

    per_class_ap = {}

    for cls in range(num_classes):
        n_gt = class_n_gt[cls]
        preds = class_preds[cls]

        if n_gt == 0 and len(preds) == 0:
            continue
        if n_gt == 0:
            per_class_ap[cls] = 0.0
            continue

        # Sort predictions by score (descending)
        preds.sort(key=lambda x: -x[0])

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = defaultdict(set)  # img_idx -> set of matched gt indices

        for det_idx, (score, img_idx, p_idx) in enumerate(preds):
            gt_indices = class_gts[cls].get(img_idx, [])
            if not gt_indices:
                fp[det_idx] = 1
                continue

            pred_box = all_preds[img_idx]['boxes'][p_idx:p_idx + 1]
            gt_boxes = all_gts[img_idx]['boxes'][np.array(gt_indices)]
            ious = box_iou(pred_box, gt_boxes)[0]

            best_iou_idx = ious.argmax()
            best_iou = ious[best_iou_idx]
            best_gt_idx = gt_indices[best_iou_idx]

            if best_iou >= iou_threshold and best_gt_idx not in matched[img_idx]:
                tp[det_idx] = 1
                matched[img_idx].add(best_gt_idx)
            else:
                fp[det_idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)

        per_class_ap[cls] = compute_ap(recalls, precisions)

    if per_class_ap:
        mAP = np.mean(list(per_class_ap.values()))
    else:
        mAP = 0.0

    return mAP, per_class_ap


# =============================================================================
# Main Evaluation
# =============================================================================

@torch.no_grad()
def run_evaluation(
    backbone: ViTBackbone,
    detector: BioReefDetector,
    dataset: DetectionDataset,
    device: torch.device,
    num_classes: int,
    conf_threshold: float,
    batch_size: int,
    nms_threshold: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """Run detector on all frames and collect predictions + GT."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    all_preds = []
    all_gts = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch['images'].to(device)
        targets = batch['targets']

        with torch.amp.autocast('cuda'):
            patch_tokens = backbone.extract_patch_tokens(images)
            outputs = detector(patch_tokens, targets=None)

        pred_logits = outputs['pred_logits']  # (B, N, C+1)
        pred_boxes = outputs['pred_boxes']    # (B, N, 4)

        B = images.size(0)
        for i in range(B):
            logits = pred_logits[i]  # (N, C+1)
            boxes = pred_boxes[i]    # (N, 4)

            # Softmax, take foreground max
            probs = torch.softmax(logits, dim=-1)
            fg_probs = probs[:, :-1]
            scores, labels = fg_probs.max(dim=-1)

            # Filter by confidence
            mask = scores >= conf_threshold
            scores_f = scores[mask]
            labels_f = labels[mask]
            boxes_f = boxes[mask]

            # Apply NMS (convert cxcywh -> xyxy for torchvision)
            if len(scores_f) > 0:
                cx, cy, w, h = boxes_f.unbind(-1)
                xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
                keep = tv_ops.nms(xyxy, scores_f, iou_threshold=nms_threshold)
                scores_f = scores_f[keep]
                labels_f = labels_f[keep]
                boxes_f = boxes_f[keep]

            pred_scores = scores_f.cpu().numpy()
            pred_labels = labels_f.cpu().numpy()
            pred_bboxes = boxes_f.cpu().numpy()

            all_preds.append({
                'boxes': pred_bboxes,
                'scores': pred_scores,
                'labels': pred_labels,
            })

            gt_labels = targets[i]['labels'].cpu().numpy()
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            all_gts.append({
                'boxes': gt_boxes,
                'labels': gt_labels,
            })

    return all_preds, all_gts


def main():
    parser = argparse.ArgumentParser(description="BioReef.ai Detection Evaluation")
    parser.add_argument("--ckpt", type=str, default="bioreef_detection.pt",
                        help="Path to trained detection checkpoint.")
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="/media/openuae/UUI/frames_waternet")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which split to evaluate on.")
    parser.add_argument("--conf_threshold", type=float, default=0.1,
                        help="Low threshold to get full PR curve. Default: 0.1")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="IoU threshold for NMS. Default: 0.5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Show per-class AP for the top/bottom K classes.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load data ---
    img_dirs = [
        args.img_dir,
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
        "/media/openuae/UUI/frames_waternet_3",
    ]
    frames, sp_to_idx, idx_to_sp = load_detection_data(args.csv_path, img_dirs)
    train_frames, val_frames, test_frames = split_detection_frames(frames)

    eval_frames = val_frames if args.split == "val" else test_frames
    logger.info(f"Evaluating on {args.split} split: {len(eval_frames)} frames")

    eval_ds = DetectionDataset(eval_frames, input_size=512, is_train=False)
    num_classes = len(sp_to_idx)

    # --- Load models ---
    logger.info("Loading backbone...")
    backbone = ViTBackbone(freeze=True).to(device)
    backbone.eval()

    logger.info(f"Loading detector: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    det_args = ckpt["args"]

    detector = BioReefDetector(
        backbone_dim=backbone.embed_dim,
        hidden_dim=det_args["hidden_dim"],
        num_queries=det_args["num_queries"],
        num_classes=ckpt["num_classes"],
        num_decoder_layers=det_args["num_decoder_layers"],
        num_fdr_bins=det_args["num_fdr_bins"],
    ).to(device)
    detector.load_state_dict(ckpt["detector"])
    detector.eval()
    logger.info(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}, val_loss: {ckpt.get('val_loss', '?')}")

    # --- Run inference ---
    all_preds, all_gts = run_evaluation(
        backbone, detector, eval_ds, device, num_classes,
        args.conf_threshold, args.batch_size, args.nms_threshold,
    )

    # --- Compute mAP at multiple thresholds ---
    iou_thresholds = [0.5, 0.75]
    coco_thresholds = np.arange(0.5, 1.0, 0.05)

    logger.info("=" * 60)
    logger.info("Detection Evaluation Results")
    logger.info("=" * 60)

    for iou_t in iou_thresholds:
        mAP, per_class = evaluate_per_class(all_preds, all_gts, iou_t, num_classes)
        logger.info(f"  mAP@{iou_t:.2f}: {mAP:.4f}")

    # COCO-style mAP@[0.5:0.95]
    coco_maps = []
    for iou_t in coco_thresholds:
        m, _ = evaluate_per_class(all_preds, all_gts, iou_t, num_classes)
        coco_maps.append(m)
    coco_mAP = np.mean(coco_maps)
    logger.info(f"  mAP@[.5:.95]: {coco_mAP:.4f}")

    # --- Per-class breakdown ---
    _, per_class_50 = evaluate_per_class(all_preds, all_gts, 0.5, num_classes)

    if per_class_50:
        sorted_classes = sorted(per_class_50.items(), key=lambda x: -x[1])
        k = min(args.top_k, len(sorted_classes))

        logger.info(f"\n  Top {k} classes by AP@0.5:")
        for cls_idx, ap in sorted_classes[:k]:
            name = idx_to_sp.get(cls_idx, f"class_{cls_idx}")
            logger.info(f"    {name:40s} AP={ap:.4f}")

        logger.info(f"\n  Bottom {k} classes by AP@0.5:")
        for cls_idx, ap in sorted_classes[-k:]:
            name = idx_to_sp.get(cls_idx, f"class_{cls_idx}")
            logger.info(f"    {name:40s} AP={ap:.4f}")

    # --- Recall (primary metric for sparsely-annotated datasets) ---
    logger.info("\n  Recall (does the model find the labeled fish?):")
    for iou_t in [0.3, 0.5, 0.75]:
        rec_any, rec_cls = compute_recall(all_preds, all_gts, iou_t)
        logger.info(
            f"    IoU≥{iou_t:.2f}  |  "
            f"Recall (any class): {rec_any:.4f}  |  "
            f"Recall (correct class): {rec_cls:.4f}"
        )

    # --- Detection statistics ---
    total_preds = sum(len(p['scores']) for p in all_preds)
    total_gts = sum(len(g['labels']) for g in all_gts)
    avg_preds = total_preds / max(len(all_preds), 1)
    avg_gts = total_gts / max(len(all_gts), 1)

    logger.info(f"\n  Total predictions: {total_preds} ({avg_preds:.1f}/frame)")
    logger.info(f"  Total ground truth: {total_gts} ({avg_gts:.1f}/frame)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
