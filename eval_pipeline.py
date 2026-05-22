"""
BioReef.ai — End-to-End Pipeline Evaluation
=============================================
Runs detector -> MCEAM on the val split and reports metrics that matter
for real-world deployment, sweeping detection confidence thresholds.

Detector mAP treats missed fish and false positives equally, but in a
two-stage pipeline they have very different costs:
    - Missed fish   = fatal (never reaches the classifier)
    - False positive = cheap (classifier gives low-confidence, filterable)

So we measure END-TO-END species accuracy on matched detections, and
report detector RECALL (not precision/mAP) as the detection-side signal.

Usage:
    python eval_pipeline.py \\
        --detection_ckpt runs/detect/trainX/weights/best.pt \\
        --stage1_ckpt bioreef_stage1.pt \\
        --conf_sweep 0.05 0.1 0.25 0.5

Caveats:
    - val set is the same deterministic 80/10/10 split as train_stage1.py,
      filtered to species with >= min_samples examples.
    - GT per frame covers only the kept species. Rare-species fish in the
      same frame are silently ignored (counted as neither hit nor miss).
"""

import argparse
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from bioreef.detection import build_detector, Detector

from bioreef.models.backbone import ViTBackbone
from bioreef.models.mceam import MCEAM
from bioreef.data.data_factory import ContextHarvester
from bioreef.evaluation.hd_evaluator import HDEvaluator

from train_stage1 import split_dataset, get_taxonomy_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_pipeline")


# =============================================================================
# Geometry
# =============================================================================

def iou_xywh(a, b):
    """IoU between two [x, y, w, h] boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def greedy_match(det_boxes, det_conf, gt_boxes, iou_thresh=0.5):
    """Greedy match detections -> GT by descending conf, IoU >= thresh.
    Returns list of (det_idx, gt_idx) pairs.
    """
    used = set()
    matches = []
    for di in np.argsort(-det_conf):
        best_iou = iou_thresh
        best_gt = -1
        for gi in range(len(gt_boxes)):
            if gi in used:
                continue
            iou = iou_xywh(det_boxes[di], gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0:
            used.add(best_gt)
            matches.append((int(di), best_gt))
    return matches


# =============================================================================
# Classifier head forward
# =============================================================================

@torch.no_grad()
def classify_boxes(backbone, mceam, head, harvester, frame_bgr, boxes_xywh, device):
    """Run backbone -> MCEAM -> head on each box. Returns softmax scores (K, C)."""
    if len(boxes_xywh) == 0:
        return np.empty((0, head.out_features), dtype=np.float32)

    stream_lists = defaultdict(list)
    for b in boxes_xywh:
        x, y, w, h = int(b[0]), int(b[1]), max(int(b[2]), 1), max(int(b[3]), 1)
        crops = harvester.harvest(frame_bgr, (x, y, w, h))
        for name, t in crops.items():
            stream_lists[name].append(t)

    batched = {name: torch.stack(ts).to(device) for name, ts in stream_lists.items()}
    feats = backbone(batched)
    emb = mceam(feats)["embedding"]
    logits = head(emb)
    return torch.softmax(logits, dim=1).cpu().numpy()


# =============================================================================
# One conf threshold
# =============================================================================

def evaluate_at_conf(
    val_by_frame, detector, backbone, mceam, head, harvester,
    sp_to_idx, idx_to_sp, device, conf, hd_evaluator,
):
    total_gt = matched_gt = top1 = top5 = 0
    hd_evaluator.reset()

    for frame_path, gt_list in tqdm(val_by_frame.items(), desc=f"conf={conf}", leave=False):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        gt_boxes = [g['bbox'] for g in gt_list]
        gt_sps = [g['species'] for g in gt_list]
        total_gt += len(gt_list)

        dets = detector.predict(frame, conf=conf)
        if len(dets) == 0:
            continue
        det_boxes = dets.xywh
        det_conf = dets.conf

        pairs = greedy_match(det_boxes, det_conf, gt_boxes, iou_thresh=0.5)
        if not pairs:
            continue

        matched_boxes = np.array([det_boxes[di] for di, _ in pairs])
        scores = classify_boxes(backbone, mceam, head, harvester, frame, matched_boxes, device)
        top1_idx = scores.argmax(axis=1)
        top5_idx = np.argsort(scores, axis=1)[:, -5:]

        for m_i, (_, gi) in enumerate(pairs):
            matched_gt += 1
            gt_sp = gt_sps[gi]
            gt_idx = sp_to_idx[gt_sp]
            pred_sp = idx_to_sp[int(top1_idx[m_i])]

            if int(top1_idx[m_i]) == gt_idx:
                top1 += 1
            if gt_idx in top5_idx[m_i]:
                top5 += 1
            hd_evaluator.log_prediction(pred_sp, gt_sp)

    hd_stats = hd_evaluator.compute_aggregate()
    return {
        'conf': conf,
        'total_gt': total_gt,
        'matched_gt': matched_gt,
        'missed_gt': total_gt - matched_gt,
        'recall': matched_gt / total_gt if total_gt else 0.0,
        'top1': top1 / matched_gt if matched_gt else 0.0,
        'top5': top5 / matched_gt if matched_gt else 0.0,
        'hd': hd_stats['mean_hd'],
        'e2e_top1': top1 / total_gt if total_gt else 0.0,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_backend", default="rfdetr", choices=["rfdetr", "yolo"],
                   help="Detector backend (production: rfdetr per #6).")
    p.add_argument("--detection_ckpt", type=str, default=None,
                   help="Detector checkpoint. Defaults to "
                        "weights/rfdetr_medium_cfd.pth for rfdetr; required for yolo.")
    p.add_argument("--rfdetr_size", default="medium", choices=["medium", "small", "nano"])
    p.add_argument("--imgsz", type=int, default=960,
                   help="YOLO inference imgsz (ignored for rfdetr).")
    p.add_argument("--stage1_ckpt", type=str, default="bioreef_stage1.pt")
    p.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    p.add_argument("--img_dir", type=str, default="data_oz/frames_waternet_1")
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--conf_sweep", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
    p.add_argument("--split", type=str, default="val", choices=["val", "test"],
                   help="Which split to evaluate on. Uses ORIGINAL labels from the CSV (pre-GDINO).")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Reproducing {args.split} split (using ORIGINAL labels from CSV)...")
    # filter_placeholders=False matches the current checkpoint's class set,
    # which was trained before the placeholder filter was added.
    _, val_samples, test_samples, num_classes, idx_to_sp, _ = split_dataset(
        args.csv_path, args.img_dir, min_samples=args.min_samples,
        filter_placeholders=False,
    )
    sp_to_idx = {sp: i for i, sp in idx_to_sp.items()}

    eval_samples = val_samples if args.split == "val" else test_samples
    logger.info(f"  {args.split}: {len(eval_samples)} fish across {num_classes} species")

    val_by_frame = defaultdict(list)
    for s in eval_samples:
        val_by_frame[s['img_path']].append({'bbox': s['bbox'], 'species': s['species']})
    logger.info(f"  unique frames: {len(val_by_frame)}")

    detector = build_detector(
        args.detector_backend,
        weights=args.detection_ckpt,
        model_size=args.rfdetr_size,
        imgsz=args.imgsz,
    )

    logger.info("Loading backbone + MCEAM + head...")
    backbone = ViTBackbone(freeze=True).to(device).eval()
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)

    ckpt_classes = ckpt['head']['weight'].shape[0]
    if ckpt_classes != num_classes:
        raise RuntimeError(
            f"Checkpoint has {ckpt_classes} classes, split_dataset returned {num_classes}. "
            "CSV or min_samples may have changed since training."
        )

    mceam = MCEAM(
        embed_dim=backbone.embed_dim, num_context_levels=3, output_dim=256, num_heads=8,
    ).to(device).eval()
    mceam.load_state_dict(ckpt['mceam'])
    head = nn.Linear(256, num_classes).to(device).eval()
    head.load_state_dict(ckpt['head'])

    harvester = ContextHarvester(target_resolution=224, small_object_threshold=0.05)
    hd_evaluator = HDEvaluator(taxonomy_tree=get_taxonomy_tree(args.csv_path))

    results = []
    for conf in args.conf_sweep:
        logger.info(f"=== conf={conf} ===")
        r = evaluate_at_conf(
            val_by_frame, detector, backbone, mceam, head, harvester,
            sp_to_idx, idx_to_sp, device, conf, hd_evaluator,
        )
        results.append(r)
        logger.info(
            f"  recall={r['recall']*100:.2f}% | "
            f"top-1 (matched)={r['top1']*100:.2f}% | top-5={r['top5']*100:.2f}% | "
            f"HD={r['hd']:.4f} | end-to-end top-1={r['e2e_top1']*100:.2f}%"
        )

    # =============================================================================
    # Summary table
    # =============================================================================
    print()
    print("=" * 100)
    print("END-TO-END PIPELINE EVALUATION")
    print("=" * 100)
    header = (
        f"{'conf':>6} | {'total GT':>9} | {'matched':>8} | {'missed':>7} | "
        f"{'recall':>8} | {'top1_m':>8} | {'top5_m':>8} | {'HD':>7} | {'e2e_top1':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['conf']:>6.2f} | {r['total_gt']:>9d} | {r['matched_gt']:>8d} | {r['missed_gt']:>7d} | "
            f"{r['recall']*100:>7.2f}% | {r['top1']*100:>7.2f}% | {r['top5']*100:>7.2f}% | "
            f"{r['hd']:>7.4f} | {r['e2e_top1']*100:>8.2f}%"
        )
    print()
    print("top1_m / top5_m : accuracy on detections that MATCHED a GT box")
    print("e2e_top1        : (correctly identified) / (all GT fish) — the number that matters")


if __name__ == "__main__":
    main()
