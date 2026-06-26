"""
HOTA metric math — pure, stateless functions.

The geometry and accuracy formulas behind HOTA, with no evaluator state or IO.
HOTAEvaluator orchestrates these; keeping them here makes each piece testable in
isolation and keeps the class focused on accumulation + reporting.

    HOTA = sqrt(DetA * AssA)   [geometric mean across IoU thresholds]
"""

from typing import Dict, List

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two [x, y, w, h] boxes (0.0–1.0)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix (M, N) between GT and predicted [x, y, w, h] boxes."""
    M, N = len(gt_boxes), len(pred_boxes)
    iou_matrix = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])
    return iou_matrix


def det_accuracy(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_threshold: float,
) -> float:
    """
    Detection Accuracy at an IoU threshold: |TP| / (|TP| + |FP| + |FN|).

    Greedy IoU matching — a TP needs IoU >= threshold between a GT and a
    predicted box.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

    matched_gt: set = set()
    matched_pred: set = set()
    tp = 0

    indices = np.argwhere(iou_matrix >= iou_threshold)
    if len(indices) > 0:
        ious_at_indices = iou_matrix[indices[:, 0], indices[:, 1]]
        for idx in np.argsort(-ious_at_indices):
            gi, pi = indices[idx]
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def assoc_accuracy(
    gt_tracks: Dict[int, List[int]],
    pred_tracks: Dict[int, List[int]],
) -> float:
    """
    Association Accuracy: ID consistency across frames, as the mean best
    temporal-Jaccard between each GT track and any predicted track.

    Penalizes ID switches when a fish re-enters frame or two similar fish
    cross paths.
    """
    if not gt_tracks or not pred_tracks:
        return 0.0

    associations = []
    for gt_frames in gt_tracks.values():
        gt_set = set(gt_frames)
        best_score = 0.0
        for pred_frames in pred_tracks.values():
            pred_set = set(pred_frames)
            union = len(gt_set | pred_set)
            jaccard = len(gt_set & pred_set) / union if union > 0 else 0.0
            best_score = max(best_score, jaccard)
        associations.append(best_score)

    return float(np.mean(associations)) if associations else 0.0


def hota_score(det_a: float, assoc_a: float) -> float:
    """HOTA = sqrt(DetA * AssA)."""
    return float(np.sqrt(det_a * assoc_a))
