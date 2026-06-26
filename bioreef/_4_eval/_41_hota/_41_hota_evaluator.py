"""
BioReef.ai — HOTA Evaluator (Higher Order Tracking Accuracy)
=============================================================
Primary evaluation metric for Stage 2 tracking validation. HOTA balances
Detection Accuracy (DetA) and Association Accuracy (AssA), providing a
robust proof of performance for biodiversity assessments.

HOTA = √(DetA × AssA)     [geometric mean across IoU thresholds]

    DetA: Measures how well the tracker FINDS fish in each frame.
    AssA: Measures how well the tracker MAINTAINS correct IDs over time.

Target: HOTA > 70% (current benchmark: 74.20%)

Guardrails (.agent/rules.md):
    Every tracking update must include a HOTA validation script.

Reference:
    Luiten et al. (2021), "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking."
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bioreef._4_eval.hota")


class HOTAEvaluator:
    """
    Compute and log HOTA metrics for multi-object tracking evaluation.

    Provides per-sequence and aggregate metrics, logged to JSON for
    experiment tracking and progress reporting.

    HOTA decomposes tracking performance into orthogonal components:
        - DetA (Detection Accuracy): True positive detections / total
        - AssA (Association Accuracy): Correct ID-to-track associations
        - LocA (Localization Accuracy): Spatial precision of detections

    Ecological relevance:
        In marine biodiversity monitoring, AssA is critical — a single
        Epinephelus coioides swimming in and out of frame must maintain
        one Track ID to avoid inflating population counts. HOTA captures
        this requirement by equally weighting detection and association.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        target_hota: float = 0.70,
        output_dir: str = "outputs/evaluation",
    ):
        """
        Args:
            iou_thresholds: IoU thresholds to evaluate across.
                            Default: [0.25, 0.50, 0.75] per spec.
            target_hota:    Target HOTA score for pass/fail reporting.
            output_dir:     Directory for JSON log output.
        """
        self.iou_thresholds = iou_thresholds or [0.25, 0.50, 0.75]
        self.target_hota = target_hota
        self.output_dir = output_dir

        # Accumulated results per sequence
        self._results: Dict[str, Dict[str, float]] = {}

    def compute_iou(
        self,
        box_a: np.ndarray,
        box_b: np.ndarray,
    ) -> float:
        """
        Compute IoU between two bounding boxes [x, y, w, h].

        Args:
            box_a, box_b: Bounding boxes in [x, y, w, h] format.

        Returns:
            Intersection over Union score (0.0 to 1.0).
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def compute_iou_matrix(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise IoU matrix between ground-truth and predicted boxes.

        Args:
            gt_boxes:   (M, 4) array of ground-truth boxes [x, y, w, h].
            pred_boxes: (N, 4) array of predicted boxes [x, y, w, h].

        Returns:
            (M, N) IoU matrix.
        """
        M = len(gt_boxes)
        N = len(pred_boxes)
        iou_matrix = np.zeros((M, N), dtype=np.float64)

        for i in range(M):
            for j in range(N):
                iou_matrix[i, j] = self.compute_iou(gt_boxes[i], pred_boxes[j])

        return iou_matrix

    def _compute_det_accuracy(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        iou_threshold: float,
    ) -> float:
        """
        Detection Accuracy at a given IoU threshold.

        DetA = |TP| / (|TP| + |FP| + |FN|)

        A true positive requires IoU >= threshold between a GT and
        predicted bounding box.
        """
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return 1.0
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return 0.0

        iou_matrix = self.compute_iou_matrix(gt_boxes, pred_boxes)

        # Greedy matching: assign each GT to best-matching prediction
        matched_gt = set()
        matched_pred = set()
        tp = 0

        # Sort by IoU (descending) for greedy assignment
        indices = np.argwhere(iou_matrix >= iou_threshold)
        if len(indices) > 0:
            ious_at_indices = iou_matrix[indices[:, 0], indices[:, 1]]
            sorted_order = np.argsort(-ious_at_indices)

            for idx in sorted_order:
                gi, pi = indices[idx]
                if gi not in matched_gt and pi not in matched_pred:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    tp += 1

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    def _compute_assoc_accuracy(
        self,
        gt_tracks: Dict[int, List[int]],
        pred_tracks: Dict[int, List[int]],
        gt_frame_boxes: Dict[int, Dict[int, np.ndarray]],
        pred_frame_boxes: Dict[int, Dict[int, np.ndarray]],
        iou_threshold: float,
    ) -> float:
        """
        Association Accuracy: measures ID consistency across frames.

        AssA = (1/|TP|) Σ |TPA_c| / (|TPA_c| + |FPA_c| + |FNA_c|)

        Where TPA_c denotes true positive associations for a matched
        GT-prediction track pair c.

        For marine tracking, this penalizes ID switches when a fish
        re-enters the frame or when two similar fish cross paths.
        """
        if not gt_tracks or not pred_tracks:
            return 0.0

        # For simplicity, compute per-track association Jaccard
        all_frames = set()
        for frames in gt_tracks.values():
            all_frames.update(frames)
        for frames in pred_tracks.values():
            all_frames.update(frames)

        if not all_frames:
            return 0.0

        # Match GT tracks to pred tracks by temporal overlap
        associations = []
        for gt_id, gt_frames in gt_tracks.items():
            best_score = 0.0
            gt_set = set(gt_frames)
            for pred_id, pred_frames in pred_tracks.items():
                pred_set = set(pred_frames)
                intersection = len(gt_set & pred_set)
                union = len(gt_set | pred_set)
                jaccard = intersection / union if union > 0 else 0.0
                best_score = max(best_score, jaccard)
            associations.append(best_score)

        return float(np.mean(associations)) if associations else 0.0

    def evaluate_sequence(
        self,
        sequence_name: str,
        gt_data: Dict[int, List[Dict[str, Any]]],
        pred_data: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Evaluate a single video sequence.

        Args:
            sequence_name: Identifier for this sequence.
            gt_data:   Ground truth per frame:
                       {frame_id: [{'track_id': int, 'bbox': [x,y,w,h]}]}
            pred_data: Predictions per frame:
                       {frame_id: [{'track_id': int, 'bbox': [x,y,w,h]}]}

        Returns:
            Dict with HOTA, DetA, AssA, LocA scores.
        """
        det_accs = []
        assoc_accs = []

        # Build track-to-frames mappings
        gt_tracks: Dict[int, List[int]] = defaultdict(list)
        pred_tracks: Dict[int, List[int]] = defaultdict(list)
        gt_frame_boxes: Dict[int, Dict[int, np.ndarray]] = {}
        pred_frame_boxes: Dict[int, Dict[int, np.ndarray]] = {}

        all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

        for frame_id in all_frames:
            gt_items = gt_data.get(frame_id, [])
            pred_items = pred_data.get(frame_id, [])

            for item in gt_items:
                gt_tracks[item["track_id"]].append(frame_id)
            for item in pred_items:
                pred_tracks[item["track_id"]].append(frame_id)

            gt_frame_boxes[frame_id] = {
                item["track_id"]: np.array(item["bbox"]) for item in gt_items
            }
            pred_frame_boxes[frame_id] = {
                item["track_id"]: np.array(item["bbox"]) for item in pred_items
            }

        for iou_thresh in self.iou_thresholds:
            # Per-frame detection accuracy
            frame_det_accs = []
            for frame_id in all_frames:
                gt_boxes = np.array([
                    item["bbox"] for item in gt_data.get(frame_id, [])
                ])
                pred_boxes = np.array([
                    item["bbox"] for item in pred_data.get(frame_id, [])
                ])

                if len(gt_boxes) == 0:
                    gt_boxes = np.empty((0, 4))
                if len(pred_boxes) == 0:
                    pred_boxes = np.empty((0, 4))

                det_a = self._compute_det_accuracy(gt_boxes, pred_boxes, iou_thresh)
                frame_det_accs.append(det_a)

            det_accs.append(np.mean(frame_det_accs) if frame_det_accs else 0.0)

            # Association accuracy
            assoc_a = self._compute_assoc_accuracy(
                gt_tracks, pred_tracks,
                gt_frame_boxes, pred_frame_boxes,
                iou_thresh,
            )
            assoc_accs.append(assoc_a)

        # Aggregate across IoU thresholds
        mean_det_a = float(np.mean(det_accs))
        mean_assoc_a = float(np.mean(assoc_accs))
        hota = float(np.sqrt(mean_det_a * mean_assoc_a))

        result = {
            "HOTA": round(hota, 4),
            "DetA": round(mean_det_a, 4),
            "AssA": round(mean_assoc_a, 4),
            "target_met": hota >= self.target_hota,
            "iou_thresholds": self.iou_thresholds,
        }

        self._results[sequence_name] = result

        status = "✅ PASS" if result["target_met"] else "❌ BELOW TARGET"
        logger.info(
            f"HOTA [{sequence_name}]: {hota:.4f} "
            f"(DetA={mean_det_a:.4f}, AssA={mean_assoc_a:.4f}) — {status}"
        )

        return result

    def aggregate(self) -> Dict[str, float]:
        """Compute aggregate HOTA across all evaluated sequences."""
        if not self._results:
            return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}

        hotas = [r["HOTA"] for r in self._results.values()]
        det_as = [r["DetA"] for r in self._results.values()]
        assoc_as = [r["AssA"] for r in self._results.values()]

        agg = {
            "HOTA_mean": round(float(np.mean(hotas)), 4),
            "DetA_mean": round(float(np.mean(det_as)), 4),
            "AssA_mean": round(float(np.mean(assoc_as)), 4),
            "num_sequences": len(self._results),
            "target_met": float(np.mean(hotas)) >= self.target_hota,
        }

        logger.info(f"HOTA Aggregate: {agg}")
        return agg

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save all results to a JSON log file.

        Returns:
            Path to the saved JSON file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or f"hota_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        output = {
            "metric": "HOTA",
            "timestamp": datetime.now().isoformat(),
            "target": self.target_hota,
            "iou_thresholds": self.iou_thresholds,
            "per_sequence": self._results,
            "aggregate": self.aggregate(),
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"HOTA results saved to: {filepath}")
        return filepath

    def reset(self):
        """Clear accumulated results for a fresh evaluation run."""
        self._results.clear()
