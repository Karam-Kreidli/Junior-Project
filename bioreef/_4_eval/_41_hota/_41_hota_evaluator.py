"""
BioReef.ai — HOTA Evaluator (Higher Order Tracking Accuracy)
=============================================================
Primary evaluation metric for Stage 2 tracking validation. HOTA balances
Detection Accuracy (DetA) and Association Accuracy (AssA):

    HOTA = sqrt(DetA * AssA)     [geometric mean across IoU thresholds]

    DetA: how well the tracker FINDS fish in each frame.
    AssA: how well the tracker MAINTAINS correct IDs over time.

This class accumulates per-sequence results and reports/saves them; the metric
math lives in `metrics.py`. Target: HOTA > 70% (benchmark: 74.20%).

Reference:
    Luiten et al. (2021), "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking."
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import compute_iou, det_accuracy, assoc_accuracy, hota_score

logger = logging.getLogger("bioreef._4_eval.hota")


class HOTAEvaluator:
    """
    Compute and log HOTA metrics for multi-object tracking evaluation.

    Provides per-sequence and aggregate metrics, logged to JSON for experiment
    tracking. In marine monitoring AssA is critical — one fish swimming in and
    out of frame must keep one Track ID, or population counts inflate.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        target_hota: float = 0.70,
        output_dir: str = "outputs/evaluation",
    ):
        """
        Args:
            iou_thresholds: IoU thresholds to evaluate across (default
                            [0.25, 0.50, 0.75]).
            target_hota:    Target HOTA for pass/fail reporting.
            output_dir:     Directory for JSON log output.
        """
        self.iou_thresholds = iou_thresholds or [0.25, 0.50, 0.75]
        self.target_hota = target_hota
        self.output_dir = output_dir
        self._results: Dict[str, Dict[str, float]] = {}

    # IoU is exposed on the instance for backward compatibility with callers
    # that used evaluator.compute_iou(...); the implementation lives in metrics.
    @staticmethod
    def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        return compute_iou(box_a, box_b)

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
            gt_data / pred_data: per frame, {frame_id: [{'track_id', 'bbox'}]}.

        Returns:
            Dict with HOTA, DetA, AssA, and target status.
        """
        # Build track -> frames maps (used by association accuracy).
        gt_tracks: Dict[int, List[int]] = defaultdict(list)
        pred_tracks: Dict[int, List[int]] = defaultdict(list)
        all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
        for frame_id in all_frames:
            for item in gt_data.get(frame_id, []):
                gt_tracks[item["track_id"]].append(frame_id)
            for item in pred_data.get(frame_id, []):
                pred_tracks[item["track_id"]].append(frame_id)

        det_accs, assoc_accs = [], []
        for iou_thresh in self.iou_thresholds:
            frame_det_accs = []
            for frame_id in all_frames:
                gt_boxes = np.array([it["bbox"] for it in gt_data.get(frame_id, [])])
                pred_boxes = np.array([it["bbox"] for it in pred_data.get(frame_id, [])])
                if len(gt_boxes) == 0:
                    gt_boxes = np.empty((0, 4))
                if len(pred_boxes) == 0:
                    pred_boxes = np.empty((0, 4))
                frame_det_accs.append(det_accuracy(gt_boxes, pred_boxes, iou_thresh))
            det_accs.append(np.mean(frame_det_accs) if frame_det_accs else 0.0)
            assoc_accs.append(assoc_accuracy(gt_tracks, pred_tracks))

        mean_det_a = float(np.mean(det_accs))
        mean_assoc_a = float(np.mean(assoc_accs))
        hota = hota_score(mean_det_a, mean_assoc_a)

        result = {
            "HOTA": round(hota, 4),
            "DetA": round(mean_det_a, 4),
            "AssA": round(mean_assoc_a, 4),
            "target_met": hota >= self.target_hota,
            "iou_thresholds": self.iou_thresholds,
        }
        self._results[sequence_name] = result

        status = "PASS" if result["target_met"] else "BELOW TARGET"
        logger.info(
            f"HOTA [{sequence_name}]: {hota:.4f} "
            f"(DetA={mean_det_a:.4f}, AssA={mean_assoc_a:.4f}) — {status}"
        )
        return result

    def aggregate(self) -> Dict[str, float]:
        """Aggregate HOTA across all evaluated sequences."""
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
        """Save per-sequence + aggregate results to JSON; returns the path."""
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
