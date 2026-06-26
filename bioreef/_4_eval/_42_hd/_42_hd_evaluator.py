"""
BioReef.ai — Hierarchical Distance (HD) Evaluator
===================================================
Primary evaluation metric for Stage 3 classification validation. HD penalizes
predictions by their taxonomic distance from ground truth, so the model stays
"biologically intelligent" even when uncertain (confusing two snappers is far
less severe than confusing a snapper with a shark).

This class logs predictions and aggregates HD statistics; the distance rule
lives in `metrics.py`. Target: HD < 2.0 (benchmark: 1.54).

Reference:
    Lee et al. (2026), "MATANet: A Multi-Context Attention and Taxonomy-Aware
    Network for Fine-Grained Underwater Recognition of Marine Species."
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import DEFAULT_LEVEL_WEIGHTS, hierarchical_distance

logger = logging.getLogger("bioreef._4_eval.hd")


class HDEvaluator:
    """
    Compute and log Hierarchical Distance (HD) for taxonomic classification.

    HD measures the tree-distance between predicted and true taxa in the
    Linnaean hierarchy (Family -> Genus -> Species), ensuring errors are
    "biologically reasonable" for fisheries reporting.
    """

    # Kept for backward compatibility (HDEvaluator.DEFAULT_LEVEL_WEIGHTS);
    # the canonical copy lives in metrics.
    DEFAULT_LEVEL_WEIGHTS = DEFAULT_LEVEL_WEIGHTS

    def __init__(
        self,
        taxonomy_tree: Optional[Dict[str, Dict[str, str]]] = None,
        level_weights: Optional[Dict[str, int]] = None,
        target_hd: float = 2.0,
        output_dir: str = "outputs/evaluation",
    ):
        """
        Args:
            taxonomy_tree: {species: {'family', 'genus', 'species'}}.
            level_weights: Override penalty weights per taxonomic level.
            target_hd:     Target HD score (lower is better).
            output_dir:    Directory for JSON log output.
        """
        self.taxonomy_tree = taxonomy_tree or {}
        self.level_weights = level_weights or DEFAULT_LEVEL_WEIGHTS
        self.target_hd = target_hd
        self.output_dir = output_dir
        self._predictions: List[Dict[str, Any]] = []
        self._hd_scores: List[float] = []

    def compute_distance(
        self, predicted_species: str, true_species: str
    ) -> Tuple[float, str]:
        """HD penalty + deepest matching level (delegates to metrics)."""
        return hierarchical_distance(
            predicted_species, true_species, self.taxonomy_tree, self.level_weights
        )

    def log_prediction(
        self,
        predicted_species: str,
        true_species: str,
        confidence: float = 0.0,
        track_id: Optional[int] = None,
        frame_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Log one prediction with its HD score; returns the entry."""
        hd_score, match_level = self.compute_distance(predicted_species, true_species)
        entry = {
            "predicted": predicted_species,
            "ground_truth": true_species,
            "hd_score": hd_score,
            "match_level": match_level,
            "confidence": round(confidence, 4),
            "correct": predicted_species == true_species,
            "track_id": track_id,
            "frame_id": frame_id,
        }
        if metadata:
            entry["metadata"] = metadata
        self._predictions.append(entry)
        self._hd_scores.append(hd_score)
        return entry

    def compute_aggregate(self) -> Dict[str, Any]:
        """Aggregate HD statistics across all logged predictions."""
        if not self._hd_scores:
            return {"mean_hd": 0.0, "num_predictions": 0}

        scores = np.array(self._hd_scores)
        level_counts = {"species": 0, "genus": 0, "family": 0, "root": 0, "unknown": 0}
        for pred in self._predictions:
            level = pred["match_level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        total = len(self._predictions)
        mean_hd = float(np.mean(scores))
        species, genus, family = (
            level_counts["species"], level_counts["genus"], level_counts["family"],
        )
        return {
            "mean_hd": round(mean_hd, 4),
            "median_hd": round(float(np.median(scores)), 4),
            "std_hd": round(float(np.std(scores)), 4),
            "max_hd": round(float(np.max(scores)), 4),
            "num_predictions": total,
            "species_accuracy": round(species / total, 4) if total else 0.0,
            "genus_accuracy": round((species + genus) / total, 4) if total else 0.0,
            "family_accuracy": round((species + genus + family) / total, 4) if total else 0.0,
            "error_breakdown": {
                "correct_species": species,
                "within_genus_errors": genus,
                "within_family_errors": family,
                "cross_family_errors": level_counts["root"],
                "unknown_taxa": level_counts.get("unknown", 0),
            },
            "target_met": mean_hd <= self.target_hd,
        }

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save prediction logs + aggregate metrics to JSON; returns the path."""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or f"hd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        output = {
            "metric": "Hierarchical Distance (HD)",
            "timestamp": datetime.now().isoformat(),
            "target": self.target_hd,
            "level_weights": self.level_weights,
            "aggregate": self.compute_aggregate(),
            "predictions": self._predictions,
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"HD results saved to: {filepath}")
        return filepath

    def reset(self):
        """Clear accumulated predictions for a fresh evaluation run."""
        self._predictions.clear()
        self._hd_scores.clear()

    def get_worst_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """The N worst classification errors by HD score (for debugging)."""
        return sorted(
            self._predictions, key=lambda x: x["hd_score"], reverse=True
        )[:n]
