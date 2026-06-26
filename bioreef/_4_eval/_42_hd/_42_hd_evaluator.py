"""
BioReef.ai — Hierarchical Distance (HD) Evaluator
===================================================
Primary evaluation metric for Stage 3 classification validation.
HD penalizes predictions based on their taxonomic "distance" from the
ground truth, ensuring that even when the model is uncertain, it remains
"biologically intelligent."

HD Logic:
    Unlike flat accuracy, HD measures how "biologically far" an error is:
        - Same Species:    HD = 0 (perfect)
        - Same Genus:      HD = 1 (minor error)
        - Same Family:     HD = 2 (moderate error)
        - Different Family: HD = 3 (major taxonomic failure)

Target: HD < 2.0 (current benchmark: 1.54)

Guardrails (.agent/rules.md):
    Every classification update must include a Hierarchical Distance log.

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

logger = logging.getLogger("bioreef._4_eval.hd")


class HDEvaluator:
    """
    Compute and log Hierarchical Distance (HD) for taxonomic classification.

    The HD metric measures the tree-distance between predicted and true
    taxa in the Linnaean hierarchy (Family → Genus → Species). This ensures
    the model's errors are "biologically reasonable" — confusing two species
    of Snapper is far less severe than confusing a Shark with a Snapper.

    Ecological relevance:
        For sustainable fisheries management in the Gulf of Oman, the
        difference between confusing Lutjanus ehrenbergii (Ehrenberg's
        Snapper) with Lutjanus kasmira (Common Bluestripe Snapper) vs.
        confusing it with Carcharhinus melanopterus (Blacktip Reef Shark)
        is the difference between a minor annotation note and a major
        report error. HD captures this biological reality mathematically.

    Taxonomy Tree Structure:
        Level 0 (Root) → Level 1 (Family) → Level 2 (Genus) → Level 3 (Species)

        HD = depth_of_LCA(predicted, ground_truth) mapped to penalty:
            Same Species   → 0
            Same Genus     → 1
            Same Family    → 2
            Different Fam. → 3
    """

    # Default taxonomic level weights following MATANet spec
    DEFAULT_LEVEL_WEIGHTS = {
        "species": 0,  # Correct — no penalty
        "genus": 1,    # Within-genus error — minor
        "family": 2,   # Within-family error — moderate
        "root": 3,     # Cross-family error — major
    }

    def __init__(
        self,
        taxonomy_tree: Optional[Dict[str, Dict[str, str]]] = None,
        level_weights: Optional[Dict[str, int]] = None,
        target_hd: float = 2.0,
        output_dir: str = "outputs/evaluation",
    ):
        """
        Args:
            taxonomy_tree: Full taxonomy mapping.
                           {species_name: {'family': ..., 'genus': ..., 'species': ...}}
            level_weights: Override penalty weights per taxonomic level.
            target_hd:     Target HD score (lower is better).
            output_dir:    Directory for JSON log output.
        """
        self.taxonomy_tree = taxonomy_tree or {}
        self.level_weights = level_weights or self.DEFAULT_LEVEL_WEIGHTS
        self.target_hd = target_hd
        self.output_dir = output_dir

        # Per-prediction log
        self._predictions: List[Dict[str, Any]] = []
        self._hd_scores: List[float] = []

    def compute_distance(
        self,
        predicted_species: str,
        true_species: str,
    ) -> Tuple[float, str]:
        """
        Compute the hierarchical distance between predicted and true species.

        Args:
            predicted_species: Predicted species name.
            true_species:      Ground-truth species name.

        Returns:
            (hd_score, match_level): The HD penalty and the deepest
            matching taxonomic level.
        """
        # Perfect match
        if predicted_species == true_species:
            return 0.0, "species"

        pred_tax = self.taxonomy_tree.get(predicted_species)
        true_tax = self.taxonomy_tree.get(true_species)

        if pred_tax is None or true_tax is None:
            logger.warning(
                f"Species not in taxonomy tree: "
                f"pred='{predicted_species}', true='{true_species}'. "
                f"Assigning maximum HD."
            )
            return float(self.level_weights["root"]), "unknown"

        # Check genus match
        if pred_tax["genus"] == true_tax["genus"]:
            return float(self.level_weights["genus"]), "genus"

        # Check family match
        if pred_tax["family"] == true_tax["family"]:
            return float(self.level_weights["family"]), "family"

        # Cross-family error — maximum penalty
        return float(self.level_weights["root"]), "root"

    def log_prediction(
        self,
        predicted_species: str,
        true_species: str,
        confidence: float = 0.0,
        track_id: Optional[int] = None,
        frame_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Log a single classification prediction with HD computation.

        Args:
            predicted_species: Model's species prediction.
            true_species:      Ground-truth species label.
            confidence:        Model confidence score (0–1).
            track_id:          Associated tracking ID (from Stage 2).
            frame_id:          Frame number in the video sequence.
            metadata:          Additional metadata (e.g., bbox, image_path).

        Returns:
            Dict with HD score, match level, and prediction details.
        """
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

        # Log warnings for severe errors (cross-family)
        # Silenced for clean terminal tracking during Phase 10
        if match_level == "root":
            pred_tax = self.taxonomy_tree.get(predicted_species, {})
            true_tax = self.taxonomy_tree.get(true_species, {})
            # logger.warning(
            #     f"⚠ CROSS-FAMILY ERROR: Predicted {predicted_species} "
            #     f"({pred_tax.get('family', '?')}) vs True {true_species} "
            #     f"({true_tax.get('family', '?')}) — HD={hd_score}"
            # )

        return entry

    def compute_aggregate(self) -> Dict[str, Any]:
        """
        Compute aggregate HD statistics across all logged predictions.

        Returns:
            Dict with mean HD, per-level breakdown, accuracy, and target status.
        """
        if not self._hd_scores:
            return {"mean_hd": 0.0, "num_predictions": 0}

        scores = np.array(self._hd_scores)

        # Per-level error breakdown
        level_counts = {"species": 0, "genus": 0, "family": 0, "root": 0, "unknown": 0}
        for pred in self._predictions:
            level = pred["match_level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        total = len(self._predictions)
        mean_hd = float(np.mean(scores))

        result = {
            "mean_hd": round(mean_hd, 4),
            "median_hd": round(float(np.median(scores)), 4),
            "std_hd": round(float(np.std(scores)), 4),
            "max_hd": round(float(np.max(scores)), 4),
            "num_predictions": total,
            "species_accuracy": round(level_counts["species"] / total, 4) if total > 0 else 0.0,
            "genus_accuracy": round(
                (level_counts["species"] + level_counts["genus"]) / total, 4
            ) if total > 0 else 0.0,
            "family_accuracy": round(
                (level_counts["species"] + level_counts["genus"] + level_counts["family"]) / total, 4
            ) if total > 0 else 0.0,
            "error_breakdown": {
                "correct_species": level_counts["species"],
                "within_genus_errors": level_counts["genus"],
                "within_family_errors": level_counts["family"],
                "cross_family_errors": level_counts["root"],
                "unknown_taxa": level_counts.get("unknown", 0),
            },
            "target_met": mean_hd <= self.target_hd,
        }

        status = "✅ PASS" if result["target_met"] else "❌ ABOVE TARGET"
        # Silenced duplicate print to keep the terminal perfectly clean
        # logger.info(
        #     f"HD Aggregate: mean={mean_hd:.4f}, "
        #     f"species_acc={result['species_accuracy']:.4f} — {status}"
        # )

        return result

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save all prediction logs and aggregate metrics to JSON.

        Returns:
            Path to the saved JSON file.
        """
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
        """
        Retrieve the N worst classification errors by HD score.

        Useful for targeted model debugging — cross-family errors
        often indicate systematic failures in the HSLM or
        missing Taxonomic Guardrails.
        """
        sorted_preds = sorted(
            self._predictions,
            key=lambda x: x["hd_score"],
            reverse=True,
        )
        return sorted_preds[:n]
