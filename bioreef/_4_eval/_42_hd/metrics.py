"""
Hierarchical Distance metric — pure, stateless.

The tree-distance between predicted and true taxa in the Linnaean hierarchy
(Family -> Genus -> Species), mapped to a penalty. HDEvaluator logs and
aggregates these; the distance rule itself lives here.

    Same Species   -> 0      Same Family    -> 2
    Same Genus     -> 1      Different Fam. -> 3
"""

from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("bioreef._4_eval.hd")

# Default taxonomic level penalties (MATANet spec).
DEFAULT_LEVEL_WEIGHTS = {
    "species": 0,  # correct — no penalty
    "genus": 1,    # within-genus error — minor
    "family": 2,   # within-family error — moderate
    "root": 3,     # cross-family error — major
}


def hierarchical_distance(
    predicted_species: str,
    true_species: str,
    taxonomy_tree: Dict[str, Dict[str, str]],
    level_weights: Optional[Dict[str, int]] = None,
) -> Tuple[float, str]:
    """
    HD penalty + deepest matching level between predicted and true species.

    Returns (hd_score, match_level). Species missing from the tree get the
    maximum (root) penalty with level 'unknown'.
    """
    weights = level_weights or DEFAULT_LEVEL_WEIGHTS

    if predicted_species == true_species:
        return 0.0, "species"

    pred_tax = taxonomy_tree.get(predicted_species)
    true_tax = taxonomy_tree.get(true_species)
    if pred_tax is None or true_tax is None:
        logger.warning(
            f"Species not in taxonomy tree: pred='{predicted_species}', "
            f"true='{true_species}'. Assigning maximum HD."
        )
        return float(weights["root"]), "unknown"

    if pred_tax["genus"] == true_tax["genus"]:
        return float(weights["genus"]), "genus"
    if pred_tax["family"] == true_tax["family"]:
        return float(weights["family"]), "family"
    return float(weights["root"]), "root"
