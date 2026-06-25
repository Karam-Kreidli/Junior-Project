"""
Hierarchical-fallback aggregation (#5) — turn per-tracklet logits into one
taxonomic verdict each (species -> genus -> family -> unidentified).

Extracted from track_stage2.py so both the Stage-2 CLI and the in-process
run_stage2 use the same code (no script-imports-script). Bodies verbatim.

    build_taxonomy_for_aggregation(idx_to_sp, csv_path) -> taxonomy | None
    aggregate_video_verdicts(tracklets, taxonomy, idx_to_sp, sp, ge, fa) -> [verdict]
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("bioreef.pipeline.aggregation")


def build_taxonomy_for_aggregation(
    idx_to_sp: Dict[int, str],
    csv_path: str,
) -> Optional[Dict]:
    """
    Build the species→genus→family maps that Tracklet.aggregate_hierarchical
    needs, from the metadata CSV. Returns None (aggregation skipped) if the CSV
    is missing or lacks taxonomy columns.
    """
    if not idx_to_sp or not os.path.exists(csv_path):
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)
    if not {"species", "genus", "family"}.issubset(df.columns):
        logger.warning("CSV lacks species/genus/family columns; "
                       "hierarchical aggregation skipped.")
        return None

    # species name → (genus, family)
    tax = {}
    for _, r in df.dropna(subset=["species", "genus", "family"]).iterrows():
        tax.setdefault(r["species"], (r["genus"], r["family"]))

    num_species = len(idx_to_sp)
    genus_names_per_sp, family_names_per_sp = [], []
    for i in range(num_species):
        g, f = tax.get(idx_to_sp.get(i), ("__unknown_genus__",
                                          "__unknown_family__"))
        genus_names_per_sp.append(g)
        family_names_per_sp.append(f)

    genus_to_idx = {g: i for i, g in enumerate(sorted(set(genus_names_per_sp)))}
    family_to_idx = {f: i for i, f in enumerate(sorted(set(family_names_per_sp)))}

    return {
        "species_to_genus": [genus_to_idx[g] for g in genus_names_per_sp],
        "species_to_family": [family_to_idx[f] for f in family_names_per_sp],
        "num_genera": len(genus_to_idx),
        "num_families": len(family_to_idx),
        "genus_names": {i: g for g, i in genus_to_idx.items()},
        "family_names": {i: f for f, i in family_to_idx.items()},
    }


def aggregate_video_verdicts(
    tracklets: List,
    taxonomy: Dict,
    idx_to_sp: Dict[int, str],
    species_thresh: float,
    genus_thresh: float,
    family_thresh: float,
) -> List[Dict]:
    """
    Run the hierarchical-fallback aggregation on every tracklet and resolve the
    chosen class index to a readable taxonomic name. One verdict dict per
    tracklet: track_id, level, name, confidence, n_frames.
    """
    verdicts = []
    for t in tracklets:
        r = t.aggregate_hierarchical(
            taxonomy["species_to_genus"],
            taxonomy["species_to_family"],
            taxonomy["num_genera"],
            taxonomy["num_families"],
            species_thresh=species_thresh,
            genus_thresh=genus_thresh,
            family_thresh=family_thresh,
        )
        level, idx = r["level"], r["index"]
        if level == "species":
            name = idx_to_sp.get(idx, f"species_{idx}")
        elif level == "genus":
            name = taxonomy["genus_names"].get(idx, f"genus_{idx}")
        elif level == "family":
            name = taxonomy["family_names"].get(idx, f"family_{idx}")
        else:
            name = "unidentified"
        verdicts.append({
            "track_id": t.track_id,
            "level": level,
            "name": name,
            "confidence": round(r["confidence"], 4),
            "n_frames": r["n_frames"],
        })
    return verdicts
