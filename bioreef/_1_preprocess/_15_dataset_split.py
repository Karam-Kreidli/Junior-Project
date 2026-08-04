"""
Dataset preparation: species filtering, train/val/test splitting, and the
taxonomy maps used by HSLM.

Extracted from train_stage1.py / infer_stage1.py so these shared functions live
in the library — scripts import them from here instead of importing each other.
The filtering logic in split_dataset / is_placeholder_species is the exact
recipe the production checkpoint (bioreef_stage1.pt) was trained with; do NOT
change it without re-deriving the species mapping (see recover_species_mapping
and problems.md #24). The constants and bodies below are kept verbatim.

Functions:
    is_placeholder_species(name)        -> bool
    get_taxonomy_tree(csv_path)         -> {species: {genus, family, species}}
    build_taxonomy_maps(idx_to_sp, tree)-> (s2g, s2f, n_genera, n_families, n_missing)
    split_dataset(csv_path, img_dir, ...) -> train/val/test + class maps
    build_species_mapping(csv_path, ...)  -> (sp_to_idx, idx_to_sp)
    resolve_species_mapping(ckpt, ...)    -> idx_to_sp (checkpoint-first)
"""

import logging
import os
import random
import re as _re
from collections import Counter
from typing import Dict, Tuple

logger = logging.getLogger("bioreef._1_preprocess._15_dataset_split")


# =============================================================================
# Placeholder species filtering
# =============================================================================

_PLACEHOLDER_SPECIES = {
    "unidentified", "fish", "unknown", "unidentifiable",
    "other", "spp",
}
_SP_PATTERN = _re.compile(r'^sp\d+$', _re.IGNORECASE)


def is_placeholder_species(name):
    """True if the species label is a placeholder (sp1, sp3, unidentified, etc.)."""
    if not isinstance(name, str):
        return True
    s = name.strip().lower()
    return s in _PLACEHOLDER_SPECIES or bool(_SP_PATTERN.match(s))


# =============================================================================
# Binomial class key + canonicalization (KNOWN_BUGS #1, #9)
# =============================================================================
# The class KEY must be the full (genus, species) binomial, NEVER the bare
# epithet: 49 epithets in OzFish are shared across genera (e.g. 'niger' spans
# Macolor/Melichthys/Odonus/Parastromateus/Scarus), so the bare key silently
# MERGES distinct species (296 classes instead of 321). Ported from the paper
# repo bioreef-classify/bioreef/data/split.py.

# Source-data genus typos -> canonical spelling. A misspelled genus otherwise
# spawns a spurious extra class (or splits one species in two).
_GENUS_CANON = {
    "pterocaesio": "Pterocaesio",   # lowercase variant of Pterocaesio
    "Epinephalis": "Epinephelus",   # misspelling of Epinephelus
}

# Binomial -> correct family, for species whose metadata carries a WRONG family
# on some rows. Keyed by binomial (not family: the wrong family may be valid for
# other species). Epinephelus faveatus is a grouper (Serranidae) mislabelled
# Percichthyidae on some OzFish rows.
_FAMILY_CANON = {
    "Epinephelus faveatus": "Serranidae",
}


def canonical_genus(genus):
    """Normalize a genus string (fix known source typos, strip whitespace)."""
    g = genus.strip() if isinstance(genus, str) else ""
    return _GENUS_CANON.get(g, g)


def canonical_family(binomial_name, family):
    """Correct a known-wrong family for a species; otherwise strip and return."""
    f = family.strip() if isinstance(family, str) else ""
    return _FAMILY_CANON.get(binomial_name, f)


def binomial(genus, species):
    """Full-binomial class label 'Genus species'. Genus is canonicalized first so
    source typos don't spawn spurious classes. Falls back to the bare epithet when
    genus is missing so a partly-labelled row still gets a stable key."""
    g = canonical_genus(genus)
    s = species.strip() if isinstance(species, str) else ""
    return f"{g} {s}".strip() if g else s


# =============================================================================
# Taxonomy maps (for HSLM marginalization)
# =============================================================================

def get_taxonomy_tree(csv_path):
    """{binomial: {genus, family, species}} from the metadata CSV, keyed by the
    full binomial. The stored parent genus/family are canonicalized with the SAME
    functions as the key (KNOWN_BUGS #9a) so a corrected species never points at
    an un-corrected genus node. Raises on a binomial that appears with two
    different families (KNOWN_BUGS #9b) — a silent last-row-wins is a real bug in
    OzFish (Epinephelus faveatus). The read error is NOT swallowed (KNOWN_BUGS #8):
    an unreadable CSV must fail loudly, not silently return an empty tree that
    pools every species into one __unknown__ bucket and invalidates HSLM."""
    import pandas as pd
    df = pd.read_csv(csv_path)   # propagate read errors (KNOWN_BUGS #8)
    tree = {}
    for _, row in df.dropna(subset=['species', 'genus', 'family']).iterrows():
        name = binomial(row['genus'], row['species'])
        entry = {
            'genus': canonical_genus(row['genus']),
            'family': canonical_family(name, row['family']),
            'species': name,
        }
        if name in tree and tree[name] != entry:
            raise ValueError(
                f"conflicting taxonomy for '{name}': {tree[name]} vs {entry}. "
                "Two rows give this binomial different genus/family — canonicalize "
                "the metadata (see _FAMILY_CANON/_GENUS_CANON) before training."
            )
        tree[name] = entry
    return tree


def build_taxonomy_maps(idx_to_sp, taxonomy_tree):
    """species-idx -> genus-idx / family-idx maps for HSLMLoss. Returns
    (species_to_genus, species_to_family, num_genera, num_families, n_missing);
    species absent from the taxonomy go to shared "__unknown__" buckets (counted
    in n_missing) so training never crashes."""
    num_species = len(idx_to_sp)
    genus_names, family_names = [], []
    n_missing = 0
    for i in range(num_species):
        tax = taxonomy_tree.get(idx_to_sp[i])
        if tax is None:
            genus_names.append("__unknown_genus__")
            family_names.append("__unknown_family__")
            n_missing += 1
        else:
            genus_names.append(tax['genus'])
            family_names.append(tax['family'])

    genus_to_idx = {g: i for i, g in enumerate(sorted(set(genus_names)))}
    family_to_idx = {f: i for i, f in enumerate(sorted(set(family_names)))}

    species_to_genus = [genus_to_idx[g] for g in genus_names]
    species_to_family = [family_to_idx[f] for f in family_names]
    return (species_to_genus, species_to_family,
            len(genus_to_idx), len(family_to_idx), n_missing)


# =============================================================================
# Train/val/test split  (the exact recipe bioreef_stage1.pt was trained with —
# see problems.md #24; do not alter without re-deriving the species mapping)
# =============================================================================

def split_dataset(csv_path, img_dir, min_samples=20, filter_placeholders=True):
    """
    Build the train/val/test split from the frame metadata CSV.

    filter_placeholders (default True): drop samples whose 'species' is a
        placeholder like sp1/sp3/spp/unidentified. Set False to reproduce the
        old behavior (e.g., when loading a checkpoint that was trained with
        these placeholders included as classes).
    """
    import pandas as pd

    IMG_DIRS = [
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
    ]

    df = pd.read_csv(csv_path)

    # --- First pass: discover which frames exist on SSD and count per species ---
    raw_samples = []
    for _, row in df.iterrows():
        if pd.isna(row['species']):
            continue
        if filter_placeholders and is_placeholder_species(row['species']):
            continue

        # Class key is the full binomial (genus + epithet), NOT the bare epithet
        # (KNOWN_BUGS #1): shared epithets across genera would otherwise merge
        # distinct species into one class.
        sp = binomial(row.get('genus'), row['species'])

        img_path = os.path.join(img_dir, row['file_name'])
        if not os.path.exists(img_path):
            for alt in IMG_DIRS:
                candidate = os.path.join(alt, row['file_name'])
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path):
            x0, y0, x1, y1 = int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])
            raw_samples.append({
                'img_path': img_path,
                'bbox': [x0, y0, x1 - x0, y1 - y0],  # xyxy → xywh for ContextHarvester
                'species': sp,
            })

    # --- Filter species below min_samples threshold ---
    sp_counter = Counter(s['species'] for s in raw_samples)
    kept_species = sorted(sp for sp, cnt in sp_counter.items() if cnt >= min_samples)

    species_to_class = {sp: idx for idx, sp in enumerate(kept_species)}
    class_to_species = {idx: sp for sp, idx in species_to_class.items()}

    # --- Second pass: build final sample list with filtered class indices ---
    sp_counts = [0] * len(kept_species)
    all_samples = []

    for s in raw_samples:
        if s['species'] not in species_to_class:
            continue
        cls_idx = species_to_class[s['species']]
        all_samples.append({
            'img_path': s['img_path'],
            'bbox': s['bbox'],
            'class_idx': cls_idx,
            'species': s['species'],
        })
        sp_counts[cls_idx] += 1

    random.seed(42)
    random.shuffle(all_samples)

    n = len(all_samples)
    train_samples = all_samples[:int(n * 0.8)]
    val_samples = all_samples[int(n * 0.8):int(n * 0.9)]
    test_samples = all_samples[int(n * 0.9):]

    sp_counts = [max(1, c) for c in sp_counts]

    return train_samples, val_samples, test_samples, len(kept_species), class_to_species, sp_counts


# =============================================================================
# Species mapping  (CSV-derived index↔name, + checkpoint-first resolution)
# =============================================================================

def build_species_mapping(csv_path: str, min_samples: int = 20
                          ) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build (species -> idx) mapping from the CSV using the SAME filtering
    as split_dataset so class indices align with the MCEAM checkpoint.

    NOTE: this is the simple count-based variant used by infer_stage1 to label
    class indices when a checkpoint lacks an embedded mapping. It does NOT apply
    the placeholder / image-on-disk filters that split_dataset does — so it can
    over-count (see #24); resolve_species_mapping guards the head/mapping size.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        logger.warning(
            "Species CSV not found (%s); species names unavailable — "
            "predictions will show placeholder labels.", csv_path,
        )
        return {}, {}

    df = pd.read_csv(csv_path).dropna(subset=['species'])
    # Binomial key (KNOWN_BUGS #1) — must match split_dataset so inference class
    # indices align with the trained head. Genus may be NaN on some rows; binomial
    # falls back to the bare epithet there.
    binomials = [binomial(g, s) for g, s in zip(df.get('genus', df['species']), df['species'])]
    sp_counter = Counter(binomials)
    kept_species = sorted(sp for sp, cnt in sp_counter.items() if cnt >= min_samples)
    sp_to_idx = {sp: i for i, sp in enumerate(kept_species)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}
    return sp_to_idx, idx_to_sp


def resolve_species_mapping(ckpt: dict, csv_path: str, min_samples: int = 20
                            ) -> Dict[int, str]:
    """Resolve the species index→name mapping for a Stage 1 checkpoint.

    Authoritative source is the checkpoint itself: training now saves
    `idx_to_sp` so the class indices are self-describing. For older checkpoints
    that lack it, fall back to re-deriving from the CSV — only correct if the
    CSV matches the exact training image set, so a warning is emitted.
    """
    stored = ckpt.get("idx_to_sp")
    if stored:
        # torch.save/load may turn int keys into str — normalize back to int.
        return {int(k): v for k, v in stored.items()}

    logger.warning(
        "Checkpoint has no embedded species mapping; re-deriving from %s. "
        "This is only correct if the CSV matches the training image set.",
        csv_path,
    )
    _, idx_to_sp = build_species_mapping(csv_path, min_samples)
    return idx_to_sp
