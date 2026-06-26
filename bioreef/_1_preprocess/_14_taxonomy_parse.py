"""TaxonomicParser — metadata parse/filter (split from data_factory)."""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger("bioreef._1_preprocess")


class TaxonomicParser:
    """
    Parse OzFish annotations into hierarchical labels: species->genus->family
    lookup, ambiguity filtering ('Unidentified', 'Fish', ...), and a spatial
    validity check (bbox must support the 5x context crop). The label vectors
    are the three supervisory signals for HSLM training.
    """

    # Labels that indicate ambiguous or incomplete annotations
    DEFAULT_FILTER_LABELS = frozenset([
        "Unidentified", "Fish", "Unknown", "unidentifiable",
        "fish", "unknown", "unidentified", "other", "Other",
        "spp", "sp1", "sp2", "sp3", "sp6", "sp10",
    ])

    def __init__(
        self,
        taxonomy_map: Optional[Dict[str, Dict[str, str]]] = None,
        filter_labels: Optional[List[str]] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        max_crop_scale: int = 5,
    ):
        self.taxonomy_map = taxonomy_map or {}
        self.filter_labels = frozenset(filter_labels) if filter_labels else self.DEFAULT_FILTER_LABELS
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.max_crop_scale = max_crop_scale

        # Build label encoders from taxonomy_map
        self._build_encoders()

    def _build_encoders(self):
        """Build integer encoders for each taxonomic level."""
        families = sorted(set(v["family"] for v in self.taxonomy_map.values()))
        genera = sorted(set(v["genus"] for v in self.taxonomy_map.values()))
        species = sorted(set(v["species"] for v in self.taxonomy_map.values()))

        self.family_to_idx = {f: i for i, f in enumerate(families)}
        self.genus_to_idx = {g: i for i, g in enumerate(genera)}
        self.species_to_idx = {s: i for i, s in enumerate(species)}

        self.idx_to_family = {i: f for f, i in self.family_to_idx.items()}
        self.idx_to_genus = {i: g for g, i in self.genus_to_idx.items()}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}

        logger.info(
            f"TaxonomicParser initialized: {len(families)} families, "
            f"{len(genera)} genera, {len(species)} species."
        )

    def is_valid_label(self, label: str) -> bool:
        """Check if a label passes ambiguity filtering."""
        return label not in self.filter_labels and label in self.taxonomy_map

    def check_spatial_validity(
        self, bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Valid if the 5x context crop keeps >50% inside the frame per axis."""
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        max_w = w * self.max_crop_scale
        max_h = h * self.max_crop_scale

        # At least 50% of the crop must be within the frame
        overlap_x = min(cx + max_w // 2, self.frame_w) - max(cx - max_w // 2, 0)
        overlap_y = min(cy + max_h // 2, self.frame_h) - max(cy - max_h // 2, 0)

        return (overlap_x / max_w) > 0.5 and (overlap_y / max_h) > 0.5

    def encode_label(self, species_name: str) -> Optional[Dict[str, int]]:
        """Hierarchical label vector {family, genus, species} indices, or None
        if the label is invalid/ambiguous."""
        if not self.is_valid_label(species_name):
            return None

        tax = self.taxonomy_map[species_name]
        return {
            "family": self.family_to_idx[tax["family"]],
            "genus": self.genus_to_idx[tax["genus"]],
            "species": self.species_to_idx[tax["species"]],
        }

    def parse_annotations(
        self,
        annotations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter + enrich raw annotations (image_path, bbox, label) into
        training-ready samples with hierarchical labels."""
        valid_samples = []
        skipped = {"ambiguous": 0, "spatial": 0, "missing_taxonomy": 0}

        for ann in annotations:
            label = ann.get("label", "")

            # Step 1a: Ambiguity filter
            if label in self.filter_labels:
                skipped["ambiguous"] += 1
                continue

            # Step 1b: Taxonomic lookup
            encoded = self.encode_label(label)
            if encoded is None:
                skipped["missing_taxonomy"] += 1
                continue

            # Step 1c: Spatial validity
            bbox = tuple(ann["bbox"])
            if not self.check_spatial_validity(bbox):
                skipped["spatial"] += 1
                continue

            valid_samples.append({
                "image_path": ann["image_path"],
                "bbox": bbox,
                "species_name": label,
                "taxonomy": self.taxonomy_map[label],
                "label_vector": encoded,
            })

        logger.info(
            f"TaxonomicParser: {len(valid_samples)} valid / "
            f"{len(annotations)} total. Skipped: {skipped}"
        )
        return valid_samples

    @property
    def num_families(self) -> int:
        return len(self.family_to_idx)

    @property
    def num_genera(self) -> int:
        return len(self.genus_to_idx)

    @property
    def num_species(self) -> int:
        return len(self.species_to_idx)


