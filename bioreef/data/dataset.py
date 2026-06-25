"""BioReefDataset — assembles the 4 preprocessing steps (split from data_factory)."""
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

logger = logging.getLogger("bioreef.data")


from bioreef.data.restoration import WaterNetRestorer
from bioreef.data.context import ContextHarvester
from bioreef.data.taxonomy_parse import TaxonomicParser
from bioreef.data.augmentation import MarineAugmentor


class BioReefDataset(Dataset):
    """
    PyTorch Dataset orchestrating the full Stage 1 preprocessing pipeline.

    Pipeline per sample:
        1. Load frame image
        2. (Optional) Restore via Water-Net
        3. Extract 4-stream context crops (Context Harvester)
        4. (During training) Apply marine augmentation
        5. Normalize to ImageNet-compatible tensors
        6. Return {streams: Dict[str, Tensor], labels: Dict[str, int], metadata}

    Each returned sample feeds directly into the DINOv2 backbone → MCEAM
    cross-attention fusion pipeline.
    """

    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        taxonomy_map: Dict[str, Dict[str, str]],
        config: Optional[Dict] = None,
        restore: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            annotations: Raw annotation list [{'image_path', 'bbox', 'label'}].
            taxonomy_map: Species → {family, genus, species} mapping.
            config: Stage 1 YAML config dict; uses defaults if None.
            restore: Whether to apply Water-Net restoration.
            augment: Whether to apply marine augmentation (set False for eval).
        """
        config = config or {}
        data_cfg = config.get("data", {})
        aug_cfg = config.get("augmentation", {})
        wn_cfg = config.get("waternet", {})

        # Step 1: Taxonomic Parser
        self.parser = TaxonomicParser(
            taxonomy_map=taxonomy_map,
            filter_labels=config.get("taxonomy", {}).get("filter_labels"),
            frame_width=data_cfg.get("frame_width", 1920),
            frame_height=data_cfg.get("frame_height", 1080),
        )
        self.samples = self.parser.parse_annotations(annotations)

        # Step 2: Water-Net Restorer
        self.restorer = WaterNetRestorer(
            checkpoint_path=wn_cfg.get("checkpoint_path"),
        ) if restore else None

        # Step 3 & 4: Context Harvester (cropping + normalization)
        self.harvester = ContextHarvester(
            crop_scales=data_cfg.get("crop_scales", [1, 3, 5]),
            target_resolution=data_cfg.get("target_resolution", 224),
            small_object_threshold=data_cfg.get("small_object_threshold", 0.05),
            highres_initial=data_cfg.get("highres_initial_crop", 512),
            include_full_frame=data_cfg.get("include_full_frame", True),
        )

        # Step 5: Marine Augmentor
        self.augmentor = MarineAugmentor(
            **{k: v for k, v in aug_cfg.items() if k != "enabled"},
            enabled=augment and aug_cfg.get("enabled", True),
        ) if augment else None

        # Image cache (optional, disabled by default for memory)
        self._image_cache: Dict[str, np.ndarray] = {}

        logger.info(
            f"BioReefDataset initialized: {len(self.samples)} samples, "
            f"restore={restore}, augment={augment}."
        )

    def _load_image(self, path: str) -> np.ndarray:
        """Load and optionally cache a frame image."""
        if path in self._image_cache:
            return self._image_cache[path]

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single preprocessed sample.

        Returns:
            {
                'streams': {
                    'roi': Tensor(3, 224, 224),
                    'social': Tensor(3, 224, 224),
                    'habitat': Tensor(3, 224, 224),
                    'full_frame': Tensor(3, 224, 224),
                },
                'labels': {
                    'family':  int,
                    'genus':   int,
                    'species': int,
                },
                'metadata': {
                    'species_name': str,
                    'taxonomy': {'family', 'genus', 'species'},
                    'bbox': (x, y, w, h),
                    'image_path': str,
                },
            }
        """
        sample = self.samples[idx]
        frame = self._load_image(sample["image_path"])

        # Step 2: Spectral restoration
        if self.restorer is not None:
            frame = self.restorer(frame)

        # Step 5: Augmentation (applied to full frame before cropping)
        if self.augmentor is not None:
            frame = self.augmentor(frame)

        # Steps 3 & 4: Context Harvester (crop + normalize)
        streams = self.harvester.harvest(frame, sample["bbox"])

        return {
            "streams": streams,
            "labels": sample["label_vector"],
            "metadata": {
                "species_name": sample["species_name"],
                "taxonomy": sample["taxonomy"],
                "bbox": sample["bbox"],
                "image_path": sample["image_path"],
            },
        }

    @property
    def num_families(self) -> int:
        return self.parser.num_families

    @property
    def num_genera(self) -> int:
        return self.parser.num_genera

    @property
    def num_species(self) -> int:
        return self.parser.num_species
