"""
Model loading for the pipeline — one place that builds every model a stage
needs, so the runner loads them once and shares them across stages.

load_models(cfg) lifts the model-setup block out of infer_stage1.main()
verbatim (backbone, detector, MCEAM, classifier head, the #24 mapping guard,
optional WaterNet, ContextHarvester). Behaviour is identical to the script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from bioreef._2_stage1 import build_detector
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._1_preprocess._11_restoration import WaterNetRestorer
from bioreef._1_preprocess._12_context import ContextHarvester
from bioreef._1_preprocess._15_dataset_split import resolve_species_mapping

logger = logging.getLogger("bioreef._9_pipeline.models")


@dataclass
class Models:
    """Everything a Stage-1/2 run needs, loaded once."""
    device: torch.device
    backbone: ViTBackbone
    detector: object                 # bioreef._2_stage1.Detector
    mceam: MCEAM
    head: nn.Module
    harvester: ContextHarvester
    idx_to_sp: Dict[int, str]
    num_classes: int
    waternet: Optional[WaterNetRestorer] = None


def resolve_device(device_str: Optional[str]) -> torch.device:
    return torch.device(
        device_str if device_str
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def load_models(cfg) -> Models:
    """Build all models from an InferenceConfig (or anything exposing the same
    attrs). Identical to infer_stage1.main()'s loading block."""
    device = resolve_device(getattr(cfg, "device", None))

    logger.info("Loading backbone...")
    backbone = ViTBackbone(freeze=True).to(device)
    backbone.eval()

    # Detector (RF-DETR per #6 by default; backend-agnostic wrapper).
    detector = build_detector(
        cfg.detector_backend,
        weights=cfg.detection_ckpt,   # None -> backend default
        model_size=cfg.rfdetr_size,
        imgsz=cfg.imgsz,
        device=getattr(cfg, "device", None),
    )
    logger.info(f"  Detector classes: {detector.names} (class-agnostic — fish only)")

    # Stage 1 MCEAM checkpoint
    logger.info(f"Loading Stage 1 model: {cfg.stage1_ckpt}")
    s1_ckpt = torch.load(cfg.stage1_ckpt, map_location=device, weights_only=False)

    # Species mapping — checkpoint-first, CSV fallback (see #24).
    num_classes = s1_ckpt["head"]["weight"].shape[0]
    idx_to_sp = resolve_species_mapping(s1_ckpt, cfg.csv_path, cfg.min_samples)
    logger.info(f"  Head classes: {num_classes}  |  species mapping entries: "
                f"{len(idx_to_sp)}")

    # #24 guard: a CSV-fallback mapping whose size != head crashes Stage 2
    # aggregation and mislabels species. Replace with obvious placeholders.
    if idx_to_sp and len(idx_to_sp) != num_classes:
        logger.error(
            "SPECIES MAPPING MISMATCH (#24): head has %d classes but the "
            "CSV-derived mapping has %d species (csv=%s, min_samples=%d). "
            "Boxes/embeddings/Re-ID are unaffected, but species verdicts will "
            "be WRONG. Using placeholder names so it's obviously unusable.",
            num_classes, len(idx_to_sp), cfg.csv_path, cfg.min_samples,
        )
        idx_to_sp = {i: f"__unmapped_{i}__" for i in range(num_classes)}

    mceam = MCEAM(
        embed_dim=backbone.embed_dim,
        num_context_levels=3,
        output_dim=256,
        num_heads=8,
    ).to(device)
    mceam.load_state_dict(s1_ckpt["mceam"])
    mceam.eval()
    logger.info("  MCEAM loaded")

    head = nn.Linear(256, num_classes).to(device)
    head.load_state_dict(s1_ckpt["head"])
    head.eval()
    logger.info(f"  Head loaded   : Linear(256, {num_classes})")

    waternet = None
    if getattr(cfg, "apply_waternet", False):
        logger.info("Loading WaterNet for inline restoration...")
        waternet = WaterNetRestorer()
        waternet._load_model()        # surface load errors early

    harvester = ContextHarvester()

    return Models(
        device=device, backbone=backbone, detector=detector, mceam=mceam,
        head=head, harvester=harvester, idx_to_sp=idx_to_sp,
        num_classes=num_classes, waternet=waternet,
    )
