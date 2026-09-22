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

import json
import os

import numpy as np
import torch
import torch.nn as nn

from bioreef._2_stage1 import build_detector
from bioreef._2_stage1._22_backbone import ViTBackbone
from bioreef._2_stage1._23_mceam import MCEAM
from bioreef._2_stage1._25_classifier import Classifier, ModelConfig
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
    # Post-hoc logit adjustment (Menon 2020 / Balanced Softmax). `logit_prior` is
    # the TRAIN class frequency, normalised, in class-index order; Stage 1
    # subtracts `logit_tau * log(prior)` from the raw logits. tau=0 (or no prior)
    # reproduces the unadjusted model exactly. Trades a little top-1 for a large
    # tail gain -- see models/README.md.
    logit_prior: Optional[np.ndarray] = None
    logit_tau: float = 0.0


def _load_logit_prior(cfg, num_classes: int, idx_to_sp: Dict[int, str]):
    """Load the post-hoc logit-adjustment prior, or (None, 0.0) if not configured.

    Returns (prior, tau). Stage 1 then computes `logits - tau * log(prior)`
    (Menon 2020 / Balanced Softmax): a per-class constant shift that lifts rare
    classes without retraining. tau=0 or a missing prior is an exact no-op, so
    the unadjusted model is always one config change away.

    The JSON holds the TRAIN class frequencies in class-index order plus the
    idx_to_sp they were exported against. Class order is the whole correctness
    condition here -- a prior misaligned with the logit columns would silently
    corrupt every prediction -- so both the length and the species mapping are
    checked against the loaded head, and a mismatch is fatal rather than a
    warning.
    """
    path = getattr(cfg, "stage1_prior", None)
    tau = float(getattr(cfg, "stage1_logit_tau", 0.0) or 0.0)
    if not path:
        if tau:
            logger.warning("stage1_logit_tau=%s set but no stage1_prior given — "
                           "logit adjustment is OFF.", tau)
        return None, 0.0
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"stage1_prior not found: {path}. Point it at the prior JSON exported "
            f"alongside the checkpoint, or unset it to disable logit adjustment.")

    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    prior = np.asarray(blob["prior"], dtype=np.float64)

    if prior.shape[0] != num_classes:
        raise ValueError(
            f"logit prior has {prior.shape[0]} classes but the head has "
            f"{num_classes} ({path}). These must be the same model.")
    if prior.min() <= 0:
        raise ValueError(f"logit prior has non-positive entries ({path}); every "
                         f"class needs training examples for log(prior).")

    # Order check: the prior is indexed by class id, exactly like the logit
    # columns. If the checkpoint's mapping disagrees with the one the prior was
    # exported against, the vector is being applied to the wrong classes.
    stored_map = blob.get("idx_to_sp")
    if stored_map and idx_to_sp:
        stored_map = {int(k): v for k, v in stored_map.items()}
        mismatched = [i for i in range(num_classes)
                      if i in idx_to_sp and i in stored_map
                      and idx_to_sp[i] != stored_map[i]]
        if mismatched:
            raise ValueError(
                f"logit prior class order does not match the checkpoint: "
                f"{len(mismatched)} indices differ (first: index {mismatched[0]}, "
                f"checkpoint={idx_to_sp[mismatched[0]]!r}, "
                f"prior={stored_map[mismatched[0]]!r}). Re-export {path} from the "
                f"same run as the checkpoint.")

    prior = prior / prior.sum()
    if tau:
        logger.info("  Logit adjustment: ON  (tau=%.3g, %d classes, from %s)",
                    tau, num_classes, path)
    else:
        logger.info("  Logit adjustment: prior loaded but tau=0 — no-op.")
    return prior, tau


def resolve_device(device_str: Optional[str]) -> torch.device:
    return torch.device(
        device_str if device_str
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def load_models(cfg) -> Models:
    """Build all models from an InferenceConfig (or anything exposing the same
    attrs). Identical to infer_stage1.main()'s loading block."""
    device = resolve_device(getattr(cfg, "device", None))

    # Detector (RF-DETR per #6 by default; backend-agnostic wrapper).
    detector = build_detector(
        cfg.detector_backend,
        weights=cfg.detection_ckpt,   # None -> backend default
        model_size=cfg.rfdetr_size,
        imgsz=cfg.imgsz,
        device=getattr(cfg, "device", None),
    )
    logger.info(f"  Detector classes: {detector.names} (class-agnostic — fish only)")

    # ---- Stage 1 classifier -------------------------------------------------
    # TWO checkpoint formats are supported:
    #
    #   "model" key  -- the research-repo format (a full Classifier.state_dict()
    #                   covering backbone+MCEAM+head, plus run_config). This is
    #                   what the deployment model D6a uses. Its backbone is FULL
    #                   FINE-TUNED, so the backbone weights ARE the model and must
    #                   come from the checkpoint, not from a fresh HF download.
    #   "mceam" key  -- the legacy format: only MCEAM+head are stored and the
    #                   backbone is a frozen, re-downloaded DINOv3 ViT-B.
    #
    # The legacy loader could not represent D6a at all: it hardcoded ViT-B
    # (768-d) and never loaded backbone weights, which would have silently
    # produced a near-random classifier from a 1280-d fine-tuned checkpoint.
    logger.info(f"Loading Stage 1 model: {cfg.stage1_ckpt}")
    s1_ckpt = torch.load(cfg.stage1_ckpt, map_location=device, weights_only=False)

    if "model" in s1_ckpt:
        state = s1_ckpt["model"]
        rc = s1_ckpt.get("run_config", {}) or {}
        # num_classes comes from the head itself, never from a stored field that
        # could disagree with the weights.
        num_classes = state["head.weight"].shape[0]
        mcfg = ModelConfig(
            backbone=rc.get("backbone", "dinov3"),
            context_levels=rc.get("context_levels", 3),
            attention_depth=rc.get("attention_depth", 1),
            # 0, not the trained -1: identical parameter shapes and names, but
            # skips a pointless unfreeze on a model we only ever run under
            # no_grad(). load_state_dict still matches strictly.
            unfreeze_blocks=0,
            probe=rc.get("probe", "mlp"),
        )
        logger.info(f"  Format: research-repo (full Classifier state_dict)")
        logger.info(f"  Backbone: {mcfg.backbone}  context_levels={mcfg.context_levels}"
                    f"  attention_depth={mcfg.attention_depth}")
        clf = Classifier(mcfg, num_classes).to(device)
        clf.load_state_dict(state)        # strict: a mismatch must not pass silently
        clf.eval()
        # Stage 1 calls backbone -> mceam -> head in sequence, so hand out the
        # submodules rather than the wrapper; they are the same objects.
        backbone, mceam, head = clf.backbone, clf.mceam, clf.head
        if mceam is None:
            raise ValueError(
                f"{cfg.stage1_ckpt} was trained with context_levels=0 (no MCEAM). "
                f"The pipeline's Stage 1 requires the multi-context model.")
        idx_to_sp = resolve_species_mapping(s1_ckpt, cfg.csv_path, cfg.min_samples)
    else:
        logger.info("  Format: legacy (MCEAM+head only; frozen ViT-B backbone)")
        backbone = ViTBackbone(freeze=True).to(device)
        backbone.eval()
        num_classes = s1_ckpt["head"]["weight"].shape[0]
        idx_to_sp = resolve_species_mapping(s1_ckpt, cfg.csv_path, cfg.min_samples)
        mceam = MCEAM(
            embed_dim=backbone.embed_dim,
            num_context_levels=3,
            output_dim=256,
            num_heads=8,
        ).to(device)
        mceam.load_state_dict(s1_ckpt["mceam"])
        mceam.eval()
        head = nn.Linear(256, num_classes).to(device)
        head.load_state_dict(s1_ckpt["head"])
        head.eval()
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

    logger.info(f"  Classifier ready: backbone embed_dim={backbone.embed_dim}, "
                f"head Linear({head.in_features}, {num_classes})")

    # ---- post-hoc logit adjustment -----------------------------------------
    # Optional; sits beside the checkpoint as a small JSON. Applied in Stage 1 to
    # the raw logits (see _92_detect.extract_embeddings).
    logit_prior, logit_tau = _load_logit_prior(cfg, num_classes, idx_to_sp)

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
        logit_prior=logit_prior, logit_tau=logit_tau,
    )
