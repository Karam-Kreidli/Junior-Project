"""
Pipeline configuration — one definition of every knob, shared by the runners
and the thin CLIs so defaults never drift between them.

Defaults below mirror the current argparse defaults in scripts/pipeline/
infer_stage1.py and track_stage2.py exactly (so behaviour is unchanged):
  - detector: rfdetr medium, conf 0.3, apply_waternet False (#14)
  - tracker:  SU-T, track_thresh 0.5 / iou 0.25 / max_age 30 / min_hits 3, stitching off
  - aggregation: species 0.50 / genus 0.60 / family 0.70
  - csv_path defaults to the recovered 256-class subset (#24)

`video_id` and `cache_dir` drive io.cached(); set cache_dir=None to disable
caching entirely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

_SUBSET_CSV = "data_oz/metadata/frame_metadata_subset.csv"   # recovered, #24

DEFAULT_CONFIG_PATH = "config.yaml"


def _load_yaml(path: str) -> dict:
    import yaml
    if not os.path.exists(path):
        raise SystemExit(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config file {path} must be a YAML mapping")
    return data


def _build(cls, merged: dict):
    """Instantiate a config dataclass from a dict, ignoring unknown keys and
    warning about them so a typo in the YAML doesn't silently do nothing."""
    known = {f.name for f in fields(cls)}
    unknown = set(merged) - known
    if unknown:
        import logging
        logging.getLogger("bioreef._9_pipeline.config").warning(
            "config: ignoring unknown keys for %s: %s",
            cls.__name__, sorted(unknown),
        )
    return cls(**{k: v for k, v in merged.items() if k in known})


@dataclass
class BaseConfig:
    """Knobs shared by training and inference."""
    csv_path: str = _SUBSET_CSV
    min_samples: int = 20
    device: Optional[str] = None          # None -> cuda if available
    apply_waternet: bool = False          # #14: off for the detector


@dataclass
class InferenceConfig(BaseConfig):
    # --- input ---
    video: Optional[str] = None
    frames_dir: Optional[List[str]] = None
    video_id: Optional[str] = None        # also the cache key

    # --- detector ---
    detector_backend: str = "rfdetr"
    detection_ckpt: Optional[str] = None  # None -> backend default weights
    rfdetr_size: str = "medium"
    imgsz: int = 960
    conf_threshold: float = 0.3
    fast_inference: bool = True           # bf16 classifier + traced fp16 RF-DETR + TF32 (CUDA only)

    # --- stage 1 classifier ---
    stage1_ckpt: str = "models/d6a_stage1.pt"

    # Post-hoc logit adjustment for the long tail. `stage1_prior` is a small JSON
    # holding the train class frequencies in class-index order (exported beside
    # the checkpoint); Stage 1 subtracts `stage1_logit_tau * log(prior)` from the
    # raw logits. tau=0, or no prior file, is an exact no-op. tau=1.0 is the
    # deployment setting -- see models/README.md for the trade-off it buys.
    stage1_prior: Optional[str] = None
    stage1_logit_tau: float = 0.0

    # --- stage 2 tracker (SU-T, motion-only) ---
    # track_thresh 0.5 (SU-T's MFT25 value is 0.6): best trade-off on 131 labelled OzFish fish,
    # metrics.md 2026-10-19; lower covers more frames but splits fish into flipping duplicate tracks
    track_thresh: float = 0.5           # conf to start a track; 0.1..this only extends one
    iou_threshold: float = 0.25         # FishIoU match threshold
    max_age: int = 30                   # frames a lost track survives
    min_hits: int = 3                   # matches before a track is reported
    stitch_max_gap: int = 0             # offline track joining, max frames apart; 0 = off (~neutral on labelled fish)
    classify_every: int = 3             # Stage 1 classifies each track when it appears, then every N frames; 1 = every box
    min_tracklet_len: int = 16
    max_tracklet_len: int = 30

    # --- stage 2 aggregation (#5) ---
    species_thresh: float = 0.50
    genus_thresh: float = 0.60
    family_thresh: float = 0.70

    # --- stage 3 (future) ---
    run_stage3: bool = False

    # --- output / cache ---
    output_dir: str = "outputs/detections"
    cache_dir: Optional[str] = "outputs/cache"
    no_cache: bool = False
    from_stage: str = "preprocess"        # preprocess|stage1|stage2|stage3
    to_stage: str = "stage2"

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_CONFIG_PATH) -> "InferenceConfig":
        """Build from config.yaml: shared section + inference section merged
        (inference keys win on conflict)."""
        data = _load_yaml(path)
        merged = {**data.get("shared", {}), **data.get("inference", {})}
        return _build(cls, merged)


@dataclass
class TrainingConfig(BaseConfig):
    # Mirrors train_stage1.py's argparse exactly (DDP trainer, torchrun-launched).
    img_dir: str = "data_oz/frames_waternet_1"
    use_waternet: bool = False
    epochs: Optional[int] = None          # None -> 10 (decouple) / 30 (standard)
    decouple: bool = False
    checkpoint: str = "bioreef_stage1_ddp.pt"
    beta: float = 0.99
    gamma: float = 1.0
    warmup_epochs: int = 3
    ema_decay: float = 0.999
    batch_size: int = 8
    hslm: bool = False
    family_weight: float = 3.0
    genus_weight: float = 2.0
    species_weight: float = 1.0

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_CONFIG_PATH) -> "TrainingConfig":
        """Build from config.yaml: shared section + training section merged
        (training keys win on conflict)."""
        data = _load_yaml(path)
        merged = {**data.get("shared", {}), **data.get("training", {})}
        return _build(cls, merged)
