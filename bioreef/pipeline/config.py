"""
Pipeline configuration — one definition of every knob, shared by the runners
and the thin CLIs so defaults never drift between them.

Defaults below mirror the current argparse defaults in scripts/pipeline/
infer_stage1.py and track_stage2.py exactly (so behaviour is unchanged):
  - detector: rfdetr medium, conf 0.3, apply_waternet False (#14)
  - tracker:  high 0.6 / low 0.1 / max_lost_age 30 / iou 0.3 /
              appearance 0.4 / ema 0.9
  - aggregation: species 0.50 / genus 0.60 / family 0.70
  - csv_path defaults to the recovered 256-class subset (#24)

`video_id` and `cache_dir` drive io.cached(); set cache_dir=None to disable
caching entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

_SUBSET_CSV = "data_oz/metadata/frame_metadata_subset.csv"   # recovered, #24


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

    # --- stage 1 classifier ---
    stage1_ckpt: str = "bioreef_stage1.pt"

    # --- stage 2 tracker ---
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    max_lost_age: int = 30
    iou_threshold: float = 0.3
    appearance_threshold: float = 0.4
    ema_alpha: float = 0.9
    no_cmc: bool = False
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


@dataclass
class TrainingConfig(BaseConfig):
    img_dir: str = "data_oz/frames_waternet_1"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    hslm: bool = False
    hslm_weights: List[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])
    output_ckpt: str = "bioreef_stage1.pt"
    # cache_dir/no_cache/video_id present so io.cached() is usable here too.
    cache_dir: Optional[str] = None
    no_cache: bool = False
    video_id: Optional[str] = None
