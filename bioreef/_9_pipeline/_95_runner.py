"""
Inference pipeline runner — the end-to-end chain.

    run_inference(cfg) -> Stage2Output (or Stage3Output once Stage 3 exists)

        preprocess (prepare_frames)
          -> Stage 1 (run_stage1)   detect + embed + classify
          -> Stage 2 (run_stage2)   track + hierarchical aggregation
          -> Stage 3 (stub)         temporal refinement (#7/#19, future)

Stages pass in-memory objects directly (no .npz between them); each is wrapped
in io.cached() so a re-run skips a completed stage when cfg.cache_dir is set.
cfg.from_stage / cfg.to_stage select a sub-range (e.g. re-run only Stage 2 from
a cached Stage 1).

Models are loaded once and shared across stages.
"""

from __future__ import annotations

import logging
import os

from bioreef._9_pipeline.config import InferenceConfig
from bioreef._9_pipeline.models import load_models
from bioreef._9_pipeline.io import (
    Frames, Stage1Output, Stage2Output, cached,
)
# Stages, numbered in flow order: _91 preprocess -> _92 detect -> _93 track ->
# _94 refine. (This file, _95_runner, drives them.) The 9x prefix marks the
# orchestration group; the 2nd digit is the call order, data input -> output.
from bioreef._9_pipeline._91_preprocess import prepare_frames
from bioreef._9_pipeline._92_detect import run_stage1
from bioreef._9_pipeline._93_track import run_stage2
from bioreef._9_pipeline._94_refine import run_stage3

logger = logging.getLogger("bioreef._9_pipeline.inference")

_STAGE_ORDER = ["preprocess", "stage1", "stage2", "stage3"]


def _want(cfg, stage: str) -> bool:
    """True if `stage` is within [from_stage, to_stage]."""
    lo = _STAGE_ORDER.index(getattr(cfg, "from_stage", "preprocess"))
    hi = _STAGE_ORDER.index(getattr(cfg, "to_stage", "stage2"))
    return lo <= _STAGE_ORDER.index(stage) <= hi


def run_inference(cfg: InferenceConfig):
    """Run the inference chain on a single clip (cfg.video). Returns the last
    stage's output that was produced."""
    if not cfg.video:
        raise SystemExit("inference.video is not set in the config.")
    if not cfg.video_id:
        cfg.video_id = os.path.basename(cfg.video)

    models = load_models(cfg)

    logger.info("=" * 60)
    logger.info("BioReef.ai — Inference Pipeline")
    logger.info(f"  Video   : {cfg.video}")
    logger.info(f"  Device  : {models.device}")
    logger.info(f"  Stages  : {cfg.from_stage} .. {cfg.to_stage}")
    logger.info(f"  Cache   : {cfg.cache_dir if not cfg.no_cache else 'off'}")
    logger.info("=" * 60)

    # --- preprocess ---------------------------------------------------------
    frames = prepare_frames(cfg.video, cfg)

    # --- Stage 1 ------------------------------------------------------------
    s1 = None
    if _want(cfg, "stage1"):
        s1 = cached(
            cfg, "stage1",
            compute=lambda: run_stage1(frames.iter_frames(), models, cfg,
                                       video_id=cfg.video_id),
            loader=lambda p: Stage1Output.load(p, cfg.video_id),
        )

    # --- Stage 2 ------------------------------------------------------------
    s2 = None
    if _want(cfg, "stage2"):
        if s1 is None:
            raise SystemExit("Stage 2 needs Stage 1 output; widen from_stage.")
        s2 = cached(
            cfg, "stage2",
            compute=lambda: run_stage2(s1, models, cfg),
            loader=None,   # Stage2Output has no simple .npz loader yet
        )
        # Persist the final detections + species mapping where the rest of the
        # toolchain expects them (parity with the old scripts' outputs).
        _write_stage_outputs(cfg, models, s1, s2)

    # --- Stage 3 (stub passthrough until #7/#19) ----------------------------
    if _want(cfg, "stage3") and getattr(cfg, "run_stage3", False):
        if s2 is None:
            raise SystemExit("Stage 3 needs Stage 2 output; widen from_stage.")
        return run_stage3(s2, models, cfg)

    return s2 if s2 is not None else s1


def _write_stage_outputs(cfg, models, s1: Stage1Output, s2: Stage2Output):
    """Write the canonical on-disk artifacts (detections .npz, species_mapping,
    tracklets, verdicts) so downstream tools see the same files as before."""
    import numpy as np
    os.makedirs(cfg.output_dir, exist_ok=True)
    safe = (cfg.video_id or "").replace(".avi", "").replace(".", "_")

    s1.save(os.path.join(cfg.output_dir, f"{safe}.npz"))
    np.savez_compressed(
        os.path.join(cfg.output_dir, "species_mapping.npz"),
        sp_to_idx={v: k for k, v in models.idx_to_sp.items()},
        idx_to_sp=models.idx_to_sp,
    )
    if s2.tracklets:
        trk_dir = "outputs/tracklets"
        s2.save(os.path.join(trk_dir, f"{safe}.npz"))
    logger.info("wrote outputs -> %s", cfg.output_dir)
