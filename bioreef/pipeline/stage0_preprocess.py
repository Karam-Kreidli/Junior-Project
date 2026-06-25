"""
Stage 0 — preprocessing (the first step of the inference sequence).

The implementation lives in bioreef.data.preprocess (with the data modules it
uses); this module re-exports it so all pipeline stages sit together, in order,
under bioreef/pipeline/:

    stage0_preprocess -> stage1_detect -> stage2_track -> stage3_refine

    prepare_frames(video, cfg) -> Frames
"""

from bioreef.data.preprocess import prepare_frames, extract_frames  # noqa: F401

__all__ = ["prepare_frames", "extract_frames"]
