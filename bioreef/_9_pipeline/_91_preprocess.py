"""
Step 1 — preprocessing (the first step of the inference sequence).

The implementation lives in bioreef._1_preprocess._17_preprocess (with the data modules it
uses); this module re-exports it so all pipeline stages sit together, in call
order, under bioreef/_9_pipeline/ (the _N_ prefix is the data-input -> output flow):

    _91_preprocess -> _92_detect -> _93_track -> _94_refine  (run by _95_runner)

    prepare_frames(video, cfg) -> Frames
"""

from bioreef._1_preprocess._17_preprocess import prepare_frames, extract_frames  # noqa: F401

__all__ = ["prepare_frames", "extract_frames"]
