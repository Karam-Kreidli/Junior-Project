"""
Stage 3 — temporal refinement over tracklets (NOT YET IMPLEMENTED).

Placeholder for the Stage-3 transformer (#7 / #19): a sequence model over each
tracklet that lifts the per-frame Stage-1 verdicts above the majority-vote
ceiling. Until it exists, run_stage3 is a no-op that returns its Stage-2 input
unchanged, so the runner can already reference the stage in sequence.

    _91_preprocess -> _92_detect -> _93_track -> [_94_refine]  (run by _95_runner)
"""

import logging

logger = logging.getLogger("bioreef._9_pipeline.stage3")


def run_stage3(stage2_out, models, cfg):
    """No-op until the Stage-3 model lands (#7/#19). Returns Stage 2 output."""
    logger.info("Stage 3 not implemented yet (#7/#19) — passing through "
                "Stage 2 output unchanged.")
    return stage2_out
