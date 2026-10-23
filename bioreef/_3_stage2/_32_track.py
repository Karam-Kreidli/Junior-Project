"""
Track — one tracked fish as the rest of the pipeline sees it.

Association state lives inside SU-T (_34_sut_tracker); this is only the record the
pipeline reads: id, latest box, and the per-frame history that becomes Stage-3
tracklets and the #5 species vote.
"""

import numpy as np


class Track:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.bbox = np.zeros(4)        # latest matched box, [x, y, w, h]
        self.confidence = 0.0
        self.confirmed = False
        # (frame_id, bbox [x,y,w,h], embedding|None, logits|None), one per matched frame;
        # embedding is None in motion-only runs and on frames filled in by stitching
        self.frame_history: list = []

    @property
    def tracklet_length(self) -> int:
        return len(self.frame_history)

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, len={self.tracklet_length}, confirmed={self.confirmed})"
