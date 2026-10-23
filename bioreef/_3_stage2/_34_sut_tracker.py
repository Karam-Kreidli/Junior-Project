"""
SU-T wrapped to the pipeline's tracker contract.

SU-T (vendored in ./sut, AAAI'26 fish tracker: UKF motion, FishIoU, ByteTrack-style
second pass, score cue) does all association. This wrapper only:
  - converts the pipeline's [x, y, w, h] boxes to SU-T's [x1, y1, x2, y2] and back,
  - maps each SU-T track's matched detection back to its input row, so the per-frame
    embedding + species logits land in Track.frame_history (tracklets, #5 voting),
    including the frames before the track was confirmed.

Motion-only by design: same-species fish look alike, so Re-ID is not used
(Stage-2 decision 2026-10-13). Defaults = SU-T's published MFT25 config, except track_thresh
0.5 (tuned on labelled OzFish fish, metrics.md 2026-10-19).
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from ._32_track import Track
from .sut import SU_T


class SUTTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,    # conf to start a track; 0.1..this = second-pass only
        iou_threshold: float = 0.25,
        max_age: int = 30,            # frames a lost track survives
        min_hits: int = 3,            # consecutive matches before a track is reported
        delta_t: int = 3,
        inertia: float = 0.05,
    ):
        args = SimpleNamespace(track_thresh=track_thresh, TCM_first_step=True, TCM_byte_step=True,
                               TCM_first_step_weight=1.0, TCM_byte_step_weight=1.0)
        self._sut = SU_T(args, det_thresh=track_thresh, max_age=max_age, min_hits=min_hits,
                         iou_threshold=iou_threshold, delta_t=delta_t, asso_func="fishiou",
                         inertia=inertia, use_byte=True)
        self._tracks: Dict[int, Track] = {}
        self._frame = -1

    def update(
        self,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,           # unused (no CMC); kept for the contract
        reid_embeddings: Optional[np.ndarray] = None,  # unused (motion-only)
        logits: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
    ) -> List[Track]:
        """One frame of [x, y, w, h] detections -> tracks confirmed and matched this frame.
        Call once per frame, including frames with no detections."""
        self._frame = self._frame + 1 if frame_id is None else int(frame_id)
        xywh = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
        xyxy = np.hstack([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]])
        self._sut.update(np.hstack([xyxy, conf[:, None]]), (1, 1), (1, 1))

        active = []
        for trk in self._sut.trackers:
            if trk.time_since_update != 0 or len(xyxy) == 0:
                continue
            obs = trk.last_observation[:4]
            box = obs if obs.sum() >= 0 else trk.get_state()[0][:4]   # brand-new track: its init box
            i = int(np.abs(xyxy - box).sum(axis=1).argmin())          # the detection it matched
            t = self._tracks.setdefault(trk.id, Track(trk.id + 1))
            t.bbox, t.confidence = xywh[i].copy(), float(conf[i])
            t.frame_history.append((self._frame, xywh[i].copy(),
                                    embeddings[i].copy() if embeddings is not None else None,
                                    logits[i].copy() if logits is not None else None))
            if trk.hit_streak >= self._sut.min_hits or self._sut.frame_count <= self._sut.min_hits:
                t.confirmed = True
                active.append(t)
        return active

    def get_all_tracks(self) -> List[Track]:
        """Every track that was ever confirmed (unconfirmed blips are dropped)."""
        return [t for t in self._tracks.values() if t.confirmed]
