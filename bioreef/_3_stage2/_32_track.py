"""
BioReef.ai — Track State Object
================================
Represents a single tracked fish across frames. Holds the Kalman Filter
kinematic state, EMA appearance embedding, bounding box history, and
lifecycle status (active / lost / dead).

Each Track is the atomic unit that Stage 2 manages. When a Track
accumulates 16–30 frames, it becomes eligible for Tracklet export
to Stage 3.

Reference:
    Aharon et al. (2022), "BoT-SORT: Robust Associations Multi-Pedestrian
    Tracking."
"""

from enum import Enum, auto
from typing import Optional

import numpy as np


class TrackState(Enum):
    """Lifecycle states for a tracked individual."""
    ACTIVE = auto()    # Currently matched to detections every frame
    LOST = auto()      # Not matched recently; Kalman predicting position
    DEAD = auto()      # Lost for too long; retired from active pool


class Track:
    """
    A single tracked fish identity across video frames.

    Attributes:
        track_id:      Unique integer ID for this individual.
        state:         Current lifecycle state (ACTIVE / LOST / DEAD).
        bbox:          Latest bounding box [x, y, w, h].
        confidence:    Latest detection confidence score.
        kf_state:      8-dim Kalman Filter state [u, v, a, h, u̇, v̇, ȧ, ḣ].
        kf_covariance: Kalman Filter covariance matrix (8×8).
        ema_embedding: EMA-smoothed DINOv2 embedding (256-dim).
        frame_history: List of (frame_id, bbox, embedding, logits) for
                       tracklet export. `logits` is the per-frame species
                       classifier output (Stage 1 prior) — used by the
                       hierarchical-fallback aggregation (issue #5). May be
                       None for frames where the classifier was not run.
        hits:          Total number of successful matches.
        age:           Total frames since track creation.
        time_since_update: Frames since last successful match.
    """

    _next_id = 1  # Class-level auto-incrementing ID

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
        frame_id: int = 0,
        logits: Optional[np.ndarray] = None,
    ):
        """
        Args:
            bbox:       Initial bounding box [x, y, w, h].
            confidence: Detection confidence score.
            embedding:  Initial DINOv2/MCEAM embedding (256-dim).
            frame_id:   Frame number where this track was born.
            logits:     Per-frame species classifier logits (Stage 1 prior),
                        carried for hierarchical-fallback aggregation (#5).
        """
        self.track_id = Track._next_id
        Track._next_id += 1

        self.state = TrackState.ACTIVE
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.confidence = confidence

        # Kalman state: initialized by KalmanFilter.initiate()
        self.kf_state: Optional[np.ndarray] = None
        self.kf_covariance: Optional[np.ndarray] = None

        # Appearance: EMA embedding
        self.ema_embedding = (
            embedding.copy() if embedding is not None else None
        )

        # History for tracklet export: (frame_id, bbox, embedding, logits)
        self.frame_history: list = []
        if embedding is not None:
            self.frame_history.append((
                frame_id,
                self.bbox.copy(),
                embedding.copy(),
                logits.copy() if logits is not None else None,
            ))

        # Counters
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray],
        frame_id: int,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update this track with a new matched detection.

        Args:
            bbox:       Matched bounding box [x, y, w, h].
            confidence: Detection confidence.
            embedding:  New frame's DINOv2/MCEAM embedding.
            frame_id:   Current frame number.
            logits:     New frame's species classifier logits (Stage 1
                        prior), for hierarchical-fallback aggregation (#5).
        """
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.ACTIVE

        if embedding is not None:
            self.frame_history.append((
                frame_id,
                self.bbox.copy(),
                embedding.copy(),
                logits.copy() if logits is not None else None,
            ))

    def mark_lost(self) -> None:
        """Transition track to LOST state (no match this frame)."""
        self.state = TrackState.LOST
        self.time_since_update += 1

    def mark_dead(self) -> None:
        """Retire this track permanently."""
        self.state = TrackState.DEAD

    def predict_step(self) -> None:
        """Advance age counter (called each frame regardless of match)."""
        self.age += 1

    @property
    def is_active(self) -> bool:
        return self.state == TrackState.ACTIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_dead(self) -> bool:
        return self.state == TrackState.DEAD

    @property
    def tracklet_length(self) -> int:
        """Number of frames with matched embeddings."""
        return len(self.frame_history)

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the auto-incrementing ID (for new video sequences)."""
        cls._next_id = 1

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, state={self.state.name}, "
            f"hits={self.hits}, age={self.age}, "
            f"tsu={self.time_since_update}, "
            f"tracklet_len={self.tracklet_length})"
        )
