"""
BioReef.ai — Spatiotemporal Tracklet Assembly
===============================================
The primary output of Stage 2 is a Spatiotemporal Tracklet: a
chronologically ordered sequence of 16–30 frames of the same biological
individual.

Structure:
    Track ID #105: [
        (Frame_001: bbox + z_context),
        (Frame_002: bbox + z_context),
        ...
        (Frame_020: bbox + z_context),
    ]

Each entry is enriched with the MCEAM embedding (z_context) from Stage 1,
so the tracklet carries knowledge of the surrounding habitat (coral, sand,
water column) at each moment. Stage 3 inherits this ecological context.

The TrackletWriter extracts eligible tracklets from completed tracks,
enforces the 16–30 frame window, and serializes them for Stage 3 input.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bioreef.tracking.tracklet")


@dataclass
class Tracklet:
    """
    A single spatiotemporal tracklet ready for Stage 3 classification.

    Attributes:
        track_id:   The track ID from Stage 2.
        frames:     List of (frame_id, bbox, embedding) tuples,
                    chronologically ordered.
    """
    track_id: int
    frames: List[Tuple[int, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def frame_ids(self) -> List[int]:
        return [f[0] for f in self.frames]

    @property
    def bboxes(self) -> np.ndarray:
        """(T, 4) array of bounding boxes."""
        if not self.frames:
            return np.empty((0, 4), dtype=np.float64)
        return np.array([f[1] for f in self.frames], dtype=np.float64)

    @property
    def embeddings(self) -> np.ndarray:
        """(T, D) array of MCEAM embeddings."""
        if not self.frames:
            return np.empty((0, 0), dtype=np.float64)
        return np.array([f[2] for f in self.frames], dtype=np.float64)

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "track_id": self.track_id,
            "length": self.length,
            "frame_ids": self.frame_ids,
            "bboxes": self.bboxes.tolist(),
            "embeddings": self.embeddings.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Tracklet":
        """Deserialize from a dictionary."""
        frames = []
        for fid, bbox, emb in zip(
            data["frame_ids"], data["bboxes"], data["embeddings"]
        ):
            frames.append((
                int(fid),
                np.array(bbox, dtype=np.float64),
                np.array(emb, dtype=np.float64),
            ))
        return cls(track_id=data["track_id"], frames=frames)


class TrackletWriter:
    """
    Extracts and saves tracklets from completed Stage 2 tracks.

    A track is eligible for tracklet export if it has accumulated at
    least `min_length` frames with valid embeddings. Tracks longer than
    `max_length` are split into multiple overlapping windows.
    """

    def __init__(
        self,
        min_length: int = 16,
        max_length: int = 30,
        overlap: int = 8,
        output_dir: str = "outputs/tracklets",
    ):
        """
        Args:
            min_length: Minimum frames for a valid tracklet (16 per spec).
            max_length: Maximum frames per tracklet window (30 per spec).
            overlap:    Overlap between consecutive windows when splitting
                        long tracks. Ensures temporal continuity.
            output_dir: Directory for saved tracklet files.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.overlap = overlap
        self.output_dir = output_dir

    def extract_tracklets(
        self, tracks: list
    ) -> List[Tracklet]:
        """
        Extract tracklets from a list of Track objects.

        Tracks with fewer than `min_length` matched frames are discarded.
        Tracks longer than `max_length` are split into overlapping windows.

        Args:
            tracks: List of Track objects (from BoTSORTTracker.get_all_tracks()).

        Returns:
            List of Tracklet objects ready for Stage 3.
        """
        tracklets = []

        for track in tracks:
            history = track.frame_history
            if len(history) < self.min_length:
                continue

            # Sort by frame_id to ensure chronological order
            history = sorted(history, key=lambda x: x[0])

            if len(history) <= self.max_length:
                # Single tracklet — fits within the window
                tracklet = Tracklet(track_id=track.track_id)
                for frame_id, bbox, embedding in history:
                    tracklet.frames.append((frame_id, bbox.copy(), embedding.copy()))
                tracklets.append(tracklet)
            else:
                # Split into overlapping windows
                step = self.max_length - self.overlap
                for start in range(0, len(history) - self.min_length + 1, step):
                    end = min(start + self.max_length, len(history))
                    window = history[start:end]

                    if len(window) < self.min_length:
                        break

                    tracklet = Tracklet(track_id=track.track_id)
                    for frame_id, bbox, embedding in window:
                        tracklet.frames.append((
                            frame_id, bbox.copy(), embedding.copy()
                        ))
                    tracklets.append(tracklet)

        logger.info(
            f"Extracted {len(tracklets)} tracklets from "
            f"{len(tracks)} tracks (min_len={self.min_length}, "
            f"max_len={self.max_length})"
        )
        return tracklets

    def save(
        self,
        tracklets: List[Tracklet],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save tracklets to a .npz file (NumPy compressed archive).

        The archive contains:
            - track_ids:  (K,) array of track IDs
            - frame_ids:  list of (T_k,) arrays per tracklet
            - bboxes:     list of (T_k, 4) arrays per tracklet
            - embeddings: list of (T_k, D) arrays per tracklet

        Args:
            tracklets: List of Tracklet objects.
            filename:  Output filename. Defaults to "tracklets.npz".

        Returns:
            Path to the saved file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or "tracklets.npz"
        filepath = os.path.join(self.output_dir, filename)

        track_ids = np.array([t.track_id for t in tracklets])
        frame_ids_list = [np.array(t.frame_ids) for t in tracklets]
        bboxes_list = [t.bboxes for t in tracklets]
        embeddings_list = [t.embeddings for t in tracklets]

        np.savez_compressed(
            filepath,
            track_ids=track_ids,
            frame_ids=np.array(frame_ids_list, dtype=object),
            bboxes=np.array(bboxes_list, dtype=object),
            embeddings=np.array(embeddings_list, dtype=object),
        )

        logger.info(f"Saved {len(tracklets)} tracklets to: {filepath}")
        return filepath

    def save_json(
        self,
        tracklets: List[Tracklet],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save tracklets to a JSON file (human-readable, for debugging).

        Args:
            tracklets: List of Tracklet objects.
            filename:  Output filename. Defaults to "tracklets.json".

        Returns:
            Path to the saved file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or "tracklets.json"
        filepath = os.path.join(self.output_dir, filename)

        data = {
            "num_tracklets": len(tracklets),
            "tracklets": [t.to_dict() for t in tracklets],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(tracklets)} tracklets (JSON) to: {filepath}")
        return filepath
