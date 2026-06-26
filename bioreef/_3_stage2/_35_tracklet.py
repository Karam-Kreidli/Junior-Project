"""
Spatiotemporal tracklet assembly — Stage 2's primary output.

A Tracklet is a chronological 16–30 frame sequence of one individual, each frame
carrying its MCEAM embedding (habitat context) for Stage 3. TrackletWriter pulls
eligible tracklets from completed tracks, enforces the window, and serializes.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bioreef._3_stage2._35_tracklet")


@dataclass
class Tracklet:
    """One spatiotemporal tracklet for Stage 3. `frames` are chronological
    (frame_id, bbox, embedding, logits) tuples; logits (Stage-1 prior) may be
    None where the classifier didn't run."""
    track_id: int
    frames: List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]] = field(
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

    @property
    def logits(self) -> Optional[np.ndarray]:
        """(T, C) per-frame species logits, or None. Frames with no logits are
        dropped, so this may be shorter than the tracklet."""
        present = [f[3] for f in self.frames if f[3] is not None]
        if not present:
            return None
        return np.array(present, dtype=np.float64)

    def aggregate_hierarchical(
        self,
        species_to_genus: List[int],
        species_to_family: List[int],
        num_genera: int,
        num_families: int,
        species_thresh: float = 0.50,
        genus_thresh: float = 0.60,
        family_thresh: float = 0.70,
    ) -> Dict:
        """
        Hierarchical-fallback species verdict for this tracklet (issue #5).

        Softmaxes each frame's logits, averages the probability vectors (a
        proper consensus over distributions, unlike majority vote over argmaxes
        which discards confidence), then emits the most specific label the
        evidence clears a threshold for: species -> genus -> family ->
        unidentified (coarser claims need higher thresholds). Returns
        {level, index, confidence, n_frames}; level='unidentified', n_frames=0
        when the tracklet has no logits.
        """
        logits = self.logits
        if logits is None or len(logits) == 0:
            return {
                "level": "unidentified", "index": None,
                "confidence": 0.0, "n_frames": 0,
            }

        # Per-frame softmax, then average over the tracklet.
        shifted = logits - logits.max(axis=1, keepdims=True)  # stable softmax
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)          # (T, C)
        mean_species = probs.mean(axis=0)                     # (C,)

        # Marginalize the mean species distribution up the taxonomy.
        s2g = np.asarray(species_to_genus, dtype=np.int64)
        s2f = np.asarray(species_to_family, dtype=np.int64)
        mean_genus = np.zeros(num_genera, dtype=np.float64)
        mean_family = np.zeros(num_families, dtype=np.float64)
        np.add.at(mean_genus, s2g, mean_species)
        np.add.at(mean_family, s2f, mean_species)

        n = len(logits)

        # Most specific level that clears its threshold.
        sp_idx = int(mean_species.argmax())
        if mean_species[sp_idx] >= species_thresh:
            return {"level": "species", "index": sp_idx,
                    "confidence": float(mean_species[sp_idx]), "n_frames": n}

        g_idx = int(mean_genus.argmax())
        if mean_genus[g_idx] >= genus_thresh:
            return {"level": "genus", "index": g_idx,
                    "confidence": float(mean_genus[g_idx]), "n_frames": n}

        f_idx = int(mean_family.argmax())
        if mean_family[f_idx] >= family_thresh:
            return {"level": "family", "index": f_idx,
                    "confidence": float(mean_family[f_idx]), "n_frames": n}

        return {"level": "unidentified", "index": None,
                "confidence": float(mean_species[sp_idx]), "n_frames": n}

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "track_id": self.track_id,
            "length": self.length,
            "frame_ids": self.frame_ids,
            "bboxes": self.bboxes.tolist(),
            "embeddings": self.embeddings.tolist(),
            # Per-frame logits; None where the classifier did not run.
            "logits": [
                (f[3].tolist() if f[3] is not None else None)
                for f in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Tracklet":
        """Deserialize from a dictionary."""
        # 'logits' is absent in tracklets serialized before issue #5.
        logits_list = data.get("logits", [None] * len(data["frame_ids"]))
        frames = []
        for fid, bbox, emb, lg in zip(
            data["frame_ids"], data["bboxes"], data["embeddings"], logits_list
        ):
            frames.append((
                int(fid),
                np.array(bbox, dtype=np.float64),
                np.array(emb, dtype=np.float64),
                np.array(lg, dtype=np.float64) if lg is not None else None,
            ))
        return cls(track_id=data["track_id"], frames=frames)


class TrackletWriter:
    """Extracts and saves tracklets from completed tracks. Eligible = >=
    min_length frames with embeddings; tracks > max_length are split into
    overlapping windows."""

    def __init__(
        self,
        min_length: int = 16,
        max_length: int = 30,
        overlap: int = 8,
        output_dir: str = "outputs/tracklets",
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.overlap = overlap
        self.output_dir = output_dir

    def extract_tracklets(
        self, tracks: list
    ) -> List[Tracklet]:
        """Tracks -> tracklets: discard < min_length, split > max_length into
        overlapping windows."""
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
                for frame_id, bbox, embedding, fr_logits in history:
                    tracklet.frames.append((
                        frame_id, bbox.copy(), embedding.copy(),
                        fr_logits.copy() if fr_logits is not None else None,
                    ))
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
                    for frame_id, bbox, embedding, fr_logits in window:
                        tracklet.frames.append((
                            frame_id, bbox.copy(), embedding.copy(),
                            fr_logits.copy() if fr_logits is not None else None,
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
        Save tracklets to a compressed .npz. Keys: track_ids (K,), and object
        arrays frame_ids/bboxes/embeddings/logits (one entry per tracklet;
        logits is empty for tracklets with no per-frame priors, #5).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or "tracklets.npz"
        filepath = os.path.join(self.output_dir, filename)

        track_ids = np.array([t.track_id for t in tracklets])
        frame_ids_list = [np.array(t.frame_ids) for t in tracklets]
        bboxes_list = [t.bboxes for t in tracklets]
        embeddings_list = [t.embeddings for t in tracklets]
        # Per-tracklet logits; an empty array where no frame carried logits.
        logits_list = [
            (t.logits if t.logits is not None else np.empty((0, 0)))
            for t in tracklets
        ]

        np.savez_compressed(
            filepath,
            track_ids=track_ids,
            frame_ids=np.array(frame_ids_list, dtype=object),
            bboxes=np.array(bboxes_list, dtype=object),
            embeddings=np.array(embeddings_list, dtype=object),
            logits=np.array(logits_list, dtype=object),
        )

        logger.info(f"Saved {len(tracklets)} tracklets to: {filepath}")
        return filepath

    def save_json(
        self,
        tracklets: List[Tracklet],
        filename: Optional[str] = None,
    ) -> str:
        """Save tracklets to a human-readable JSON (for debugging)."""
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
