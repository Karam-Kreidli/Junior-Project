"""
BioReef.ai — EMA Feature Bank for Visual Re-Identification
============================================================
Maintains a per-track Exponential Moving Average (EMA) of DINOv2/MCEAM
embeddings to create stable "fingerprints" for individual fish.

    eₜ = α · eₜ₋₁ + (1 − α) · zₜ

    eₜ  : Updated EMA embedding (the track's smoothed identity)
    zₜ  : Current frame's Stage 1 embedding (raw observation)
    α   : Smoothing factor (higher = more memory, slower adaptation)

As a fish rotates from lateral to frontal view, its raw per-frame
embedding changes drastically. The EMA smooths these transitions,
producing a temporally stable identity signature resistant to pose
changes and light refraction artefacts.

When motion cues are ambiguous (two same-species fish crossing paths),
the tracker computes Cosine Similarity between a new detection's
embedding and each track's EMA bank to resolve identity.

Reference:
    Aharon et al. (2022), "BoT-SORT: Robust Associations Multi-Pedestrian
    Tracking."
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("bioreef._3_stage2.ema")


class EMABank:
    """
    Manages EMA-smoothed appearance embeddings for a single track.

    The bank stores one running average vector that is updated each
    time the track is matched to a detection with a valid embedding.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        embedding_dim: int = 256,
    ):
        """
        Args:
            alpha: EMA smoothing factor. 0.9 means 90% weight on history,
                   10% on the new observation. Higher values produce more
                   stable signatures but adapt slower to legitimate appearance
                   changes (e.g., fish turning).
            embedding_dim: Dimension of the Stage 1 MCEAM embedding.
        """
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self._embedding: Optional[np.ndarray] = None

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Current EMA embedding, or None if never initialized."""
        return self._embedding

    def initialize(self, embedding: np.ndarray) -> None:
        """
        Set the initial embedding (first detection of a new track).

        Args:
            embedding: First observed embedding vector.
        """
        self._embedding = np.asarray(embedding, dtype=np.float64).copy()

    def update(self, new_embedding: np.ndarray) -> None:
        """
        Update the EMA with a new matched detection's embedding.

            eₜ = α · eₜ₋₁ + (1 − α) · zₜ

        Args:
            new_embedding: Current frame's embedding from Stage 1.
        """
        z = np.asarray(new_embedding, dtype=np.float64)
        if self._embedding is None:
            self._embedding = z.copy()
        else:
            self._embedding = self.alpha * self._embedding + (1 - self.alpha) * z

    def cosine_similarity(self, query: np.ndarray) -> float:
        """
        Compute cosine similarity between a query embedding and the
        stored EMA embedding.

            sim(z_new, eᵢ) = (z_new · eᵢ) / (‖z_new‖ · ‖eᵢ‖)

        Args:
            query: Detection embedding to compare against this track.

        Returns:
            Cosine similarity in [-1, 1]. Returns -1.0 if the bank
            has no stored embedding.
        """
        if self._embedding is None:
            return -1.0

        q = np.asarray(query, dtype=np.float64)
        norm_q = np.linalg.norm(q)
        norm_e = np.linalg.norm(self._embedding)

        if norm_q < 1e-8 or norm_e < 1e-8:
            return 0.0

        return float(np.dot(q, self._embedding) / (norm_q * norm_e))


def cosine_distance_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix between two sets of embeddings.

    Distance = 1 - cosine_similarity, so lower = more similar.

    Args:
        embeddings_a: (M, D) array of embeddings.
        embeddings_b: (N, D) array of embeddings.

    Returns:
        (M, N) distance matrix where entry [i, j] is the cosine distance
        between embeddings_a[i] and embeddings_b[j].
    """
    # Normalize rows
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    norms_a = np.maximum(norms_a, 1e-8)
    norms_b = np.maximum(norms_b, 1e-8)

    a_normed = embeddings_a / norms_a
    b_normed = embeddings_b / norms_b

    # Cosine similarity matrix
    sim = a_normed @ b_normed.T

    # Convert to distance
    return 1.0 - sim
