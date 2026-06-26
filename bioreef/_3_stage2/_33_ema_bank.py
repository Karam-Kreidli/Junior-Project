"""
EMA feature bank for visual re-identification (BoT-SORT).

Per-track exponential moving average of Stage-1 embeddings — a pose-stable
"fingerprint" the tracker compares (cosine) to resolve ambiguous matches.

    eₜ = α·eₜ₋₁ + (1−α)·zₜ      (α = history weight; higher = slower to adapt)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("bioreef._3_stage2.ema")


class EMABank:
    """EMA-smoothed appearance embedding for a single track."""

    def __init__(self, alpha: float = 0.9, embedding_dim: int = 256):
        self.alpha = alpha  # 0.9 = 90% history, 10% new observation
        self.embedding_dim = embedding_dim
        self._embedding: Optional[np.ndarray] = None

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return self._embedding

    def initialize(self, embedding: np.ndarray) -> None:
        self._embedding = np.asarray(embedding, dtype=np.float64).copy()

    def update(self, new_embedding: np.ndarray) -> None:
        z = np.asarray(new_embedding, dtype=np.float64)
        if self._embedding is None:
            self._embedding = z.copy()
        else:
            self._embedding = self.alpha * self._embedding + (1 - self.alpha) * z

    def cosine_similarity(self, query: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]; -1.0 if the bank is empty."""
        if self._embedding is None:
            return -1.0

        q = np.asarray(query, dtype=np.float64)
        norm_q = np.linalg.norm(q)
        norm_e = np.linalg.norm(self._embedding)

        if norm_q < 1e-8 or norm_e < 1e-8:  # guard against zero vectors
            return 0.0

        return float(np.dot(q, self._embedding) / (norm_q * norm_e))


def cosine_distance_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """Pairwise cosine distance (1 - similarity), shape (M, N). Lower = closer."""
    norms_a = np.maximum(np.linalg.norm(embeddings_a, axis=1, keepdims=True), 1e-8)
    norms_b = np.maximum(np.linalg.norm(embeddings_b, axis=1, keepdims=True), 1e-8)
    a_normed = embeddings_a / norms_a
    b_normed = embeddings_b / norms_b
    return 1.0 - a_normed @ b_normed.T
