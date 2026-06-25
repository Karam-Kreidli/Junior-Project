"""
Pipeline data contracts + caching.

In-memory result objects that stages pass to each other, each with save()/load()
so the runner can optionally cache a stage's output to disk and skip recomputing
it on a re-run (the "in-memory + optional cache" design). The on-disk format is
the SAME .npz the standalone scripts already write, so these are interchangeable
with the existing detections/tracklets archives.

    Frames        — preprocessing output (frame iterator + ids)
    Stage1Output  — detections .npz contract (bboxes/embeddings/reid/logits)
    Stage2Output  — tracklets .npz contract (+ optional verdicts)
    Stage3Output  — (future) refined verdicts
    cached(...)    — load-or-compute-and-save helper used by the runner
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np


# =============================================================================
# Frames — preprocessing output
# =============================================================================

@dataclass
class Frames:
    """A clip's frames ready for Stage 1.

    Either `paths` (extracted PNGs on disk) drives iteration, or a caller
    supplies frames another way. `frame_ids` are the source frame indices and
    MUST stay aligned with whatever a downstream consumer (e.g. CVAT) expects.
    """
    video_id: str
    frame_ids: List[int]
    paths: Optional[List[str]] = None        # parallel to frame_ids, or None

    def __len__(self) -> int:
        return len(self.frame_ids)

    def iter_frames(self) -> Iterator[Tuple[int, "np.ndarray"]]:
        """Yield (frame_id, bgr_image). Reads PNGs lazily from `paths`."""
        import cv2
        if self.paths is None:
            raise RuntimeError("Frames has no paths to iterate; supply paths "
                               "or override iteration.")
        for fid, p in zip(self.frame_ids, self.paths):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield fid, img


# =============================================================================
# Stage 1 — detections (mirrors the detections .npz exactly)
# =============================================================================

@dataclass
class Stage1Output:
    """Per-detection arrays — identical contract to the detections .npz that
    infer_stage1 writes today (frame_ids/bboxes/confidences/embeddings/
    reid_embeddings/logits/class_ids)."""
    video_id: str
    frame_ids: np.ndarray        # (N,)   int64
    bboxes: np.ndarray           # (N,4)  float  xywh
    confidences: np.ndarray      # (N,)   float
    embeddings: np.ndarray       # (N,256) float  MCEAM fused
    reid_embeddings: np.ndarray  # (N,D)  float   DINOv3 CLS (Stage-2 Re-ID, #1)
    logits: np.ndarray           # (N,C)  float16 species prior (#2)
    class_ids: np.ndarray        # (N,)   int64   (all 0 = fish)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            frame_ids=self.frame_ids,
            bboxes=self.bboxes,
            confidences=self.confidences,
            embeddings=self.embeddings,
            reid_embeddings=self.reid_embeddings,
            logits=self.logits,
            class_ids=self.class_ids,
        )
        return path

    @classmethod
    def load(cls, path: str, video_id: str = "") -> "Stage1Output":
        d = np.load(path, allow_pickle=True)
        reid = d["reid_embeddings"] if "reid_embeddings" in d else d["embeddings"]
        return cls(
            video_id=video_id,
            frame_ids=d["frame_ids"],
            bboxes=d["bboxes"],
            confidences=d["confidences"],
            embeddings=d["embeddings"],
            reid_embeddings=reid,
            logits=d["logits"] if "logits" in d else np.empty((0, 0)),
            class_ids=d["class_ids"] if "class_ids" in d
            else np.zeros(len(d["frame_ids"]), dtype=np.int64),
        )


# =============================================================================
# Stage 2 — tracklets (+ verdicts)
# =============================================================================

@dataclass
class Stage2Output:
    """Tracklets for a clip, plus optional hierarchical verdicts. Tracklets are
    the same objects TrackletWriter produces; save() defers to it so the on-disk
    format matches the existing tracklets .npz."""
    video_id: str
    tracklets: list                     # List[Tracklet]
    verdicts: Optional[List[dict]] = None

    def __len__(self) -> int:
        return len(self.tracklets)

    def save(self, path: str) -> str:
        from bioreef.tracking import TrackletWriter
        out_dir = os.path.dirname(path) or "."
        writer = TrackletWriter(output_dir=out_dir)
        writer.save(self.tracklets, filename=os.path.basename(path))
        if self.verdicts is not None:
            import json
            vpath = os.path.splitext(path)[0] + "_verdicts.json"
            with open(vpath, "w", encoding="utf-8") as f:
                json.dump(self.verdicts, f, indent=2)
        return path


# =============================================================================
# Stage 3 — (future)
# =============================================================================

@dataclass
class Stage3Output:
    video_id: str
    refined_verdicts: List[dict] = field(default_factory=list)


# =============================================================================
# Cache helper — the "in-memory + optional disk cache" core
# =============================================================================

def cache_path(cache_dir: str, video_id: str, stage: str) -> str:
    safe = video_id.replace(".avi", "").replace(".", "_")
    return os.path.join(cache_dir, f"{safe}.{stage}.npz")


def cached(cfg, stage: str, compute: Callable, loader: Optional[Callable] = None):
    """Load a stage result from cache if present, else compute and save it.

    cfg must expose: cache_dir (str|None), no_cache (bool), video_id (str).
    `compute()` -> a result object with .save(path).
    `loader(path)` -> reconstructs the result (defaults to the obvious one).
    If cfg.cache_dir is None, caching is disabled entirely (always compute).
    """
    cache_dir = getattr(cfg, "cache_dir", None)
    no_cache = getattr(cfg, "no_cache", False)
    video_id = getattr(cfg, "video_id", "") or ""

    if cache_dir and not no_cache:
        path = cache_path(cache_dir, video_id, stage)
        if os.path.exists(path) and loader is not None:
            return loader(path)

    result = compute()

    if cache_dir and result is not None and hasattr(result, "save"):
        os.makedirs(cache_dir, exist_ok=True)
        result.save(cache_path(cache_dir, video_id, stage))
    return result
