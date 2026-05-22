"""
BioReef.ai — BoTSORT Tracker
==============================
Full BoT-SORT implementation: ByteTrack dual-threshold association extended
with Camera Motion Compensation (CMC) and EMA-based visual Re-ID.

Cascaded Matching Hierarchy (per frame):
    1. CMC-Corrected Motion Gate
       Before any appearance comparison, the CMC-corrected Kalman prediction
       establishes a physical boundary. Detections outside the predicted
       search region are rejected — fish cannot teleport.

    2. Primary Match (high-confidence detections)
       High-confidence detections (≥ high_thresh) are matched to active
       tracks by IoU. A DINOv2 cosine similarity veto prevents mismatches
       between spatially overlapping but visually different individuals.

    3. Low-Confidence Rescue (ByteTrack second pass)
       Remaining unmatched tracks attempt matching against low-confidence
       detections (low_thresh ≤ score < high_thresh) using IoU only.
       This "rescues" blurry or partially occluded fish, preserving their
       track ID instead of creating a new one.

    4. EMA Appearance Rescue (lost track recovery)
       For tracks that have been lost for multiple frames, the system
       falls back entirely on EMA embedding comparisons against a gallery
       of lost tracks, re-identifying individuals by unique scale and fin
       texture signatures.

The Hungarian Algorithm solves each cost matrix to find the globally
optimal assignment of detections to tracks.

Reference:
    Aharon et al. (2022), "BoT-SORT: Robust Associations Multi-Pedestrian
    Tracking."
    Zhang et al. (2022), "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box."
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ema_bank import EMABank, cosine_distance_matrix
from .kalman_filter import CMC, KalmanFilter
from .track import Track, TrackState

logger = logging.getLogger("bioreef.tracking.botsort")

# Mahalanobis gating threshold (chi-squared 95% for 4-DOF)
_GATING_THRESHOLD = 9.4877


def iou_batch(
    bboxes_a: np.ndarray, bboxes_b: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of [x, y, w, h] bounding boxes.

    Args:
        bboxes_a: (M, 4) array.
        bboxes_b: (N, 4) array.

    Returns:
        (M, N) IoU matrix.
    """
    M = len(bboxes_a)
    N = len(bboxes_b)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    # Convert [x, y, w, h] → [x1, y1, x2, y2]
    a = bboxes_a.copy()
    a[:, 2] += a[:, 0]
    a[:, 3] += a[:, 1]

    b = bboxes_b.copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]

    # Intersection
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = bboxes_a[:, 2] * bboxes_a[:, 3]
    area_b = bboxes_b[:, 2] * bboxes_b[:, 3]

    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def _hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the assignment problem via the Hungarian Algorithm and filter
    by a cost threshold.

    Args:
        cost_matrix: (M, N) cost matrix (lower = better match).
        threshold:   Maximum acceptable cost for a valid match.

    Returns:
        matches:        List of (row_idx, col_idx) matched pairs.
        unmatched_rows: Row indices with no valid match.
        unmatched_cols: Column indices with no valid match.
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class BoTSORTTracker:
    """
    BoT-SORT multi-object tracker for underwater fish tracking.

    Combines:
        - ByteTrack dual-threshold association
        - Kalman Filter motion prediction
        - Camera Motion Compensation (CMC)
        - EMA-smoothed DINOv2 embeddings for Re-ID
        - Hungarian Algorithm for optimal global assignment
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        max_lost_age: int = 30,
        min_hits_to_confirm: int = 3,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.4,
        ema_alpha: float = 0.9,
        embedding_dim: Optional[int] = None,
        lambda_iou: float = 0.7,
        enable_cmc: bool = True,
    ):
        """
        Args:
            high_thresh:          Confidence threshold for first-pass matching.
            low_thresh:           Confidence threshold for second-pass rescue.
            max_lost_age:         Max frames a track can be LOST before DEAD.
            min_hits_to_confirm:  Minimum consecutive hits before a track is
                                  considered confirmed (reduces false tracks).
            iou_threshold:        IoU cost threshold for valid matches.
            appearance_threshold: Cosine distance threshold for Re-ID matching.

                                  NOTE (issue #1): the Re-ID descriptor is the
                                  raw DINOv3 ROI [CLS] token (768-D), NOT the
                                  MCEAM-fused embedding (which collapses
                                  same-species individuals). The 768-D cosine
                                  distance distribution differs from the old
                                  256-D one, AND Khorfakkan's underwater
                                  statistics differ from DINOv3's natural-image
                                  pretraining. This threshold (0.4) MUST be
                                  re-tuned empirically on Khorfakkan footage:
                                  measure typical inter-frame vs inter-track
                                  cosine distances on a short clip and set this
                                  near the midpoint.
            ema_alpha:            EMA smoothing factor for appearance bank.
            embedding_dim:        Re-ID embedding dimension. If None, inferred
                                  lazily from the first embedding seen (robust
                                  to either 768-D DINOv3 [CLS] or legacy 256-D).
            lambda_iou:           Weight for IoU vs appearance in combined cost.
                                  cost = λ·IoU_cost + (1-λ)·appearance_cost.
                                  Default 0.7 (was 0.98): with the meaningful
                                  DINOv3 Re-ID embedding, appearance should
                                  actually contribute rather than act only as a
                                  veto. Validate/tune on real tracking data.
            enable_cmc:           Whether to use Camera Motion Compensation.
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost_age = max_lost_age
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.ema_alpha = ema_alpha
        # Resolved lazily from the first Re-ID embedding seen (see update()).
        self.embedding_dim = embedding_dim
        self.lambda_iou = lambda_iou

        # Core components
        self.kf = KalmanFilter()
        self.cmc = CMC() if enable_cmc else None

        # Track pools
        self.active_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.dead_tracks: List[Track] = []

        # Per-track EMA banks, keyed by track_id
        self._ema_banks: Dict[int, EMABank] = {}

        # Frame counter
        self._frame_count = 0

    def _init_track(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray],
        reid_embedding: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> Track:
        """
        Create a new Track with Kalman and EMA initialization.

        `embedding` is the MCEAM-fused vector — it flows into the Track's
        frame_history and becomes the Stage 3 tracklet (habitat-aware
        z_context). `reid_embedding` is the raw DINOv3 [CLS] token used
        ONLY for the EMA appearance bank / association (issue #1). Keeping
        them separate prevents the Re-ID swap from corrupting Stage 3 data.
        `logits` is the per-frame species prior, carried for the
        hierarchical-fallback aggregation (issue #5).
        """
        track = Track(
            bbox=bbox,
            confidence=confidence,
            embedding=embedding,
            frame_id=self._frame_count,
            logits=logits,
        )

        # Initialize Kalman state
        state, cov = self.kf.initiate(bbox)
        track.kf_state = state
        track.kf_covariance = cov

        # Initialize EMA bank with the Re-ID embedding (not the fused one)
        bank = EMABank(alpha=self.ema_alpha, embedding_dim=self.embedding_dim)
        if reid_embedding is not None:
            bank.initialize(reid_embedding)
        self._ema_banks[track.track_id] = bank

        return track

    def _predict_tracks(
        self,
        tracks: List[Track],
        warp: Optional[np.ndarray],
    ) -> None:
        """Run Kalman predict (with optional CMC warp) on all tracks."""
        for track in tracks:
            if track.kf_state is None:
                continue

            # Step 1: CMC — compensate for camera motion
            if warp is not None:
                track.kf_state = self.cmc.apply_warp_to_state(
                    track.kf_state, warp
                )

            # Step 2: Kalman predict
            track.kf_state, track.kf_covariance = self.kf.predict(
                track.kf_state, track.kf_covariance
            )

            # Update bbox from predicted state
            track.bbox = self.kf.state_to_bbox(track.kf_state)
            track.predict_step()

    def _get_track_bboxes(self, tracks: List[Track]) -> np.ndarray:
        """Extract bboxes from a list of tracks as (N, 4) array."""
        if not tracks:
            return np.empty((0, 4), dtype=np.float64)
        return np.array([t.bbox for t in tracks], dtype=np.float64)

    def _get_track_embeddings(self, tracks: List[Track]) -> np.ndarray:
        """Extract EMA Re-ID embeddings from a list of tracks as (N, D) array."""
        dim = self.embedding_dim or 256  # fallback before lazy resolution
        if not tracks:
            return np.empty((0, dim), dtype=np.float64)

        embeddings = []
        for t in tracks:
            bank = self._ema_banks.get(t.track_id)
            if bank is not None and bank.embedding is not None:
                embeddings.append(bank.embedding)
            else:
                embeddings.append(np.zeros(dim))
        return np.array(embeddings, dtype=np.float64)

    def _update_track(
        self,
        track: Track,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray],
        reid_embedding: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update a matched track with Kalman correction and EMA update.

        `embedding` (MCEAM-fused) flows into the Track's frame_history for
        Stage 3. `reid_embedding` (DINOv3 [CLS]) updates the EMA appearance
        bank used for association only (issue #1). `logits` is the per-frame
        species prior, stored for hierarchical-fallback aggregation (#5).
        """
        # Kalman update
        if track.kf_state is not None:
            track.kf_state, track.kf_covariance = self.kf.update(
                track.kf_state, track.kf_covariance, bbox
            )

        # Track state update (fused embedding + logits → Stage 3 tracklet)
        track.update(bbox, confidence, embedding, self._frame_count, logits)

        # EMA update (Re-ID embedding → association bank)
        if reid_embedding is not None:
            bank = self._ema_banks.get(track.track_id)
            if bank is not None:
                bank.update(reid_embedding)

    def update(
        self,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        reid_embeddings: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Process one frame of detections through the tracking cascade.

        Args:
            bboxes:      (N, 4) array of detections [x, y, w, h].
            confidences: (N,) array of confidence scores.
            embeddings:  (N, 256) MCEAM-fused embeddings, or None. These flow
                         into Track.frame_history and become the Stage 3
                         tracklet (habitat-aware z_context). NOT used for
                         association.
            frame:       Raw video frame for CMC computation (BGR).
            reid_embeddings: (N, 768) raw DINOv3 ROI [CLS] tokens, or None.
                         Used exclusively for the EMA appearance bank and
                         association cost (issue #1). If None, falls back to
                         `embeddings` for association (legacy behavior — old
                         callers keep working, just with the suboptimal
                         MCEAM-as-Re-ID descriptor).
            logits:      (N, C) per-detection species classifier logits, or
                         None. Stored in Track.frame_history for the
                         hierarchical-fallback aggregation (issue #5). Not
                         used for association.

        Returns:
            List of all confirmed active tracks (with updated bboxes).
        """
        self._frame_count += 1

        bboxes = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
        confidences = np.asarray(confidences, dtype=np.float64).flatten()
        N = len(bboxes)

        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float64)

        if logits is not None:
            logits = np.asarray(logits, dtype=np.float64)

        # Re-ID descriptor: prefer the dedicated DINOv3 [CLS] embeddings;
        # fall back to the fused embeddings for backward compatibility.
        if reid_embeddings is not None:
            reid_embeddings = np.asarray(reid_embeddings, dtype=np.float64)
        else:
            reid_embeddings = embeddings

        # Lazily resolve the Re-ID embedding dimension from the first
        # non-empty array we see (robust to 768-D DINOv3 or legacy 256-D).
        if self.embedding_dim is None and reid_embeddings is not None \
                and len(reid_embeddings) > 0:
            self.embedding_dim = int(reid_embeddings.shape[1])

        # =====================================================================
        # Step 0: CMC — estimate camera motion
        # =====================================================================
        warp = None
        if self.cmc is not None and frame is not None:
            warp = self.cmc.compute_warp(frame)

        # =====================================================================
        # Step 1: Kalman predict for all active + lost tracks
        # =====================================================================
        self._predict_tracks(self.active_tracks, warp)
        self._predict_tracks(self.lost_tracks, warp)

        # =====================================================================
        # Step 2: Split detections by confidence
        # =====================================================================
        high_mask = confidences >= self.high_thresh
        low_mask = (confidences >= self.low_thresh) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_bboxes = bboxes[high_indices]
        high_confs = confidences[high_indices]
        # Fused embeddings → Stage 3 tracklet; reid embeddings → association.
        high_embeds = embeddings[high_indices] if embeddings is not None else None
        high_reid = (
            reid_embeddings[high_indices] if reid_embeddings is not None else None
        )
        # Per-frame species logits → Stage 3 tracklet (issue #5).
        high_logits = logits[high_indices] if logits is not None else None

        low_bboxes = bboxes[low_indices]
        low_confs = confidences[low_indices]

        # =====================================================================
        # Step 3: Primary match — high-confidence detections vs active tracks
        # =====================================================================
        matched_track_indices = []
        matched_det_indices = []
        unmatched_tracks_1st = list(range(len(self.active_tracks)))
        unmatched_dets_1st = list(range(len(high_indices)))

        if len(self.active_tracks) > 0 and len(high_indices) > 0:
            track_bboxes = self._get_track_bboxes(self.active_tracks)

            # IoU cost
            iou_matrix = iou_batch(track_bboxes, high_bboxes)
            iou_cost = 1.0 - iou_matrix

            # Appearance cost (if Re-ID embeddings available)
            if high_reid is not None:
                track_embeds = self._get_track_embeddings(self.active_tracks)
                app_cost = cosine_distance_matrix(track_embeds, high_reid)

                # Combined cost: λ·IoU + (1-λ)·appearance
                cost = (
                    self.lambda_iou * iou_cost
                    + (1 - self.lambda_iou) * app_cost
                )

                # Apply motion gate via Mahalanobis distance
                for i, track in enumerate(self.active_tracks):
                    if track.kf_state is not None:
                        for j in range(len(high_bboxes)):
                            gate_dist = self.kf.gating_distance(
                                track.kf_state,
                                track.kf_covariance,
                                high_bboxes[j],
                            )
                            if gate_dist > _GATING_THRESHOLD:
                                cost[i, j] = 1e5  # Block this match

                # Appearance veto: block if cosine distance too high
                for i in range(len(self.active_tracks)):
                    for j in range(len(high_bboxes)):
                        if app_cost[i, j] > self.appearance_threshold:
                            cost[i, j] = 1e5
            else:
                cost = iou_cost

            matches, unmatched_tracks_1st, unmatched_dets_1st = (
                _hungarian_match(cost, 1.0 - self.iou_threshold)
            )

            for t_idx, d_idx in matches:
                matched_track_indices.append(t_idx)
                matched_det_indices.append(d_idx)

                det_embed = high_embeds[d_idx] if high_embeds is not None else None
                det_reid = high_reid[d_idx] if high_reid is not None else None
                det_logits = high_logits[d_idx] if high_logits is not None else None
                self._update_track(
                    self.active_tracks[t_idx],
                    high_bboxes[d_idx],
                    high_confs[d_idx],
                    det_embed,
                    det_reid,
                    det_logits,
                )

        # =====================================================================
        # Step 4: Low-confidence rescue — unmatched tracks vs low-conf dets
        # =====================================================================
        remaining_tracks = [self.active_tracks[i] for i in unmatched_tracks_1st]
        matched_in_2nd = set()

        if len(remaining_tracks) > 0 and len(low_indices) > 0:
            track_bboxes = self._get_track_bboxes(remaining_tracks)
            iou_matrix = iou_batch(track_bboxes, low_bboxes)
            iou_cost = 1.0 - iou_matrix

            matches_2nd, unmatched_t2, _ = _hungarian_match(
                iou_cost, 1.0 - self.iou_threshold
            )

            for t_idx, d_idx in matches_2nd:
                real_t_idx = unmatched_tracks_1st[t_idx]
                self._update_track(
                    self.active_tracks[real_t_idx],
                    low_bboxes[d_idx],
                    low_confs[d_idx],
                    None,  # No embedding for low-conf detections
                )
                matched_in_2nd.add(real_t_idx)

        # =====================================================================
        # Step 5: EMA Appearance Rescue — unmatched high-conf dets vs lost tracks
        # =====================================================================
        # unmatched_dets_1st contains indices into high_bboxes/high_embeds
        remaining_det_indices = list(unmatched_dets_1st)

        if (
            len(self.lost_tracks) > 0
            and len(remaining_det_indices) > 0
            and high_reid is not None
        ):
            remaining_reid = high_reid[remaining_det_indices]
            lost_embeds = self._get_track_embeddings(self.lost_tracks)

            app_cost = cosine_distance_matrix(lost_embeds, remaining_reid)

            matches_3rd, _, unmatched_d3 = _hungarian_match(
                app_cost, self.appearance_threshold
            )

            recovered = set()
            for t_idx, d_idx in matches_3rd:
                real_d_idx = remaining_det_indices[d_idx]
                track = self.lost_tracks[t_idx]

                self._update_track(
                    track,
                    high_bboxes[real_d_idx],
                    high_confs[real_d_idx],
                    high_embeds[real_d_idx] if high_embeds is not None else None,
                    high_reid[real_d_idx],
                    high_logits[real_d_idx] if high_logits is not None else None,
                )
                recovered.add(t_idx)

            # Move recovered tracks to active pool
            recovered_tracks = [
                t for i, t in enumerate(self.lost_tracks)
                if i in recovered
            ]
            self.lost_tracks = [
                t for i, t in enumerate(self.lost_tracks)
                if i not in recovered
            ]

            # Update remaining unmatched detection indices
            remaining_det_indices = [
                remaining_det_indices[i] for i in unmatched_d3
            ]
        else:
            recovered_tracks = []

        # =====================================================================
        # Step 6: Handle unmatched original tracks — mark lost
        # =====================================================================
        # Build set of matched original track indices
        matched_original = set(matched_track_indices) | matched_in_2nd
        n_original = len(self.active_tracks)

        still_active = []
        for i in range(n_original):
            track = self.active_tracks[i]
            if i in matched_original:
                still_active.append(track)
            else:
                track.mark_lost()
                self.lost_tracks.append(track)

        # Add recovered tracks (from step 5)
        still_active.extend(recovered_tracks)

        # =====================================================================
        # Step 7: Create new tracks from unmatched high-confidence detections
        # =====================================================================
        for d_idx in remaining_det_indices:
            det_embed = high_embeds[d_idx] if high_embeds is not None else None
            det_reid = high_reid[d_idx] if high_reid is not None else None
            det_logits = high_logits[d_idx] if high_logits is not None else None
            new_track = self._init_track(
                high_bboxes[d_idx], high_confs[d_idx], det_embed, det_reid,
                det_logits,
            )
            still_active.append(new_track)

        self.active_tracks = still_active

        # Retire old lost tracks
        still_lost = []
        for track in self.lost_tracks:
            if track.time_since_update > self.max_lost_age:
                track.mark_dead()
                self.dead_tracks.append(track)
            else:
                track.time_since_update += 1
                still_lost.append(track)

        self.lost_tracks = still_lost

        # =====================================================================
        # Step 8: Return confirmed tracks
        # =====================================================================
        confirmed = [
            t for t in self.active_tracks
            if t.hits >= self.min_hits_to_confirm
        ]

        return confirmed

    def get_all_tracks(self) -> List[Track]:
        """Return all tracks (active + lost + dead) for tracklet export."""
        return self.active_tracks + self.lost_tracks + self.dead_tracks

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.dead_tracks.clear()
        self._ema_banks.clear()
        self._frame_count = 0
        Track.reset_id_counter()
        if self.cmc is not None:
            self.cmc.reset()
