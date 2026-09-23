"""
BoT-SORT tracker — ByteTrack dual-threshold association + CMC + EMA Re-ID.

Per-frame cascade (Hungarian-solved at each stage):
    0/1. CMC-corrected Kalman predict + Mahalanobis motion gate (no teleports).
    2.   Primary: high-conf dets vs active tracks by IoU, with an appearance veto.
    3.   Rescue: unmatched tracks vs low-conf dets by IoU (saves blurry/occluded).
    4.   EMA rescue: unmatched high-conf dets vs lost tracks by appearance
         (motion-gated — see Step 5 in update()).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._33_ema_bank import EMABank, cosine_distance_matrix
from ._31_kalman_filter import CMC, KalmanFilter
from ._32_track import Track, TrackState

logger = logging.getLogger("bioreef._3_stage2.botsort")

# Mahalanobis gating threshold (chi-squared 95% for 4-DOF)
_GATING_THRESHOLD = 9.4877


def iou_batch(
    bboxes_a: np.ndarray, bboxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise IoU between two sets of [x, y, w, h] boxes -> (M, N)."""
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


def diou_batch(
    bboxes_a: np.ndarray, bboxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise Distance-IoU between two sets of [x, y, w, h] boxes -> (M, N).

    DIoU = IoU - ρ²(centers) / c²  where ρ is the center distance and c is the
    diagonal of the smallest box enclosing both. Range (-1, 1].

    Why DIoU instead of plain IoU for association (#jitter): when a detector
    jitters a box by ~half its size on a stationary fish, the jumped box barely
    overlaps its predicted position, so plain IoU collapses to ~0 and the match
    is rejected — the track fragments and a new ID is born every frame. DIoU
    still rewards boxes whose *centers* are close even when they no longer
    overlap, so association degrades gracefully instead of falling off a cliff
    at IoU=0. Two boxes at the same center but different size score near +1;
    two boxes drifting apart go smoothly negative rather than snapping to 0.
    """
    M = len(bboxes_a)
    N = len(bboxes_b)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    iou = iou_batch(bboxes_a, bboxes_b)

    # Centers
    ca = bboxes_a[:, :2] + bboxes_a[:, 2:4] / 2.0   # (M, 2)
    cb = bboxes_b[:, :2] + bboxes_b[:, 2:4] / 2.0   # (N, 2)
    dx = ca[:, 0:1] - cb[:, 0:1].T
    dy = ca[:, 1:2] - cb[:, 1:2].T
    center_dist_sq = dx * dx + dy * dy               # (M, N)

    # Diagonal of the smallest enclosing box, squared
    a_x1 = bboxes_a[:, 0:1]; a_y1 = bboxes_a[:, 1:2]
    a_x2 = (bboxes_a[:, 0] + bboxes_a[:, 2])[:, None]
    a_y2 = (bboxes_a[:, 1] + bboxes_a[:, 3])[:, None]
    b_x1 = bboxes_b[:, 0:1].T; b_y1 = bboxes_b[:, 1:2].T
    b_x2 = (bboxes_b[:, 0] + bboxes_b[:, 2])[None, :]
    b_y2 = (bboxes_b[:, 1] + bboxes_b[:, 3])[None, :]

    enc_x1 = np.minimum(a_x1, b_x1); enc_y1 = np.minimum(a_y1, b_y1)
    enc_x2 = np.maximum(a_x2, b_x2); enc_y2 = np.maximum(a_y2, b_y2)
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    enc_diag_sq = np.maximum(enc_diag_sq, 1e-9)      # guard div-by-zero

    return iou - center_dist_sq / enc_diag_sq


def _hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment filtered by a max cost -> (matches,
    unmatched_rows, unmatched_cols)."""
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
    """BoT-SORT tracker: ByteTrack dual-threshold association + Kalman + CMC +
    EMA DINOv3 Re-ID, with Hungarian global assignment."""

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        max_lost_age: int = 30,
        min_hits_to_confirm: int = 3,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.4,
        rescue_appearance_threshold: Optional[float] = None,
        min_iou_for_match: float = -0.5,
        ema_alpha: float = 0.9,
        embedding_dim: Optional[int] = None,
        lambda_iou: float = 0.7,
        enable_cmc: bool = True,
        use_diou: bool = True,
        motion_weight: float = 0.0,
        size_weight: float = 0.0,
        proximity_iou: float = 0.15,
        grace_period: int = 0,
        grace_gate_scale: float = 4.0,
        kf_r_weight: float = 1.0 / 20,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost_age = max_lost_age
        self.min_hits_to_confirm = min_hits_to_confirm
        self.iou_threshold = iou_threshold
        # Cosine-distance VETO for the Step-3 primary match: block an
        # active-track↔detection pair when their Re-ID cosine distance exceeds
        # this. Re-ID descriptor is the raw DINOv3 ROI [CLS] (768-D), not the
        # MCEAM-fused vector (#1). 0.4 MUST be re-tuned on Khorfakkan footage.
        self.appearance_threshold = appearance_threshold
        # Cosine-distance GATE for the Step-5 lost-track rescue: accept a
        # rescue only when distance <= this. Split from appearance_threshold
        # (#T7) because the veto ("too different to keep a match") and the
        # rescue gate ("similar enough to resurrect a dead track") want
        # different values — a strict rescue gate avoids resurrecting the wrong
        # identity, while the veto can be looser. Defaults to appearance_threshold
        # to preserve prior behavior until tuned.
        self.rescue_appearance_threshold = (
            rescue_appearance_threshold
            if rescue_appearance_threshold is not None
            else appearance_threshold
        )
        # Minimum IoU a primary (Step-3) match must have, regardless of how
        # good the appearance term looks (#T2). Without this floor, two
        # appearance-identical conspecifics (app_cost≈0) match on almost any
        # spatial overlap, because the combined cost λ·(1-IoU)+(1-λ)·0 clears
        # the Hungarian threshold at low IoU — the conspecific ID-swap path.
        # Minimum association score a primary (Step-3) match must have (#T2).
        # With DIoU (range (-1, 1]) a detector-jittered box can legitimately
        # score slightly negative, so this floor is negative: it still blocks
        # boxes that are genuinely elsewhere (DIoU very negative) while letting
        # a near-center jump through. With plain IoU set this to a small
        # positive value (~0.1) instead.
        self.min_iou_for_match = min_iou_for_match
        self.ema_alpha = ema_alpha
        # Resolved lazily from the first Re-ID embedding seen (768-D or 256-D).
        self.embedding_dim = embedding_dim
        # cost = λ·IoU + (1-λ)·appearance. 0.7 (was 0.98) lets the DINOv3 Re-ID
        # actually contribute rather than act only as a veto; tune on real data.
        self.lambda_iou = lambda_iou

        # --- Association metric & extra geometric cues -----------------------
        # DIoU instead of plain IoU for the spatial term (#jitter): survives
        # detector box-jitter that collapses IoU to 0 on a stationary fish.
        self.use_diou = use_diou
        # Velocity-direction consistency weight (#swap): penalizes a match whose
        # implied displacement disagrees with the track's Kalman velocity — the
        # signal that distinguishes two identical-looking conspecifics crossing
        # (one implied assignment requires a ~180° reversal). Uses [u̇,v̇],
        # already in the Kalman state; 0 disables.
        self.motion_weight = motion_weight
        # Size/depth consistency weight (#swap): penalizes a match with a large
        # box-height ratio, a weak per-individual cue for conspecifics at
        # different distances from the camera. 0 disables.
        self.size_weight = size_weight
        # Adaptive appearance muting (#swap): when a track has ANOTHER track
        # within this DIoU proximity (a contested crossing), appearance is
        # near-useless for conspecifics and actively misleads — its weight is
        # driven to 0 for that track and the match leans on motion+geometry.
        self.proximity_iou = proximity_iou

        # --- New-track grace period (#jitter) --------------------------------
        # A brand-new track has zero velocity and tight covariance, so frame-2
        # detector jitter breaks it before the filter learns the fish's motion.
        # For a track's first `grace_period` frames, the Mahalanobis motion gate
        # is loosened by `grace_gate_scale` so early jitter can't fragment it.
        self.grace_period = grace_period
        self.grace_gate_scale = grace_gate_scale

        # Core components. Looser measurement noise (kf_r_weight) tells the
        # filter detections are jittery so it smooths rather than chases them.
        self.kf = KalmanFilter(std_weight_measurement=kf_r_weight)
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
        """New Track with Kalman + EMA init. `embedding` (MCEAM-fused) -> Stage 3
        tracklet; `reid_embedding` (DINOv3 [CLS]) -> EMA bank only, kept separate
        so Re-ID can't corrupt Stage 3 (#1); `logits` -> #5 aggregation."""
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

    def _gate_matrix(
        self, tracks: List[Track], det_bboxes: np.ndarray,
    ) -> np.ndarray:
        """Vectorized Mahalanobis motion gate -> boolean (T, N) mask, True where
        a track↔detection pair is spatially IMPLAUSIBLE and must be blocked.

        Replaces the previous triple-nested Python loop (T×N gating_distance
        calls) — a real per-frame latency cost on crowded frames for the demo.
        New/tentative tracks (hits <= grace_period) get the gate loosened by
        grace_gate_scale so early detector jitter can't fragment them (#jitter).
        """
        T, N = len(tracks), len(det_bboxes)
        blocked = np.zeros((T, N), dtype=bool)
        if T == 0 or N == 0:
            return blocked
        for i, track in enumerate(tracks):
            if track.kf_state is None:
                continue
            # Per-track gate threshold: looser during the grace period.
            thr = _GATING_THRESHOLD
            if track.hits <= self.grace_period:
                thr = thr * self.grace_gate_scale
            for j in range(N):
                d = self.kf.gating_distance(
                    track.kf_state, track.kf_covariance, det_bboxes[j],
                )
                if d > thr:
                    blocked[i, j] = True
        return blocked

    def _motion_cost(
        self, tracks: List[Track], det_bboxes: np.ndarray,
    ) -> np.ndarray:
        """Velocity-direction inconsistency cost (T, N) in [0, 1] (#swap).

        For each pair, compares the displacement the detection implies (from the
        track's last center to the detection center) against the track's Kalman
        velocity heading. 0 = same heading, 1 = full reversal. This is the cue
        that survives identical appearance: when two conspecifics cross, the
        swapped assignment implies a ~180° reversal for one of them -> cost ~1.
        Tracks with negligible speed contribute ~0.5 (uninformative), so a
        near-stationary lone fish isn't penalized.
        """
        T, N = len(tracks), len(det_bboxes)
        cost = np.full((T, N), 0.5, dtype=np.float64)
        if T == 0 or N == 0:
            return cost
        det_centers = det_bboxes[:, :2] + det_bboxes[:, 2:4] / 2.0  # (N,2)
        for i, track in enumerate(tracks):
            if track.kf_state is None:
                continue
            vx, vy = track.kf_state[4], track.kf_state[5]
            speed = np.hypot(vx, vy)
            if speed < 1.0:            # too slow to have a reliable heading
                continue
            tc = track.kf_state[:2]    # predicted center (u, v)
            disp = det_centers - tc    # (N, 2)
            disp_norm = np.linalg.norm(disp, axis=1)
            valid = disp_norm > 1e-6
            cos_sim = np.zeros(N)
            cos_sim[valid] = (
                (disp[valid, 0] * vx + disp[valid, 1] * vy)
                / (disp_norm[valid] * speed)
            )
            cost[i] = (1.0 - cos_sim) / 2.0
        return cost

    def _appearance_weights(self, track_bboxes: np.ndarray) -> np.ndarray:
        """Per-track appearance weight (T, 1) in {0, 1} for adaptive muting
        (#swap). A track with ANOTHER track within proximity_iou (DIoU) is in a
        contested crossing where conspecific appearance is unreliable -> weight
        0 (mute appearance, lean on motion/geometry). Isolated tracks keep
        weight 1 so appearance still helps where it discriminates."""
        T = len(track_bboxes)
        w = np.ones((T, 1), dtype=np.float64)
        if T < 2:
            return w
        pair = diou_batch(track_bboxes, track_bboxes)  # (T,T)
        np.fill_diagonal(pair, -np.inf)                # ignore self
        contested = (pair >= self.proximity_iou).any(axis=1)
        w[contested, 0] = 0.0
        return w

    def _size_cost(
        self, track_bboxes: np.ndarray, det_bboxes: np.ndarray,
    ) -> np.ndarray:
        """Box-height ratio inconsistency cost (T, N), ~[0, 1] (#swap). A weak
        per-individual cue: conspecifics at different camera distances differ in
        apparent height, so a mismatched-size assignment is penalized."""
        T, N = len(track_bboxes), len(det_bboxes)
        if T == 0 or N == 0:
            return np.zeros((T, N), dtype=np.float64)
        h_t = np.maximum(track_bboxes[:, 3:4], 1e-3)   # (T,1)
        h_d = np.maximum(det_bboxes[:, 3:4].T, 1e-3)   # (1,N)
        log_ratio = np.abs(np.log(h_d / h_t))
        return np.clip(log_ratio, 0.0, 1.0)

    def _update_track(
        self,
        track: Track,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray],
        reid_embedding: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        """Update a matched track: Kalman correction + EMA update. Embedding
        roles as in _init_track (#1); logits stored for #5."""
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
        Process one frame's detections through the cascade -> confirmed tracks.

        bboxes (N,4) and confidences (N,) are required. embeddings (MCEAM-fused)
        feed Stage 3; reid_embeddings (DINOv3 [CLS]) drive association, falling
        back to embeddings if None (#1); logits feed #5. None of embeddings/
        logits affect association. frame (BGR) enables CMC.
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
        unmatched_tracks_1st = list(range(len(self.active_tracks)))
        unmatched_dets_1st = list(range(len(high_indices)))

        if len(self.active_tracks) > 0 and len(high_indices) > 0:
            track_bboxes = self._get_track_bboxes(self.active_tracks)

            # --- Spatial term: DIoU (survives detector jitter, #jitter) ------
            if self.use_diou:
                spatial_score = diou_batch(track_bboxes, high_bboxes)  # (-1,1]
            else:
                spatial_score = iou_batch(track_bboxes, high_bboxes)   # [0,1]
            spatial_cost = 1.0 - spatial_score

            # --- Geometric cues that survive identical appearance (#swap) ----
            motion_cost = self._motion_cost(self.active_tracks, high_bboxes)
            size_cost = self._size_cost(track_bboxes, high_bboxes)

            # --- Appearance term, adaptively muted in contested crossings ----
            # A track with another track nearby (DIoU proximity) is in a
            # potential conspecific crossing where appearance misleads; its
            # appearance weight is driven to 0 so the match leans on motion.
            if high_reid is not None:
                track_embeds = self._get_track_embeddings(self.active_tracks)
                app_cost = cosine_distance_matrix(track_embeds, high_reid)
                app_weight = self._appearance_weights(track_bboxes)  # (T,1)
            else:
                app_cost = np.zeros_like(spatial_cost)
                app_weight = np.zeros((len(self.active_tracks), 1))

            # Combined cost. lambda_iou keeps the spatial term dominant; motion
            # and size are additive nudges; appearance is per-track weighted.
            cost = (
                self.lambda_iou * spatial_cost
                + self.motion_weight * motion_cost
                + self.size_weight * size_cost
                + app_weight * (1 - self.lambda_iou) * app_cost
            )

            # Vectorized Mahalanobis motion gate (grace-loosened for new tracks)
            blocked = self._gate_matrix(self.active_tracks, high_bboxes)
            cost[blocked] = 1e5

            # Appearance veto: block a pair that looks too different — but only
            # where appearance is actually trusted (app_weight > 0), so a muted
            # crossing pair isn't vetoed on unreliable appearance.
            if high_reid is not None:
                veto = (app_cost > self.appearance_threshold) & (app_weight > 0)
                cost[veto] = 1e5

            # Spatial floor (#T2): block pairs whose DIoU is below the floor,
            # even when other terms look cheap — stops appearance/motion from
            # dragging a spatially-implausible match through.
            below_floor = spatial_score < self.min_iou_for_match
            cost[below_floor] = 1e5

            # Accept threshold on the spatial term: a match must clear the
            # spatial gate (score >= iou_threshold => spatial_cost <= 1-thr).
            # Extra cost budget for the additive motion/size terms so a good
            # spatial match isn't rejected by a moderate heading penalty.
            accept = (1.0 - self.iou_threshold) \
                + self.motion_weight + self.size_weight
            matches, unmatched_tracks_1st, unmatched_dets_1st = (
                _hungarian_match(cost, accept)
            )

            for t_idx, d_idx in matches:
                matched_track_indices.append(t_idx)

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
            if self.use_diou:
                spatial_score = diou_batch(track_bboxes, low_bboxes)
            else:
                spatial_score = iou_batch(track_bboxes, low_bboxes)
            iou_cost = 1.0 - spatial_score

            # Motion gate the low-conf rescue (#T4), vectorized. Step 4 formerly
            # matched remaining tracks to low-confidence detections on raw IoU
            # alone — no gate — so a low-conf false positive near a track's
            # predicted box could silently hijack it. Gate as Steps 3 and 5 do.
            blocked = self._gate_matrix(remaining_tracks, low_bboxes)
            iou_cost[blocked] = 1e5

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
            remaining_bboxes = high_bboxes[remaining_det_indices]

            app_cost = cosine_distance_matrix(lost_embeds, remaining_reid)

            # Motion gate the appearance rescue (vectorized). Without this, a
            # LOST track is re-associated to ANY appearance-similar detection
            # regardless of position — so when a fish's detection drops for a
            # few frames, its LOST track gets hijacked by a *different* similar-
            # looking fish elsewhere in the frame (the 700px+ "teleport" ID
            # swaps). The Kalman gate blocks rescues that are spatially
            # implausible given where the lost track was last predicted to be.
            blocked = self._gate_matrix(self.lost_tracks, remaining_bboxes)
            app_cost[blocked] = 1e5  # spatially impossible — block

            matches_3rd, _, unmatched_d3 = _hungarian_match(
                app_cost, self.rescue_appearance_threshold
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
