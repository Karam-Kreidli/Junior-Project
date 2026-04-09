"""
BioReef.ai — Kalman Filter with Camera Motion Compensation
============================================================
Maintains the kinematic state of each tracked fish using a constant-velocity
Kalman Filter operating on the 8-dimensional state vector:

    x = [u, v, a, h, u̇, v̇, ȧ, ḣ]ᵀ

    u, v : Bounding box center coordinates
    a    : Aspect ratio (w / h)
    h    : Height of the bounding box
    u̇, v̇, ȧ, ḣ : Respective first-order velocities

Camera Motion Compensation (CMC):
    Before each Kalman predict step, the system estimates the global affine
    transform between consecutive frames (caused by camera vibration / surge)
    and warps the predicted state accordingly. Without this correction, the
    Kalman Filter would misinterpret background drift as fish movement.

Reference:
    Aharon et al. (2022), "BoT-SORT: Robust Associations Multi-Pedestrian
    Tracking."
    Bewley et al. (2016), "Simple Online and Realtime Tracking" (SORT).
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("bioreef.tracking.kalman")

# Measurement dimension: [u, v, a, h]
_MEAS_DIM = 4
# State dimension: [u, v, a, h, u̇, v̇, ȧ, ḣ]
_STATE_DIM = 8


class KalmanFilter:
    """
    Constant-velocity Kalman Filter for bounding box tracking.

    State vector: [u, v, a, h, u̇, v̇, ȧ, ḣ]ᵀ
    Measurement:  [u, v, a, h]

    The filter uses a standard linear prediction model with constant
    velocity assumption. Process and measurement noise are tuned for
    underwater fish movement (slower, more erratic than pedestrians).
    """

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ):
        """
        Args:
            std_weight_position: Relative weight for position uncertainty.
                                 Higher values = more uncertain predictions.
            std_weight_velocity: Relative weight for velocity uncertainty.
        """
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix F (constant velocity model)
        # x_{t+1} = F @ x_t
        self._F = np.eye(_STATE_DIM, dtype=np.float64)
        for i in range(_MEAS_DIM):
            self._F[i, _MEAS_DIM + i] = 1.0  # position += velocity * dt

        # Measurement matrix H: extracts [u, v, a, h] from state
        self._H = np.eye(_MEAS_DIM, _STATE_DIM, dtype=np.float64)

    def initiate(
        self, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an unmatched detection.

        Args:
            bbox: Bounding box [x, y, w, h].

        Returns:
            (state, covariance): Initial Kalman state and covariance matrix.
        """
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
        a = w / max(h, 1e-6)

        # State: [u, v, a, h, 0, 0, 0, 0]  (zero initial velocity)
        state = np.array(
            [cx, cy, a, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        # Initial covariance: high uncertainty for velocity components
        std = [
            2 * self._std_weight_position * h,   # u
            2 * self._std_weight_position * h,   # v
            1e-2,                                 # a (aspect ratio stable)
            2 * self._std_weight_position * h,   # h
            10 * self._std_weight_velocity * h,  # u̇
            10 * self._std_weight_velocity * h,  # v̇
            1e-5,                                 # ȧ
            10 * self._std_weight_velocity * h,  # ḣ
        ]
        covariance = np.diag(np.square(std))

        return state, covariance

    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Kalman predict step: advance state by one frame.

        Args:
            state:      Current 8-dim state vector.
            covariance: Current 8×8 covariance matrix.

        Returns:
            (predicted_state, predicted_covariance)
        """
        h = state[3]

        # Process noise Q (scales with object size)
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        Q = np.diag(np.square(std_pos + std_vel))

        # Predict: x' = F @ x, P' = F @ P @ F^T + Q
        state = self._F @ state
        covariance = self._F @ covariance @ self._F.T + Q

        return state, covariance

    def update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Kalman update step: correct state with a matched detection.

        Args:
            state:      Predicted 8-dim state vector.
            covariance: Predicted 8×8 covariance matrix.
            bbox:       Matched detection [x, y, w, h].

        Returns:
            (updated_state, updated_covariance)
        """
        # Convert bbox to measurement [u, v, a, h]
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
        a = w / max(h, 1e-6)
        measurement = np.array([cx, cy, a, h], dtype=np.float64)

        # Measurement noise R
        std = [
            self._std_weight_position * state[3],
            self._std_weight_position * state[3],
            1e-1,
            self._std_weight_position * state[3],
        ]
        R = np.diag(np.square(std))

        # Innovation
        S = self._H @ covariance @ self._H.T + R
        K = covariance @ self._H.T @ np.linalg.inv(S)
        y = measurement - self._H @ state

        # Update
        state = state + K @ y
        covariance = (np.eye(_STATE_DIM) - K @ self._H) @ covariance

        return state, covariance

    def state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """
        Convert Kalman state [u, v, a, h, ...] to bbox [x, y, w, h].

        Args:
            state: 8-dim Kalman state vector.

        Returns:
            Bounding box as [x, y, w, h].
        """
        u, v, a, h = state[:4]
        w = a * h
        x = u - w / 2.0
        y = v - h / 2.0
        return np.array([x, y, w, h], dtype=np.float64)

    def gating_distance(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        bbox: np.ndarray,
    ) -> float:
        """
        Compute the Mahalanobis distance between state and a detection.

        Used as the motion gate: detections beyond a threshold are rejected
        (fish cannot "teleport").

        Args:
            state:      Predicted 8-dim state.
            covariance: Predicted 8×8 covariance.
            bbox:       Detection [x, y, w, h].

        Returns:
            Squared Mahalanobis distance (lower = closer match).
        """
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
        a = w / max(h, 1e-6)
        measurement = np.array([cx, cy, a, h], dtype=np.float64)

        # Projected state and covariance in measurement space
        mean = self._H @ state
        S = self._H @ covariance @ self._H.T

        # Measurement noise
        std = [
            self._std_weight_position * state[3],
            self._std_weight_position * state[3],
            1e-1,
            self._std_weight_position * state[3],
        ]
        S += np.diag(np.square(std))

        # Mahalanobis distance
        diff = measurement - mean
        chol = np.linalg.cholesky(S)
        z = np.linalg.solve(chol, diff)

        return float(z @ z)


class CMC:
    """
    Camera Motion Compensation via sparse optical flow.

    Estimates the global affine transform between consecutive frames
    caused by camera vibration, water surge, or current. The transform
    is applied to Kalman states before prediction so the filter tracks
    only the fish's biological movement, not background drift.

    Uses sparse feature matching (Shi-Tomasi corners + Lucas-Kanade
    optical flow) which is lightweight and suitable for real-time
    underwater video.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 30.0,
        block_size: int = 3,
    ):
        """
        Args:
            max_corners:   Max corners for Shi-Tomasi detector.
            quality_level: Minimum accepted quality of corners.
            min_distance:  Minimum distance between corners.
            block_size:    Neighborhood size for corner detection.
        """
        self._feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03,
            ),
        )
        self._prev_gray: Optional[np.ndarray] = None

    def compute_warp(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute the 2×3 affine warp matrix between the previous and
        current frame.

        Args:
            frame: Current video frame (BGR or grayscale).

        Returns:
            2×3 affine matrix if successful, None if first frame or
            insufficient features.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        warp = None

        if self._prev_gray is not None:
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray, **self._feature_params
            )
            if prev_pts is not None and len(prev_pts) >= 4:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, prev_pts, None,
                    **self._lk_params,
                )
                # Keep only successfully tracked points
                mask = status.flatten() == 1
                if mask.sum() >= 4:
                    prev_good = prev_pts[mask].reshape(-1, 2)
                    curr_good = curr_pts[mask].reshape(-1, 2)

                    # Estimate affine transform with RANSAC
                    warp, inliers = cv2.estimateAffinePartial2D(
                        prev_good, curr_good,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0,
                    )

        self._prev_gray = gray.copy()
        return warp

    def apply_warp_to_state(
        self,
        state: np.ndarray,
        warp: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the affine warp to a Kalman state vector to compensate
        for camera motion.

        Transforms the position (u, v) and velocity (u̇, v̇) components.
        Aspect ratio and height are preserved (camera motion doesn't
        change object scale in a single frame step).

        Args:
            state: 8-dim Kalman state [u, v, a, h, u̇, v̇, ȧ, ḣ].
            warp:  2×3 affine matrix from compute_warp().

        Returns:
            Warped state vector.
        """
        state = state.copy()
        u, v = state[0], state[1]
        du, dv = state[4], state[5]

        # Transform position: [u', v'] = R @ [u, v] + t
        R = warp[:, :2]
        t = warp[:, 2]

        pos = R @ np.array([u, v]) + t
        state[0], state[1] = pos[0], pos[1]

        # Transform velocity: [u̇', v̇'] = R @ [u̇, v̇]  (no translation)
        vel = R @ np.array([du, dv])
        state[4], state[5] = vel[0], vel[1]

        return state

    def reset(self) -> None:
        """Reset CMC state (e.g., on scene cut)."""
        self._prev_gray = None
