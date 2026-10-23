"""
Offline track stitching: join a track that ends to one that starts shortly after,
where the first track's motion points at the second's start, then fill the gap.

Survey video is processed offline, so the whole clip is available. This repairs
the fragmentation SU-T leaves when a fish is lost longer than its max_age, or
comes back off its predicted path. Links are one-to-one (Hungarian), so chains
A->B->C form naturally; filled-in frames carry no embedding/logits, so they never
reach Stage-3 tracklets or the species vote.
"""

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._32_track import Track


def _centre(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])


def stitch_tracks(tracks: List[Track], max_gap: int = 50, max_dist: float = 1.5,
                  vel_frames: int = 5) -> List[Track]:
    """Join fragments; returns a new track list (a joined chain keeps its first id).

    A -> B is allowed when B starts 1..max_gap frames after A ends, box sizes are
    within 2x, and B's first centre lies within max_dist x A's box diagonal of
    either A's last centre or A's centre extrapolated over the gap (velocity from
    A's last vel_frames frames). Cost = that distance / diagonal.
    """
    # ponytail: constant-velocity extrapolation, no species check yet; add the species gate (point 4) here
    hs = [sorted(t.frame_history, key=lambda h: h[0]) for t in tracks]
    n = len(tracks)
    cost = np.full((n, n), np.inf)
    for a, ha in enumerate(hs):
        if not ha:
            continue
        end_f, end_b = ha[-1][0], ha[-1][1]
        tail = ha[-vel_frames:]
        span = tail[-1][0] - tail[0][0]
        vel = (_centre(tail[-1][1]) - _centre(tail[0][1])) / span if span else np.zeros(2)
        diag = np.hypot(end_b[2], end_b[3])
        for b, hb in enumerate(hs):
            if b == a or not hb:
                continue
            gap = hb[0][0] - end_f
            if not 0 < gap <= max_gap:
                continue
            start_b = hb[0][1]
            ratio = (start_b[2] * start_b[3]) / max(end_b[2] * end_b[3], 1e-9)
            if not 0.25 <= ratio <= 4:      # area within 4x = sides within ~2x
                continue
            c = _centre(start_b)
            d = min(np.linalg.norm(c - _centre(end_b)),
                    np.linalg.norm(c - (_centre(end_b) + vel * gap))) / diag
            if d <= max_dist:
                cost[a, b] = d

    nxt = {}
    finite = np.isfinite(cost)
    if finite.any():
        rows, cols = linear_sum_assignment(np.where(finite, cost, 1e6))
        nxt = {r: c for r, c in zip(rows, cols) if finite[r, c]}

    out, has_prev = [], set(nxt.values())
    for head in range(n):
        if head in has_prev or not hs[head]:
            continue
        merged = Track(tracks[head].track_id)
        merged.confirmed = True
        cur = head
        merged.frame_history = list(hs[cur])
        while cur in nxt:
            nb = nxt[cur]
            (f0, b0, *_), (f1, b1, *_) = merged.frame_history[-1], hs[nb][0]
            for f in range(f0 + 1, f1):      # linear fill of the gap
                w = (f - f0) / (f1 - f0)
                merged.frame_history.append((f, (1 - w) * b0 + w * b1, None, None))
            merged.frame_history += hs[nb]
            cur = nb
        merged.bbox = merged.frame_history[-1][1].copy()
        out.append(merged)
    return out


if __name__ == "__main__":
    # one fish moving right, lost for 10 frames; plus an unrelated fish far away
    def track(tid, frames, x0):
        t = Track(tid)
        t.frame_history = [(f, np.array([x0 + 5.0 * f, 100, 40, 20]), None, None) for f in frames]
        return t
    a, b = track(1, range(0, 20), 0), track(2, range(30, 50), 0)
    far = track(3, range(25, 40), 1500)
    res = {t.track_id: t for t in stitch_tracks([a, b, far])}
    assert set(res) == {1, 3}, res
    fr = [h[0] for h in res[1].frame_history]
    assert fr == list(range(50)), fr                                    # gap 20..29 filled
    assert np.allclose(res[1].frame_history[25][1], [125, 100, 40, 20])  # on the straight path
    assert stitch_tracks([a, track(4, range(30, 50), 600)]).__len__() == 2  # too far: not joined
    print("stitch self-check OK")
