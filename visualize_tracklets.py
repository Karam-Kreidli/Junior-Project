"""
Visualize the tracklets in a tracklet .npz on top of the source clip, to
diagnose why the tracker produced more IDs than there are real fish (the #11
fragmentation question).

Two outputs:

  1. An overlaid MP4 (<clip>_tracks_viz.mp4): every tracklet's boxes drawn on
     the raw frames, colored deterministically by track_id (same golden-ratio
     hue scheme as demo_video.py), with "#id" labels. Scrub it to watch where
     a single fish's ID changes.

  2. A printed TIMELINE to stdout: one row per track_id showing its first→last
     frame, length, and an ASCII bar over the clip's frame range. Fragmentation
     shows up immediately as two bars that are clearly the same fish but never
     overlap in time (an ID split across an occlusion/gap), or two bars active
     at once in the same area (an ID swap). This is usually faster than
     watching the video.

This reads the SAME .npz tracklets_to_cvat.py reads, so it shows exactly the
IDs that went into the CVAT XML — not a fresh pipeline run.

Usage:
    python visualize_tracklets.py \
        --tracklets outputs/tracklets/clip01_mp4.npz \
        --video Khorfakkan/.../clip01.mp4
    # timeline only, no video render (fast):
    python visualize_tracklets.py --tracklets ... --video ... --no_video
"""

import argparse
import colorsys
import os
import sys
from collections import defaultdict

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tracklets", required=True,
                   help="tracklets .npz from track_stage2.py / prelabel_clips.py")
    p.add_argument("--video", required=True, help="Source clip (raw frames).")
    p.add_argument("--out", default=None,
                   help="Output viz MP4. Default: <clip>_tracks_viz.mp4 next "
                        "to the source.")
    p.add_argument("--no_video", action="store_true",
                   help="Skip the MP4 render; print the timeline only (fast).")
    p.add_argument("--codec", default="mp4v",
                   help="FourCC for the output writer. Default mp4v.")
    p.add_argument("--timeline_width", type=int, default=80,
                   help="ASCII timeline width in characters. Default 80.")
    return p.parse_args()


def color_for_id(track_id: int):
    """Deterministic vivid BGR color from track ID (matches demo_video.py)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def load_tracklets(path: str):
    """Return list of (track_id, frame_ids(T,), bboxes(T,4 xywh))."""
    if not os.path.exists(path):
        raise SystemExit(f"tracklets not found: {path}")
    data = np.load(path, allow_pickle=True)
    out = []
    for tid, fids, bxs in zip(data["track_ids"], data["frame_ids"],
                              data["bboxes"]):
        out.append((int(tid), np.asarray(fids, dtype=int),
                    np.asarray(bxs, dtype=float)))
    return out


def print_timeline(tracklets, total_frames, width: int):
    """Print a per-id ASCII timeline + a fragmentation summary."""
    # Merge tracklets that share a track_id (windowed export reuses ids), so
    # the timeline reflects tracker identities, not export segments.
    by_id = defaultdict(list)
    for tid, fids, _ in tracklets:
        if len(fids):
            by_id[tid].append((int(fids.min()), int(fids.max()), len(fids)))

    span = max(1, total_frames - 1)
    print(f"\n  Timeline — {len(by_id)} distinct track IDs over "
          f"{total_frames} frames")
    print(f"  {'id':>4}  {'first':>6} {'last':>6} {'len':>5}  gaps  "
          f"|{'frames →':<{width}}|")
    print("  " + "-" * (4 + 2 + 21 + 7 + width + 2))

    for tid in sorted(by_id):
        segs = sorted(by_id[tid])
        first = segs[0][0]
        last = max(s[1] for s in segs)
        length = sum(s[2] for s in segs)
        # Internal gaps within this id (frames it was lost then recovered).
        gaps = 0
        for a, b in zip(segs, segs[1:]):
            if b[0] > a[1] + 1:
                gaps += 1

        bar = [" "] * width
        for s0, s1, _ in segs:
            i0 = int(s0 / span * (width - 1))
            i1 = int(s1 / span * (width - 1))
            for i in range(i0, i1 + 1):
                bar[i] = "#"
        print(f"  {tid:>4}  {first:>6} {last:>6} {length:>5}  {gaps:>4}  "
              f"|{''.join(bar)}|")

    print()
    # Fragmentation hint: ids whose lifespans are adjacent in time (one ends,
    # another begins shortly after) are candidate splits of the same fish.
    ids = sorted(by_id)
    spans = {tid: (min(s[0] for s in by_id[tid]),
                   max(s[1] for s in by_id[tid])) for tid in ids}
    candidates = []
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            gap = spans[b][0] - spans[a][1]
            if 0 < gap <= 30:  # b starts within 30 frames of a ending
                candidates.append((a, b, gap))
    if candidates:
        print("  Possible ID splits (one ends, another starts ≤30 frames "
              "later — likely the same fish across a gap):")
        for a, b, gap in sorted(candidates, key=lambda x: x[2]):
            print(f"    #{a} (ends f{spans[a][1]}) → #{b} "
                  f"(starts f{spans[b][0]}), gap {gap} frames")
        print("  Verify in the video; merge these in CVAT if they're one fish.")
    else:
        print("  No obvious temporal ID-split candidates "
              "(fragmentation may be spatial/overlapping instead).")


def render_video(tracklets, video, out_path, codec):
    """Draw all tracklet boxes per frame onto the clip."""
    # Build frame_idx -> list of (track_id, bbox xywh)
    per_frame = defaultdict(list)
    for tid, fids, bxs in tracklets:
        for fid, box in zip(fids.tolist(), bxs):
            per_frame[int(fid)].append((tid, box))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec),
                             fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"VideoWriter failed for codec '{codec}' -> {out_path}")

    idx = 0
    drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, (bx, by, bw, bh) in per_frame.get(idx, []):
            x, y = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            color = color_for_id(tid)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 1)
            ty = max(y, th + 4)
            cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2),
                          color, -1)
            cv2.putText(frame, label, (x + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            drawn += 1
        # Frame counter HUD
        cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {idx}", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"  wrote {out_path}  ({idx} frames, {drawn} boxes drawn)")


def main() -> int:
    args = parse_args()
    tracklets = load_tracklets(args.tracklets)
    if not tracklets:
        print("no tracklets in archive", file=sys.stderr)
        return 1

    # Total frame count from the video for accurate timeline scaling.
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() \
        else max(int(f.max()) for _, f, _ in tracklets if len(f)) + 1
    cap.release()

    n_segments = len(tracklets)
    n_ids = len({tid for tid, _, _ in tracklets})
    print(f"  {n_segments} tracklet segment(s), {n_ids} distinct track ID(s)")

    print_timeline(tracklets, total_frames, args.timeline_width)

    if not args.no_video:
        out = args.out or (os.path.splitext(args.video)[0] + "_tracks_viz.mp4")
        render_video(tracklets, args.video, out, args.codec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
