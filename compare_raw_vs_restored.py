"""
Does WaterNet actually help RF-DETR on Khorfakkan?

Run RF-DETR on the same N frames in two modes — raw, and WaterNet-restored
— and compare detection counts, average confidence, and how often the
restored result genuinely adds boxes the raw didn't have (IoU-based).

If raw and restored produce similar detection sets, **WaterNet doesn't
need to be in the production inference pipeline at all** — only the
labeling-verification workflow needs it (a human eye benefits even if
the detector doesn't). That single conclusion eliminates the biggest
single bottleneck in the production speed problem.

Usage:
    python compare_raw_vs_restored.py                          # default: 50 frames from demo/video/fish.mp4
    python compare_raw_vs_restored.py --video <path> --n 100
    python compare_raw_vs_restored.py --images <dir> --n 50
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

from bioreef.data.data_factory import WaterNetRestorer
from bioreef.detection import build_detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="demo/video/fish.mp4")
    p.add_argument("--images", default=None,
                   help="If set, use frame PNGs from this dir instead of --video.")
    p.add_argument("--n", type=int, default=50,
                   help="Number of frames to test. Default: 50.")
    p.add_argument("--conf", type=float, default=0.05,
                   help="RF-DETR confidence threshold. Default: 0.05.")
    p.add_argument("--iou_match", type=float, default=0.5,
                   help="IoU for matching raw vs restored detections. Default: 0.5.")
    return p.parse_args()


def load_frames(args) -> List[np.ndarray]:
    if args.images:
        files = sorted(
            f for f in os.listdir(args.images)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:args.n]
        return [cv2.imread(os.path.join(args.images, f)) for f in files]
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    frames = []
    while len(frames) < args.n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit("zero frames loaded")
    return frames


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two box arrays in xyxy. Returns (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_counts(raw_xyxy: np.ndarray, rest_xyxy: np.ndarray, thr: float
                 ) -> Tuple[int, int, int]:
    """
    Per-frame: how many boxes appear in both / only raw / only restored.

    Returns (shared, only_raw, only_restored).
    """
    if len(raw_xyxy) == 0 and len(rest_xyxy) == 0:
        return 0, 0, 0
    if len(raw_xyxy) == 0:
        return 0, 0, len(rest_xyxy)
    if len(rest_xyxy) == 0:
        return 0, len(raw_xyxy), 0
    iou = iou_xyxy(raw_xyxy, rest_xyxy)
    # Greedy matching: highest IoU first
    matched_raw = np.zeros(len(raw_xyxy), dtype=bool)
    matched_rest = np.zeros(len(rest_xyxy), dtype=bool)
    flat = sorted(
        ((iou[i, j], i, j) for i in range(len(raw_xyxy)) for j in range(len(rest_xyxy))),
        key=lambda t: -t[0],
    )
    shared = 0
    for v, i, j in flat:
        if v < thr:
            break
        if matched_raw[i] or matched_rest[j]:
            continue
        matched_raw[i] = True
        matched_rest[j] = True
        shared += 1
    return shared, int((~matched_raw).sum()), int((~matched_rest).sum())


def main() -> int:
    args = parse_args()
    frames = load_frames(args)
    print(f"loaded {len(frames)} frames @ {frames[0].shape}")

    print("loading RF-DETR (rfdetr backend, default weights)...")
    detector = build_detector("rfdetr")
    print("loading WaterNet...")
    restorer = WaterNetRestorer()
    restorer._load_model()
    print()

    raw_counts: List[int] = []
    rest_counts: List[int] = []
    raw_conf_means: List[float] = []
    rest_conf_means: List[float] = []
    shared_per_frame: List[int] = []
    only_raw_per_frame: List[int] = []
    only_rest_per_frame: List[int] = []

    t_raw_total = 0.0
    t_rest_total = 0.0
    t_wn_total = 0.0

    for i, fr in enumerate(frames):
        # raw detection
        t0 = time.perf_counter()
        d_raw = detector.predict(fr, conf=args.conf)
        t_raw_total += time.perf_counter() - t0

        # restore + detect
        t0 = time.perf_counter()
        fr_rest = restorer(fr)
        t_wn_total += time.perf_counter() - t0
        t0 = time.perf_counter()
        d_rest = detector.predict(fr_rest, conf=args.conf)
        t_rest_total += time.perf_counter() - t0

        raw_counts.append(len(d_raw))
        rest_counts.append(len(d_rest))
        raw_conf_means.append(float(d_raw.conf.mean()) if len(d_raw) else 0.0)
        rest_conf_means.append(float(d_rest.conf.mean()) if len(d_rest) else 0.0)

        s, or_, orest = match_counts(d_raw.xyxy, d_rest.xyxy, args.iou_match)
        shared_per_frame.append(s)
        only_raw_per_frame.append(or_)
        only_rest_per_frame.append(orest)

        if (i + 1) % 10 == 0:
            print(f"  {i+1:3d}/{len(frames)}  raw={len(d_raw):2d} restored={len(d_rest):2d} "
                  f"shared={s:2d} only-raw={or_:2d} only-rest={orest:2d}")

    # --- report -----------------------------------------------------------
    raw_c = np.array(raw_counts); rest_c = np.array(rest_counts)
    shared = np.array(shared_per_frame); o_raw = np.array(only_raw_per_frame); o_rest = np.array(only_rest_per_frame)

    print()
    print("=" * 64)
    print(f"DETECTION COUNT — over {len(frames)} frames")
    print("=" * 64)
    print(f"  raw      : {raw_c.sum():>5d} total, {raw_c.mean():.2f} mean/frame, "
          f"avg conf {np.mean(raw_conf_means):.3f}")
    print(f"  restored : {rest_c.sum():>5d} total, {rest_c.mean():.2f} mean/frame, "
          f"avg conf {np.mean(rest_conf_means):.3f}")
    diff = rest_c.sum() - raw_c.sum()
    pct = 100 * diff / max(raw_c.sum(), 1)
    print(f"  delta    : restored - raw = {diff:+d} ({pct:+.1f}%)")
    print()
    print(f"PER-FRAME OVERLAP (IoU>={args.iou_match})")
    print(f"  shared (both saw it)        : {shared.sum():>5d}  mean/frame {shared.mean():.2f}")
    print(f"  only raw (lost on restore)  : {o_raw.sum():>5d}  mean/frame {o_raw.mean():.2f}")
    print(f"  only restored (added)       : {o_rest.sum():>5d}  mean/frame {o_rest.mean():.2f}")
    print()
    print(f"TIMING (per frame)")
    print(f"  RF-DETR on raw      : {1000*t_raw_total/len(frames):.1f} ms")
    print(f"  RF-DETR on restored : {1000*t_rest_total/len(frames):.1f} ms")
    print(f"  WaterNet restore   : {1000*t_wn_total/len(frames):.1f} ms")
    print()
    print("INTERPRETATION:")
    if abs(pct) < 5 and o_rest.sum() < 0.1 * raw_c.sum():
        print(f"  Restored adds <5% extra detections AND <10% genuinely-new boxes.")
        print(f"  → WaterNet does not meaningfully help RF-DETR on this footage.")
        print(f"    Strong case for dropping WaterNet from production inference,")
        print(f"    keeping it only for human-verification labeling.")
    elif o_rest.sum() > 0.3 * raw_c.sum():
        print(f"  Restored adds {o_rest.sum()/raw_c.sum()*100:.0f}% genuinely-new boxes.")
        print(f"  → WaterNet IS meaningfully helping the detector.")
        print(f"    Keep WaterNet in inference; speed work focuses on optimising it.")
    else:
        print(f"  Mixed signal: {pct:+.1f}% change in count, "
              f"{o_rest.sum()}/{raw_c.sum()} new boxes from restored.")
        print(f"  → Worth eyeballing the actual overlays before deciding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
