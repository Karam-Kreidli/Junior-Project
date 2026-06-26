"""
Profile WaterNetRestorer.forward() — per-stage timing breakdown.

The current "~2.36 s/frame" pipeline cost is one unbroken number. This
script measures each WaterNet stage in isolation so you know which one
to optimise:

    1. _wn_white_balance      (Python NumPy loop — suspected worst offender)
    2. _wn_gamma              (one np.power — cheap)
    3. _wn_histeq             (LAB + CLAHE — moderate)
    4. tensor build / to(GPU) (4 small tensors)
    5. WaterNet forward       (the actual neural net)
    6. tensor -> numpy + cvtColor (postprocess)

Run on the same hardware as the production pipeline. The breakdown tells
you whether to attack CPU preprocessing, GPU batching, or downscaling.

Usage:
    python profile_waternet.py                          # default: 30 frames from demo/video/fish.mp4
    python profile_waternet.py --video <path> --n 50
    python profile_waternet.py --images <dir> --n 50    # use an image dir instead
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List

import cv2
import numpy as np
import torch

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._1_preprocess._11_restoration import WaterNetRestorer, _wn_white_balance, _wn_gamma, _wn_histeq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="demo/video/fish.mp4",
                   help="Source video. Default: demo/video/fish.mp4")
    p.add_argument("--images", default=None,
                   help="If set, use frame PNGs from this dir instead of --video.")
    p.add_argument("--n", type=int, default=30,
                   help="Number of frames to profile. Default: 30.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warm-up frames (excluded from stats). Default: 3.")
    return p.parse_args()


@contextmanager
def cuda_timer(times: List[float], use_cuda: bool):
    """Time a block, syncing CUDA if needed for honest GPU numbers."""
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)


def load_frames(args) -> List[np.ndarray]:
    """Load N BGR frames from either a video or an image directory."""
    n = args.n + args.warmup
    if args.images:
        files = sorted(
            f for f in os.listdir(args.images)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:n]
        return [cv2.imread(os.path.join(args.images, f)) for f in files]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"video {args.video} yielded zero frames")
    return frames


def profile_one_frame(restorer: WaterNetRestorer, frame_bgr: np.ndarray,
                      stages: Dict[str, List[float]], use_cuda: bool):
    """Run the WaterNet pipeline on one frame, recording each stage's time."""
    model = restorer._model
    device = next(model.parameters()).device

    with cuda_timer(stages["bgr2rgb"], use_cuda):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with cuda_timer(stages["white_balance"], use_cuda):
        wb = _wn_white_balance(rgb)
    with cuda_timer(stages["gamma"], use_cuda):
        gc = _wn_gamma(rgb)
    with cuda_timer(stages["histeq"], use_cuda):
        he = _wn_histeq(rgb)

    with cuda_timer(stages["to_tensor"], use_cuda):
        def to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).to(device)
        rgb_t = to_tensor(rgb)
        wb_t = to_tensor(wb)
        he_t = to_tensor(he)
        gc_t = to_tensor(gc)

    with cuda_timer(stages["net_forward"], use_cuda):
        with torch.no_grad():
            out_t = model(rgb_t, wb_t, he_t, gc_t)

    with cuda_timer(stages["postprocess"], use_cuda):
        out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> int:
    args = parse_args()

    print(f"loading {args.n + args.warmup} frames "
          f"({args.warmup} warmup + {args.n} measured)...")
    frames = load_frames(args)
    print(f"got {len(frames)} frames, shape {frames[0].shape}")

    use_cuda = torch.cuda.is_available()
    print(f"cuda available: {use_cuda}")

    print("loading WaterNet...")
    restorer = WaterNetRestorer()
    restorer._load_model()  # force load now, not on first forward
    print("WaterNet loaded.")
    print()

    stages: Dict[str, List[float]] = {
        "bgr2rgb": [], "white_balance": [], "gamma": [], "histeq": [],
        "to_tensor": [], "net_forward": [], "postprocess": [],
    }

    print(f"running {len(frames)} frames ({args.warmup} warmup discarded)...")
    for i, fr in enumerate(frames):
        profile_one_frame(restorer, fr, stages, use_cuda)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(frames)}")

    # Discard warmup
    for k in stages:
        stages[k] = stages[k][args.warmup:]

    # Report
    n = len(next(iter(stages.values())))
    print()
    print("=" * 64)
    print(f"PER-STAGE TIMING — mean ± std over {n} frames")
    print(f"frame size: {frames[0].shape}  device: {'cuda' if use_cuda else 'cpu'}")
    print("=" * 64)
    print(f"  {'stage':<18s} {'mean (ms)':>12s} {'std (ms)':>10s} "
          f"{'pct':>6s}")
    total_mean = sum(np.mean(v) for v in stages.values())
    print(f"  {'-'*16:<18s} {'-'*10:>12s} {'-'*8:>10s} {'-'*4:>6s}")
    for stage, ts in stages.items():
        arr = np.array(ts) * 1000  # ms
        mean = arr.mean(); std = arr.std()
        pct = 100 * (mean / 1000) / total_mean
        print(f"  {stage:<18s} {mean:>12.2f} {std:>10.2f} {pct:>5.1f}%")
    print(f"  {'-'*16:<18s} {'-'*10:>12s} {'-'*8:>10s} {'-'*4:>6s}")
    print(f"  {'TOTAL':<18s} {total_mean*1000:>12.2f}")
    print()
    print(f"  -> per-frame WaterNet cost: {total_mean*1000:.1f} ms "
          f"({total_mean:.3f} s/frame)")
    print(f"  -> WaterNet throughput   : {1/total_mean:.2f} frames/sec")

    # Help the reader interpret the breakdown
    print()
    print("INTERPRETATION:")
    biggest = max(stages, key=lambda k: np.mean(stages[k]))
    biggest_pct = 100 * np.mean(stages[biggest]) / total_mean
    print(f"  biggest stage: '{biggest}' ({biggest_pct:.0f}% of total)")
    if biggest in ("white_balance", "histeq"):
        print(f"  → CPU preprocessing dominates. The first optimisation should")
        print(f"    target {biggest} (vectorise / use C++ OpenCV equivalent).")
    elif biggest == "net_forward":
        print(f"  → GPU forward dominates. Worth: batching, lower resolution,")
        print(f"    or model.optimize_for_inference().")
    elif biggest in ("to_tensor", "postprocess"):
        print(f"  → Tensor I/O dominates. Likely a CPU-GPU transfer bottleneck.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
