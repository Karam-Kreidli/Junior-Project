"""
Apply WaterNet to every PNG in a directory, writing restored copies to a
parallel directory. *Does not delete or modify the original frames* —
unlike offline_waternet_cache.py, which is purpose-built for the OzFish
training-data pipeline and rolling-deletes its inputs.

Used for one-shot comparison experiments (e.g. raw-vs-restored detector
testing) where you want both versions of the same frames coexisting.

Usage:
    python restore_frames.py \\
        --input  Khorfakkan/10-59-21_DEEP_TREKKER_SD_13260_13459_frames/images/default \\
        --output Khorfakkan/10-59-21_DEEP_TREKKER_SD_13260_13459_frames/images/restored
"""

import argparse
import os
import sys
import time

import cv2

from bioreef.data.data_factory import WaterNetRestorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Directory containing source PNGs (read-only).")
    p.add_argument("--output", required=True,
                   help="Directory for restored PNGs (created if missing).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip frames already present in --output (resumable).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.input):
        print(f"error: input dir not found: {args.input}", file=sys.stderr)
        return 2

    files = sorted(
        f for f in os.listdir(args.input)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        print(f"error: no images in {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    print(f"input : {args.input} ({len(files)} files)")
    print(f"output: {args.output}")

    print("loading WaterNet...")
    restorer = WaterNetRestorer()
    restorer._load_model()
    print("ready.")

    saved = skipped = failed = 0
    t0 = time.time()
    for i, fname in enumerate(files):
        in_path = os.path.join(args.input, fname)
        out_path = os.path.join(args.output, fname)

        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        img = cv2.imread(in_path)
        if img is None:
            print(f"  warn: could not read {fname}", file=sys.stderr)
            failed += 1
            continue

        try:
            restored = restorer(img)
            cv2.imwrite(out_path, restored)
            saved += 1
        except Exception as e:
            print(f"  warn: failed on {fname}: {e}", file=sys.stderr)
            failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(files)}] {rate:.2f} img/s  eta {eta:5.1f}s",
                  end="\r")

    print()
    print(f"saved {saved} | skipped {skipped} | failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
