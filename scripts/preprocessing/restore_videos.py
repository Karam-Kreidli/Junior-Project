"""
Apply WaterNet to one or more video files, writing restored copies as
<source>_restored.mp4 next to each source.

Intended for the #17 labeling workflow: a human verifier sees restored
frames clearly in CVAT even though the detector itself runs on raw frames
in production. Inference does NOT use these files (per #14, dropping
WaterNet from inference improves detector recall by 5.6 pp); they exist
only for the human-eye part of the labeling loop.

Cost notice: WaterNet costs ~1.78 s/frame on the Quadro 4000s. A 1-minute
clip at 30 fps = 1800 frames = ~53 minutes. Plan accordingly — this script
is meant for multi-hour / multi-day batch runs, with --skip_existing for
resumability if interrupted.

Audio NOT preserved (OpenCV's VideoWriter is video-only). Reef footage
audio is usually engine noise, so this is fine for labeling. If you ever
need audio, mux it back with ffmpeg after the fact:
    ffmpeg -i source.mp4 -i source_restored.mp4 -c copy -map 1:v -map 0:a out.mp4

Usage:
    # Single video
    python restore_videos.py --video Khorfakkan/folder/clip01.mp4

    # All .mp4 files in one directory
    python restore_videos.py --dir Khorfakkan/folder

    # Walk recursively, all .mp4 files anywhere under --dir
    python restore_videos.py --dir Khorfakkan --recursive

    # Force re-restoration of already-restored files (default: skip them)
    python restore_videos.py --dir Khorfakkan --recursive --no_skip
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import cv2

# --- repo-root bootstrap: this script lives in scripts/<area>/; add the
# repo root (two levels up) to sys.path so `import bioreef` resolves no
# matter the cwd or how the script is invoked. ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from bioreef._1_preprocess._11_restoration import WaterNetRestorer


SUFFIX = "_restored"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to a single source video.")
    src.add_argument("--dir", help="Directory of source videos.")
    p.add_argument("--recursive", action="store_true",
                   help="With --dir, walk subdirectories. Default: only "
                        "files directly in --dir.")
    p.add_argument("--no_skip", action="store_true",
                   help="Re-restore videos even if <name>_restored.mp4 "
                        "already exists. Default: skip them (resumable).")
    p.add_argument("--exts", default="mp4,avi,mov,mkv",
                   help="Comma-separated source extensions to consider when "
                        "using --dir. Default: mp4,avi,mov,mkv.")
    p.add_argument("--codec", default="mp4v",
                   help="FourCC code for the output VideoWriter. Default: "
                        "'mp4v' (broadly compatible). Try 'avc1' (H.264) "
                        "if your environment has it.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Resize frames by this factor before WaterNet and in "
                        "the output. 0.5 = half resolution (4x CPU speedup). "
                        "Default: 1.0 (no resize).")
    p.add_argument("--every_n", type=int, default=1,
                   help="Process every Nth frame; duplicate the restored frame "
                        "N times so output length matches input. Default: 1 "
                        "(every frame). 2 = 2x speedup at half temporal res.")
    return p.parse_args()


def is_source_video(path: str, ext_set: set) -> bool:
    """True if path is a video file we should process, and not itself a
    previously-restored output."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in ext_set:
        return False
    # Don't recursively re-restore previously-restored outputs.
    base = os.path.splitext(os.path.basename(path))[0]
    return not base.endswith(SUFFIX)


def gather_sources(args) -> List[str]:
    ext_set = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if args.video:
        if not os.path.exists(args.video):
            raise SystemExit(f"video not found: {args.video}")
        return [os.path.abspath(args.video)]
    if not os.path.isdir(args.dir):
        raise SystemExit(f"directory not found: {args.dir}")

    sources: List[str] = []
    if args.recursive:
        for root, _, files in os.walk(args.dir):
            for f in files:
                full = os.path.join(root, f)
                if is_source_video(full, ext_set):
                    sources.append(os.path.abspath(full))
    else:
        for f in sorted(os.listdir(args.dir)):
            full = os.path.join(args.dir, f)
            if os.path.isfile(full) and is_source_video(full, ext_set):
                sources.append(os.path.abspath(full))
    sources.sort()
    return sources


def output_path_for(source: str) -> str:
    """Place <name>_restored.mp4 next to the source."""
    root, ext = os.path.splitext(source)
    # Always write .mp4 (the input might be .avi/.mov but we standardise out).
    return root + SUFFIX + ".mp4"


def restore_one(
    source: str, output: str, restorer: WaterNetRestorer, codec: str,
    scale: float = 1.0, every_n: int = 1,
) -> Tuple[bool, float, int]:
    """
    Restore one video frame-by-frame.

    Returns (success, elapsed_seconds, frames_written).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"    ERROR: could not open {source}", file=sys.stderr)
        return False, 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = max(2, int(w * scale) // 2 * 2)
    out_h = max(2, int(h * scale) // 2 * 2)
    scale_label = f" (scaled to {out_w}x{out_h})" if scale != 1.0 else ""
    every_label = f", every {every_n} frames" if every_n > 1 else ""
    print(f"    {w}x{h} @ {fps:.1f} fps, ~{total} frames"
          f"{scale_label}{every_label}")

    out_w = max(2, int(w * scale) // 2 * 2)
    out_h = max(2, int(h * scale) // 2 * 2)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    # Write to a `.partial.mp4` first so an interrupted run doesn't leave
    # a half-finished _restored.mp4 that would later be wrongly skipped.
    # The temp extension keeps `.mp4` at the end so OpenCV/FFmpeg infers
    # the right muxer from the extension (writing to `<name>.mp4.partial`
    # silently fails because FFmpeg can't recognise that as a container).
    root, ext = os.path.splitext(output)
    tmp_out = root + ".partial" + ext
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        print(f"    ERROR: VideoWriter would not open with codec '{codec}' "
              f"writing to {tmp_out!r}. Source: {w}x{h} @ {fps} fps. "
              f"Try a different --codec (XVID/MJPG output as .avi).",
              file=sys.stderr)
        return False, 0.0, 0

    n = 0
    src_frame = 0
    t0 = time.time()
    last_restored = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_frame += 1
        if (src_frame - 1) % every_n == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            last_restored = restorer(frame)
        # Write last_restored every source frame (duplicates on skip frames)
        for _ in range(1):
            writer.write(last_restored)
        n += 1
        if n % 30 == 0:
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (total - n) / rate if rate > 0 and total > 0 else 0
            print(f"    [{n:>6d}/{total}]  {rate:.2f} f/s  "
                  f"eta {eta/60:.1f} min", end="\r")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    # Atomic-rename the .partial into the final name only after success.
    os.replace(tmp_out, output)
    print()
    return True, elapsed, n


def main() -> int:
    args = parse_args()

    sources = gather_sources(args)
    if not sources:
        print("no source videos found", file=sys.stderr)
        return 1

    print(f"found {len(sources)} source video(s):")
    for s in sources:
        print(f"  {s}")

    # Filter out already-restored unless --no_skip
    pending = []
    skipped = []
    for s in sources:
        out = output_path_for(s)
        if os.path.exists(out) and not args.no_skip:
            skipped.append(s)
        else:
            pending.append((s, out))
    if skipped:
        print(f"\nskipping {len(skipped)} already-restored "
              f"({'--no_skip to force' if not args.no_skip else 'no_skip on'}):")
        for s in skipped:
            print(f"  {s}")

    if not pending:
        print("\nnothing to do.")
        return 0

    print(f"\nrestoring {len(pending)} video(s) "
          f"(this is the long-running part)...")

    print("\nloading WaterNet...")
    restorer = WaterNetRestorer()
    restorer._load_model()
    print("WaterNet loaded.\n")

    total_elapsed = 0.0
    total_frames = 0
    failures = 0

    t_global = time.time()
    for i, (src, out) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {src}")
        print(f"     -> {out}")
        ok, elapsed, n = restore_one(src, out, restorer, args.codec,
                                      scale=args.scale, every_n=args.every_n)
        if not ok:
            failures += 1
            continue
        total_elapsed += elapsed
        total_frames += n
        per_frame = elapsed / max(n, 1)
        print(f"    done: {n} frames in {elapsed/60:.1f} min "
              f"({per_frame:.2f} s/frame)")

    grand_total = time.time() - t_global
    print(f"\nFinished {len(pending) - failures}/{len(pending)} videos "
          f"in {grand_total/60:.1f} min total "
          f"({total_frames} frames, "
          f"avg {(total_elapsed / max(total_frames, 1)):.2f} s/frame)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
