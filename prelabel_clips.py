"""
BioReef.ai — #17 pre-labeling orchestrator
============================================
Run the full pre-label chain on one clip, one folder of clips, or a whole
tree of clips, producing a CVAT-for-Video XML per clip ready to import as
editable Tracks.

Chain per clip (see context/problems.md #17):

    extract_frames  (mp4 -> PNGs)          [CPU]
      -> infer_stage1.py  --frames_dir     [GPU]  -> detections .npz
      -> track_stage2.py  --detections     [CPU]  -> tracklets .npz + verdicts
      -> tracklets_to_cvat.py              [CPU]  -> <clip>_tracks_cvat.xml

The detector runs on RAW frames (WaterNet is NOT applied — per #14 it costs
5.6 pp of detector recall). The WaterNet-restored copies are a separate
branch: they are the images the human looks at in CVAT, produced by
restore_videos.py. Bbox coords are pixel-identical between raw and restored,
so labels transfer losslessly.

GPU: --device defaults to cuda (GPU 0). While GPU 0 is busy with a WaterNet
restore pass, point this at the idle second card with --device cuda:1; later,
when GPU 0 is free, drop the flag.

Usage:
    # One clip
    python prelabel_clips.py --video Khorfakkan/folderX/clip01.mp4 --device cuda:1

    # All clips directly in one folder
    python prelabel_clips.py --dir Khorfakkan/folderX --device cuda:1

    # Recurse through every clip under a tree
    python prelabel_clips.py --dir Khorfakkan --recursive --device cuda:1

    # Re-run even clips whose XML already exists (default: skip them)
    python prelabel_clips.py --dir Khorfakkan --recursive --no_skip

Outputs, per clip <clip>.mp4 in its own folder:
    <clip>_frames/            extracted raw PNGs (kept for CVAT image upload;
                              --clean_frames to delete after pre-labeling)
    <clip>.mp4.NNNNNN.png     (inside <clip>_frames/, Stage 1 naming)
    outputs/detections/<clip>.mp4.npz
    outputs/tracklets/<clip>.mp4.npz   (+ _verdicts.json)
    <clip>_tracks_cvat.xml    the CVAT import file (next to the source clip)

Resumability: each per-clip stage is skipped if its output already exists,
so an interrupted batch can be re-run and picks up where it left off. The
WaterNet-restored copies (<clip>_restored.mp4) are ignored as sources.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

RESTORE_SUFFIX = "_restored"  # restore_videos.py output marker — never a source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to a single source clip.")
    src.add_argument("--dir", help="Directory of source clips.")
    p.add_argument("--recursive", action="store_true",
                   help="With --dir, walk subdirectories. Default: only files "
                        "directly in --dir.")
    p.add_argument("--exts", default="mp4,avi,mov,mkv",
                   help="Comma-separated source extensions. Default: "
                        "mp4,avi,mov,mkv.")
    p.add_argument("--no_skip", action="store_true",
                   help="Re-run clips even if their _tracks_cvat.xml exists. "
                        "Default: skip completed clips (resumable).")
    p.add_argument("--clean_frames", action="store_true",
                   help="Delete the extracted <clip>_frames/ dir after the "
                        "CVAT XML is written. Default: keep them (you need "
                        "them to upload as the CVAT task's images).")

    # --- GPU / detector knobs (threaded into infer_stage1.py) ---------------
    p.add_argument("--device", default="cuda",
                   help="Torch device for Stage 1. Default: 'cuda' (GPU 0). "
                        "Use 'cuda:1' while GPU 0 is busy with WaterNet.")
    p.add_argument("--detection_ckpt", default=None,
                   help="Detector checkpoint (default: RF-DETR CFD weights).")
    p.add_argument("--stage1_ckpt", default="bioreef_stage1.pt",
                   help="Stage 1 (MCEAM) checkpoint.")
    p.add_argument("--csv_path",
                   default="data_oz/metadata/frame_metadata_subset.csv",
                   help="Metadata CSV for the species mapping / taxonomy. "
                        "Defaults to the recovered 256-class subset that "
                        "matches bioreef_stage1.pt (see recover_species_"
                        "mapping.py / #24). Do NOT use the full 307-species "
                        "frame_metadata.csv with this checkpoint.")
    p.add_argument("--conf_threshold", type=float, default=0.3,
                   help="Detector confidence threshold (infer_stage1 default).")

    # --- tracker knobs passed to track_stage2.py ---------------------------
    p.add_argument("--label", default="fish",
                   help="CVAT label name (must match the task's label).")
    p.add_argument("--min_track_length", type=int, default=1,
                   help="Drop tracks shorter than this in the CVAT XML.")
    p.add_argument("--interp_gap", type=int, default=10,
                   help="Max detection-dropout gap (frames) CVAT interpolates "
                        "across instead of marking the fish gone. Removes box "
                        "flicker from brief detector misses; longer gaps still "
                        "terminate (probable real exit). Default: 10.")
    p.add_argument("--windowed_tracklets", dest="whole_tracks",
                   action="store_false",
                   help="Export Stage-3 style 16-30 frame windowed tracklets "
                        "instead of whole tracks. Default: whole tracks (one "
                        "CVAT track per tracker identity, no overlap-duplicate "
                        "boxes) — the right form for human merging in CVAT.")
    p.set_defaults(whole_tracks=True)
    p.add_argument("--verdicts", action="store_true",
                   help="Run the species->genus->family hierarchical "
                        "aggregation (writes <clip>_verdicts.json). OFF by "
                        "default: #17 pre-labeling only needs boxes + track "
                        "IDs, and the current checkpoint's species mapping is "
                        "unreliable (#24), which crashes aggregation. Species "
                        "labels come later via #22.")

    # --- output dirs (shared across clips) ---------------------------------
    p.add_argument("--detections_dir", default="outputs/detections")
    p.add_argument("--tracklets_dir", default="outputs/tracklets")
    return p.parse_args()


def is_source_clip(path: str, ext_set: set) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in ext_set:
        return False
    base = os.path.splitext(os.path.basename(path))[0]
    return not base.endswith(RESTORE_SUFFIX)


def gather_sources(args) -> List[str]:
    ext_set = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if args.video:
        if not os.path.exists(args.video):
            raise SystemExit(f"clip not found: {args.video}")
        return [os.path.abspath(args.video)]
    if not os.path.isdir(args.dir):
        raise SystemExit(f"directory not found: {args.dir}")

    sources: List[str] = []
    if args.recursive:
        for root, _, files in os.walk(args.dir):
            for f in files:
                full = os.path.join(root, f)
                if is_source_clip(full, ext_set):
                    sources.append(os.path.abspath(full))
    else:
        for f in os.listdir(args.dir):
            full = os.path.join(args.dir, f)
            if os.path.isfile(full) and is_source_clip(full, ext_set):
                sources.append(os.path.abspath(full))
    sources.sort()
    return sources


def run(cmd: List[str]) -> None:
    """Run a subprocess, streaming output; raise on non-zero exit."""
    print("    $ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")


def extract_all_frames(video: str, out_dir: str, video_key: str) -> int:
    """
    Extract every frame of `video` into out_dir as `<video_key>.NNNNNN.png`,
    the naming infer_stage1.py's FRAME_PATTERN groups on. Returns the frame
    count. Skips extraction if the dir already holds the expected count.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(video_key + ".") and f.endswith(".png")]
    if total > 0 and len(existing) >= total:
        cap.release()
        print(f"    frames: {len(existing)} already extracted, skipping")
        return len(existing)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{video_key}.{n:06d}.png"), frame)
        n += 1
        if n % 200 == 0:
            print(f"    extracting frames... {n}", end="\r")
    cap.release()
    print(f"    frames: extracted {n}            ")
    return n


def prelabel_one(video: str, args) -> bool:
    """Run the full chain on one clip. Returns True on success."""
    clip_dir = os.path.dirname(video)
    clip_base = os.path.splitext(os.path.basename(video))[0]      # clip01
    clip_file = os.path.basename(video)                           # clip01.mp4
    # video_key carries the extension so a "clip01.mp4" and a "clip01.avi"
    # in the same tree never collide on the shared outputs/ dirs.
    video_key = clip_file
    frames_dir = os.path.join(clip_dir, f"{clip_base}_frames")

    # infer_stage1.process_video sanitizes the output filename:
    #   video_id.replace(".avi","").replace(".","_")  -> "clip01_mp4"
    # We must look for that exact name, not "clip01.mp4.npz", or the
    # skip-if-exists / hand-off-to-Stage-2 logic silently misfires.
    safe_name = video_key.replace(".avi", "").replace(".", "_")   # clip01_mp4
    det_npz = os.path.join(HERE, args.detections_dir, f"{safe_name}.npz")
    trk_npz = os.path.join(HERE, args.tracklets_dir, f"{safe_name}.npz")
    cvat_xml = os.path.join(clip_dir, f"{clip_base}_tracks_cvat.xml")

    # --- 1. extract raw frames -----------------------------------------
    n_frames = extract_all_frames(video, frames_dir, video_key)
    if n_frames == 0:
        print("    no frames decoded; skipping clip")
        return False

    # --- 2. Stage 1 inference (GPU) ------------------------------------
    if os.path.exists(det_npz):
        print(f"    detections: {det_npz} exists, skipping Stage 1")
    else:
        cmd = [PY, "infer_stage1.py",
               "--frames_dir", frames_dir,
               "--video_id", video_key,
               "--stage1_ckpt", args.stage1_ckpt,
               "--csv_path", args.csv_path,
               "--conf_threshold", str(args.conf_threshold),
               "--device", args.device,
               "--output_dir", args.detections_dir]
        if args.detection_ckpt:
            cmd += ["--detection_ckpt", args.detection_ckpt]
        run(cmd)

    # --- 3. Stage 2 tracking (CPU) -------------------------------------
    # track_stage2.py single-video mode writes a FIXED "tracklets.npz"
    # (TrackletWriter.save default) plus a "verdicts.json" — both would be
    # clobbered by the next clip. Run it, then rename to per-clip names.
    if os.path.exists(trk_npz):
        print(f"    tracklets: {trk_npz} exists, skipping Stage 2")
    else:
        cmd = [PY, "track_stage2.py",
               "--no_frames",
               "--detections", det_npz,
               "--output_dir", args.tracklets_dir]
        # Export WHOLE tracks for CVAT, not Stage-3 windowed tracklets. The
        # default 16-30 frame windowing splits one long track into several
        # overlapping tracklets (overlap=8), which would (a) inflate the count
        # a human has to reconcile and (b) write duplicate <box> keyframes at
        # the overlaps under the same track_id. min=1 keeps short tracks;
        # max huge disables windowing so each tracker identity is exactly one
        # CVAT track. (Stage-3 training still uses the windowed defaults.)
        if args.whole_tracks:
            cmd += ["--min_tracklet_len", "1",
                    "--max_tracklet_len", "100000"]
        # Control hierarchical aggregation via --csv_path. track_stage2 runs
        # it only when the CSV path *exists*; otherwise it skips and just
        # writes tracklets — all #17 pre-labeling needs. When verdicts aren't
        # wanted we pass an explicitly non-existent path so aggregation is
        # guaranteed off. (NOT os.devnull: on Linux that's /dev/null, which
        # DOES exist, so track_stage2 would try to pd.read_csv it and crash
        # with EmptyDataError. Use a name that genuinely does not exist.)
        no_csv = os.path.join(HERE, "__no_taxonomy_disable_aggregation__")
        cmd += ["--csv_path",
                args.csv_path if args.verdicts else no_csv]
        run(cmd)
        fixed = os.path.join(HERE, args.tracklets_dir, "tracklets.npz")
        if os.path.exists(fixed):
            os.replace(fixed, trk_npz)
            # Rename the companion verdicts.json too, if aggregation ran.
            fixed_v = os.path.join(HERE, args.tracklets_dir, "verdicts.json")
            if os.path.exists(fixed_v):
                os.replace(fixed_v,
                           os.path.join(HERE, args.tracklets_dir,
                                        f"{safe_name}_verdicts.json"))

    if not os.path.exists(trk_npz):
        print("    no tracklets produced (no tracks met min length); "
              "skipping CVAT export")
        return False

    # --- 4. tracklets -> CVAT XML (CPU) --------------------------------
    run([PY, "tracklets_to_cvat.py",
         "--tracklets", trk_npz,
         "--video", video,
         "--label", args.label,
         "--min_track_length", str(args.min_track_length),
         "--interp_gap", str(args.interp_gap),
         "--out", cvat_xml])

    if args.clean_frames:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"    cleaned {frames_dir}")

    print(f"    DONE -> {cvat_xml}")
    return True


def main() -> int:
    args = parse_args()
    sources = gather_sources(args)
    if not sources:
        print("no source clips found", file=sys.stderr)
        return 1

    pending, skipped = [], []
    for s in sources:
        base = os.path.splitext(os.path.basename(s))[0]
        xml = os.path.join(os.path.dirname(s), f"{base}_tracks_cvat.xml")
        if os.path.exists(xml) and not args.no_skip:
            skipped.append(s)
        else:
            pending.append(s)

    print(f"found {len(sources)} clip(s): {len(pending)} to do, "
          f"{len(skipped)} already pre-labeled"
          f"{' (--no_skip to redo)' if skipped else ''}")
    print(f"device: {args.device}  |  WaterNet: OFF (raw frames, per #14)")
    for s in pending:
        print(f"  TODO  {s}")

    if not pending:
        print("nothing to do.")
        return 0

    t0 = time.time()
    failures = 0
    for i, video in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {video}")
        try:
            ok = prelabel_one(video, args)
            if not ok:
                failures += 1
        except Exception as e:  # one bad clip shouldn't kill the batch
            failures += 1
            print(f"    ERROR: {e}", file=sys.stderr)

    mins = (time.time() - t0) / 60
    print(f"\nFinished {len(pending) - failures}/{len(pending)} clips "
          f"in {mins:.1f} min")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
