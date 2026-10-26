"""Run RF-DETR (CFD) once per clip and cache all boxes, so every tracker in the
benchmark sees identical detections. Cached at a low conf (0.05) so trackers
that use low-score boxes (ByteTrack-style second pass) get them too.

    python detect_cache.py --n_species 15 --per_species 2
Writes cache/<species>__<clip>.npz with dets: object array of (K,5) [x1,y1,x2,y2,conf] per frame.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLIPS = REPO / "data" / "results" / "stage3_clips"
sys.path.insert(0, str(REPO))
from bioreef._2_stage1._21_detector import build_detector  # noqa: E402


def cache_clip(det, rel_path, conf=0.05):
    """Detect every frame of stage3_clips/<rel_path> once -> cache/<species>__<clip>.npz (skips if cached)."""
    dst = HERE / "cache" / (Path(rel_path).with_suffix("").as_posix().replace("/", "__") + ".npz")
    if dst.exists():
        return dst
    dst.parent.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(CLIPS / rel_path), cv2.CAP_FFMPEG)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        d = det.predict(img, conf=conf)
        frames.append(np.hstack([d.xyxy, d.conf[:, None]]).astype(np.float32))
    cap.release()
    arr = np.empty(len(frames), dtype=object)
    arr[:] = frames
    np.savez(dst, dets=arr, video=str(CLIPS / rel_path))
    return dst


def load_detector():
    return build_detector("rfdetr", weights=str(REPO / "models" / "rfdetr_medium_cfd.pth"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", help="stage3_clips folder (default: Junior-Project/data/results/stage3_clips)")
    ap.add_argument("--tracks_csv", help="cache every clip in this labels csv (stage3/labels/tracks.csv) instead")
    ap.add_argument("--n_species", type=int, default=15)
    ap.add_argument("--per_species", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.clips:
        global CLIPS
        CLIPS = Path(args.clips).expanduser()
    if args.tracks_csv:
        paths = sorted({r["path"] for r in csv.DictReader(open(args.tracks_csv, newline=""))})
        det = load_detector()
        for i, rel in enumerate(paths, 1):
            cache_clip(det, rel, args.conf)
            if i % 25 == 0:
                print(f"{i}/{len(paths)}", flush=True)
        return
    rows = [r for r in csv.DictReader(open(CLIPS / "clip_paths.csv", newline="")) if r["stage"] == "50"]
    species = sorted({r["species"] for r in rows})
    random.Random(args.seed).shuffle(species)
    picked = [r for s in species[:args.n_species]
              for r in sorted((r for r in rows if r["species"] == s), key=lambda r: int(r["label_rank"]))[:args.per_species]]

    det = load_detector()
    for i, r in enumerate(picked, 1):
        dets = np.load(cache_clip(det, r["path"], args.conf), allow_pickle=True)["dets"]
        n = [len(f[f[:, 4] >= 0.3]) for f in dets]
        print(f"[{i}/{len(picked)}] {r['path']}: {len(dets)} frames, fish/frame@0.3 mean {np.mean(n):.1f} max {max(n)}", flush=True)

if __name__ == "__main__":
    main()
