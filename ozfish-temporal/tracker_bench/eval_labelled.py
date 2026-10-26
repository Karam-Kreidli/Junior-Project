"""Score the tracker on the fish labelled in Stage-3 CVAT exports (problems.html #27).

Stage-3 labels box only some fish per clip, so this scores those fish, not the scene:
predicted tracks on unlabelled fish are ignored, and two unlabelled fish merging is invisible.
Per labelled fish (a GT track), each frame its box is matched one-to-one (IoU >= 0.5, Hungarian)
to a predicted box:
  coverage   share of the fish's frames matched to some track
  ids        distinct track IDs matched over its life (1 = never split)
  switches   times the matched ID changes between consecutive matched frames
  main_share share of its frames covered by its single most common ID (ID recall)
  transfers  a track that was on this fish later sits > 1 box-diagonal away from it while
             the fish is still labelled (the ID moved to another object)
  det_recall share of the fish's frames with ANY detection (conf >= 0.1) at IoU >= 0.5:
             the detector's ceiling for coverage

    python eval_labelled.py "<OzFish Clips>"/*/            # every species folder; caches detections
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import bench
from detect_cache import CLIPS, HERE, cache_clip, load_detector

IOU = 0.5


def parse_cvat(xml_bytes, folder_species):
    """One CVAT-for-video export -> (rel_clip_path, {track_id: {frame: xyxy}}), fish tracks only."""
    root = ET.fromstring(xml_bytes)
    task = (root.findtext("meta/task/name") or root.findtext("meta/job/name") or "").removesuffix(".mp4")
    parts = task.split("__")
    species = parts[0] if len(parts) == 3 else folder_species
    tracks = {}
    for tr in root.iter("track"):
        if tr.get("label") != "fish":
            continue
        boxes = {int(b.get("frame")): np.array([float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")])
                 for b in tr.iter("box") if b.get("outside") == "0"}
        if boxes:
            tracks[tr.get("id")] = boxes
    return f"{species}/{'__'.join(parts[-2:])}.mp4", tracks


def from_tracks_csv(path):
    """stage3/labels/tracks.csv (collect_annotations.py: every export shape, checked against
    clip_paths) -> {rel_clip_path: gt}. Expert and inferred tracks are both real fish."""
    import csv
    out = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path, newline="")):
        if r["outside"] == "0":
            out[r["path"]][r["track"]][int(r["frame"])] = np.array(
                [float(r[k]) for k in ("x0", "y0", "x1", "y1")])
    return {k: dict(v) for k, v in out.items()}


def collect(paths):
    """Species folders and/or zips -> {rel_clip_path: gt}. Reads every zip and every loose .xml with fish
    tracks (preload-only files have none); a clip exported twice is kept once. A .csv is read as
    stage3/labels/tracks.csv instead (the preferred source)."""
    if len(paths) == 1 and str(paths[0]).endswith(".csv"):
        return from_tracks_csv(paths[0])
    out = {}
    for p in map(Path, paths):
        files = sorted(p.iterdir()) if p.is_dir() else [p]
        folder = p if p.is_dir() else p.parent
        species = re.split(r"[ ({]", folder.name)[0]
        for f in files:
            if f.suffix == ".zip":
                with zipfile.ZipFile(f) as z:
                    blobs = [z.read(n) for n in z.namelist() if n.endswith(".xml")]
            elif f.suffix == ".xml":
                blobs = [f.read_bytes()]
            else:
                continue
            for x in blobs:
                if b'label="fish"' in x:
                    rel, gt = parse_cvat(x, species)
                    if gt:
                        out.setdefault(rel, gt)
    return out


def score(gt, pred, dets):
    """gt {gid: {f: xyxy}}, pred [frame][(tid, x1,y1,x2,y2)], dets [frame](K,5) -> list of per-fish dicts."""
    matched = defaultdict(dict)                       # gid -> {frame: tid}
    frames = sorted({f for t in gt.values() for f in t})
    for f in frames:
        if f >= len(pred):
            continue
        gids = [g for g in gt if f in gt[g]]
        P = pred[f]
        if not P:
            continue
        ious = bench.iou(np.array([gt[g][f] for g in gids]), np.array([p[1:] for p in P]))
        r, c = linear_sum_assignment(-ious)
        for i, j in zip(r, c):
            if ious[i, j] >= IOU:
                matched[gids[i]][f] = P[j][0]
    pred_box = defaultdict(dict)                      # tid -> {frame: xyxy}
    for f, P in enumerate(pred):
        for tid, *b in P:
            pred_box[tid][f] = np.array(b)
    rows = []
    for g, boxes in gt.items():
        m = matched[g]
        seq = [m[f] for f in sorted(m)]
        ids = set(seq)
        transfers = 0
        for tid in ids:                               # did a track that was on g wander off while g is still here?
            first = min(f for f in m if m[f] == tid)
            for f, pb in sorted(pred_box[tid].items()):
                if f > first and f in boxes and m.get(f) != tid:
                    gb = boxes[f]
                    diag = np.hypot(gb[2] - gb[0], gb[3] - gb[1])
                    if np.linalg.norm((pb[:2] + pb[2:]) / 2 - (gb[:2] + gb[2:]) / 2) > diag:
                        transfers += 1
                        break
        n = len(boxes)
        det_hit = sum(1 for f, gb in boxes.items() if f < len(dets) and len(d := dets[f][dets[f][:, 4] >= bench.FLOOR])
                      and bench.iou(gb[None], d[:, :4]).max() >= IOU)
        rows.append(dict(frames=n, coverage=len(m) / n, ids=len(ids),
                         switches=sum(a != b for a, b in zip(seq, seq[1:])),
                         main_share=(max(seq.count(t) for t in ids) / n) if ids else 0.0,
                         transfers=transfers, det_recall=det_hit / n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="species folders (zips + loose labelled .xml) or zip files")
    args = ap.parse_args()

    clips = list(collect(args.paths).items())
    missing = [c for c, _ in clips if not (CLIPS / c).exists()]
    if missing:
        print(f"skipping {len(missing)} clips with no local video, e.g. {missing[0]}")
    clips = [(c, g) for c, g in clips if (CLIPS / c).exists() and g]
    print(f"{len(clips)} clips, {sum(len(g) for _, g in clips)} labelled fish")

    det = None
    results = {k: [] for k in bench.trackers()}
    for i, (rel, gt) in enumerate(clips, 1):
        cached = HERE / "cache" / (rel[:-4].replace("/", "__") + ".npz")
        if not cached.exists():
            det = det or load_detector()
            cache_clip(det, rel)
        z = np.load(cached, allow_pickle=True)
        dets, video = list(z["dets"]), str(z["video"])
        for k, make in bench.trackers().items():
            for r in score(gt, bench.run(make, dets, video), dets):
                results[k].append(dict(clip=rel, **r))
        print(f"[{i}/{len(clips)}] {rel}", flush=True)

    print(f"\n{'tracker':12} {'fish':>5} {'det_recall':>10} {'coverage':>9} {'1-ID fish':>9} "
          f"{'ids/fish':>8} {'switch/fish':>11} {'main_share':>10} {'transfers':>9}")
    for k, rows in results.items():
        a = lambda key: np.mean([r[key] for r in rows])
        print(f"{k:12} {len(rows):5d} {a('det_recall'):10.3f} {a('coverage'):9.3f} "
              f"{np.mean([r['ids'] == 1 for r in rows]):9.2%} {a('ids'):8.2f} {a('switches'):11.2f} "
              f"{a('main_share'):10.3f} {sum(r['transfers'] for r in rows):9d}")
    import csv
    with open(HERE / "labelled_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tracker", *results[next(iter(results))][0]])
        w.writeheader()
        w.writerows(dict(tracker=k, **r) for k, rows in results.items() for r in rows)


if __name__ == "__main__":
    main()
