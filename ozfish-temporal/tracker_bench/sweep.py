"""Sweep SU-T + stitching settings against the labelled fish (eval_labelled.py metrics, #28).

Uses only cached detections (detect_cache.py / eval_labelled.py first), no video decoding. Each tracker
setting is tracked once and scored with stitching off and at each stitch strictness.
Stations are split in two halves (fixed hash): pick the setting on the TUNE half, read its number on
the TEST half, so the reported score is not tuned on the fish it is reported on.

    python sweep.py ../stage3/labels/tracks.csv      -> sweep_results.csv + table sorted by tune main_share
"""

import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

TRACK = dict(track_thresh=[0.4, 0.5, 0.6], iou_threshold=[0.15, 0.25, 0.35], max_age=[30])
STITCH = [None, 1.0, 1.5]                    # None = off; else max_dist in box diagonals


def half(rel):
    """0 = tune, 1 = test; by station (deployment camera) so a site never sits in both."""
    import zlib
    station = rel.split("__")[1].split(".")[0]          # Genus_species/001__G000228_L.avi.47513.mp4
    return zlib.crc32(station.encode()) % 2


def load(paths):
    import eval_labelled as e
    clips = []
    for rel, gt in e.collect(paths).items():
        p = HERE / "cache" / (rel[:-4].replace("/", "__") + ".npz")
        if p.exists():
            clips.append((rel, gt, list(np.load(p, allow_pickle=True)["dets"])))
    return clips


def per_frame(tracks, n):
    out = [[] for _ in range(n)]
    for t in tracks:
        for f, b, *_ in t.frame_history:
            if f < n:
                out[f].append((t.track_id, b[0], b[1], b[0] + b[2], b[1] + b[3]))
    return out


def run_setting(args):
    (thresh, iou, age), clips = args
    import eval_labelled as e
    from bioreef._3_stage2 import SUTTracker, stitch_tracks
    rows = {s: [] for s in STITCH}
    for rel, gt, dets in clips:
        t = SUTTracker(track_thresh=thresh, iou_threshold=iou, max_age=age)
        for d in dets:
            d = d[d[:, 4] >= 0.1].astype(np.float64)
            xywh = d[:, :4].copy()
            xywh[:, 2:] -= xywh[:, :2]
            t.update(xywh, d[:, 4])
        tracks = t.get_all_tracks()
        for s in STITCH:
            tr = tracks if s is None else stitch_tracks(tracks, max_dist=s)
            rows[s] += [dict(species=rel.split("/")[0], half=half(rel), multi=len(gt) > 1, **r)
                        for r in e.score(gt, per_frame(tr, len(dets)), dets)]
    return [(thresh, iou, age, s, rs) for s, rs in rows.items()]


def summary(rs):
    m = lambda k: float(np.mean([r[k] for r in rs]))
    return dict(fish=len(rs), coverage=m("coverage"), one_id=float(np.mean([r["ids"] == 1 for r in rs])),
                ids=m("ids"), switches=m("switches"), main_share=m("main_share"),
                transfers=sum(r["transfers"] for r in rs))


def main():
    clips = load(sys.argv[1:])
    print(f"{len(clips)} clips with cached detections")
    settings = list(itertools.product(TRACK["track_thresh"], TRACK["iou_threshold"], TRACK["max_age"]))
    with ProcessPoolExecutor() as ex:
        results = [x for batch in ex.map(run_setting, [(s, clips) for s in settings]) for x in batch]

    table = []
    for thresh, iou, age, s, rs in results:
        tune, test = [r for r in rs if r["half"] == 0], [r for r in rs if r["half"] == 1]
        row = dict(track_thresh=thresh, iou=iou, max_age=age, stitch=s or "off",
                   **{f"tune_{k}": v for k, v in summary(tune).items()},
                   test_main=summary(test)["main_share"], test_cover=summary(test)["coverage"],
                   test_ids=summary(test)["ids"], test_transfers=summary(test)["transfers"],
                   test_multi_main=summary([r for r in test if r["multi"]])["main_share"],
                   all_main=summary(rs)["main_share"], det_recall=float(np.mean([r["det_recall"] for r in rs])))
        table.append(row)
    table.sort(key=lambda r: -r["tune_main_share"])
    current = next(r for r in table if r["track_thresh"] == 0.5 and r["iou"] == 0.25 and r["stitch"] == "off")
    print(f"tune fish {current['tune_fish']}, test fish {sum(1 for r in results[0][4] if r['half'] == 1)}, "
          f"det_recall {current['det_recall']:.3f}; current pipeline (0.5 / 0.25 / off): "
          f"test main {current['test_main']:.3f}")
    with open(HERE / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    keys = ["track_thresh", "iou", "stitch", "tune_main_share", "tune_ids", "tune_transfers",
            "test_main", "test_multi_main", "test_ids", "test_transfers", "test_cover"]
    print("  ".join(f"{k:>11}" for k in keys))
    for r in table:
        print("  ".join(f"{r[k]:>11.3f}" if isinstance(r[k], float) else f"{str(r[k]):>11}" for k in keys))


if __name__ == "__main__":
    main()
