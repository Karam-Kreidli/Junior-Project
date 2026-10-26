"""Run SU-T on cached RF-DETR detections (detect_cache.py), no GT needed.
Same SU-T as the pipeline (bioreef._3_stage2), motion-only.
Label-free proxy metrics per tracker, summed over clips:
  ids        distinct track IDs output (lower = less fragmentation, for equal coverage)
  ids/maxN   ids relative to MaxN (most confident fish in one frame); 1.0 = every fish one track
  cover      fraction of confident dets (conf>=0.5) covered by a track box (IoU>=0.5)
             -- guards against "winning" by outputting nothing
  med_len    median track length in frames
  jumps      same-ID centre moves > 1 box diagonal in one frame (likely ID switch)

    python bench.py                 # metrics table -> results.csv
    python bench.py --render 4      # + videos for the 4 most crowded clips
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

FLOOR = 0.1  # common detection floor fed to every tracker
CONF_GOOD = 0.5


class SUT:
    """The pipeline's SU-T (bioreef._3_stage2.SUTTracker), adapted to this bench's [x1,y1,x2,y2]."""

    def __init__(self):
        from bioreef._3_stage2 import SUTTracker
        self.t = SUTTracker()

    def update(self, d, img):
        xywh = d[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        return [(t.track_id, t.bbox[0], t.bbox[1], t.bbox[0] + t.bbox[2], t.bbox[1] + t.bbox[3])
                for t in self.t.update(xywh, d[:, 4])]

    def finish(self, n_frames):
        """Whole-clip output from each confirmed track's full history (incl. pre-confirmation frames)."""
        return self._per_frame(self.t.get_all_tracks(), n_frames)

    @staticmethod
    def _per_frame(tracks, n_frames):
        out = [[] for _ in range(n_frames)]
        for t in tracks:
            for f, b, *_ in t.frame_history:
                out[f].append((t.track_id, b[0], b[1], b[0] + b[2], b[1] + b[3]))
        return out


class SUTStitch(SUT):
    """SU-T, then offline stitching + gap fill (bioreef._3_stage2._36_stitch)."""

    def finish(self, n_frames):
        from bioreef._3_stage2._36_stitch import stitch_tracks
        return self._per_frame(stitch_tracks(self.t.get_all_tracks()), n_frames)


def trackers():
    return {"sut": SUT, "sut+stitch": SUTStitch}


def iou(a, b):
    """(N,4) x (M,4) xyxy -> (N,M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def frames(video):
    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield img
    cap.release()


def run(make, dets, video):
    """-> list per frame of [(id, x1, y1, x2, y2), ...]"""
    t = make()
    out = []
    for d, img in zip(dets, frames(video)):
        out.append(t.update(d[d[:, 4] >= FLOOR].astype(np.float64), img))
    # offline trackers rebuild the whole clip at the end
    return t.finish(len(out)) if hasattr(t, "finish") else out


def metrics(out, dets):
    good = [d[d[:, 4] >= CONF_GOOD, :4] for d in dets]
    n_good = sum(len(g) for g in good)
    covered = sum(int((iou(g, np.array([o[1:] for o in f]).reshape(-1, 4)).max(axis=1, initial=0) >= 0.5).sum())
                  for g, f in zip(good, out))
    life, last, jumps = {}, {}, 0
    for f in out:
        for tid, *box in f:
            box = np.array(box)
            life[tid] = life.get(tid, 0) + 1
            if tid in last:
                c0, c1 = (last[tid][:2] + last[tid][2:]) / 2, (box[:2] + box[2:]) / 2
                jumps += np.linalg.norm(c1 - c0) > np.linalg.norm(box[2:] - box[:2])
            last[tid] = box
    return dict(ids=len(life), maxN=max(len(g) for g in good), n_good=n_good, covered=covered,
                med_len=float(np.median(list(life.values()))) if life else 0.0, jumps=int(jumps))


def color(tid):
    rng = np.random.default_rng(int(tid))
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def render(name, video, outs, dst, panel=(640, 360)):
    names = list(outs)
    cols = min(3, len(names))
    rows = -(-len(names) // cols)
    W, H = panel
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W * cols, H * rows))
    for i, img in enumerate(frames(video)):
        grid = np.zeros((H * rows, W * cols, 3), np.uint8)
        for k, n in enumerate(names):
            p = img.copy()
            for tid, *b in outs[n][i]:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(p, (x1, y1), (x2, y2), color(tid), 4)
                cv2.putText(p, str(tid), (x1, max(y1 - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color(tid), 4)
            p = cv2.resize(p, panel)
            ids_so_far = len({t for f in outs[n][:i + 1] for t, *_ in f})
            cv2.putText(p, f"{n}  ids:{ids_so_far}  f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            r, c = divmod(k, cols)
            grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = p
        vw.write(grid)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=int, default=0, help="render the N most crowded clips")
    ap.add_argument("--only", nargs="*", help="subset of tracker names")
    args = ap.parse_args()

    makers = {k: v for k, v in trackers().items() if not args.only or k in args.only}
    caches = sorted((HERE / "cache").glob("*.npz"))
    tot = {k: dict(ids=0, maxN=0, n_good=0, covered=0, jumps=0, lens=[]) for k in makers}
    rows, crowd = [], []
    for c in caches:
        z = np.load(c, allow_pickle=True)
        dets, video = list(z["dets"]), str(z["video"])
        outs = {k: run(m, dets, video) for k, m in makers.items()}
        for k, out in outs.items():
            m = metrics(out, dets)
            rows.append(dict(clip=c.stem, tracker=k, **m))
            for f in ("ids", "maxN", "n_good", "covered", "jumps"):
                tot[k][f] += m[f]
            tot[k]["lens"].append(m["med_len"])
        crowd.append((rows[-1]["maxN"], c, video, outs))
        print(f"{c.stem}: maxN {rows[-1]['maxN']}  " + "  ".join(
            f"{r['tracker']} {r['ids']}" for r in rows[-len(makers):]), flush=True)

    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(caches)} clips, total MaxN {next(iter(tot.values()))['maxN']}")
    print(f"{'tracker':12} {'ids':>6} {'ids/maxN':>9} {'cover':>7} {'med_len':>8} {'jumps':>6}")
    for k, t in sorted(tot.items(), key=lambda kv: kv[1]["ids"]):
        print(f"{k:12} {t['ids']:6d} {t['ids'] / t['maxN']:9.2f} {t['covered'] / t['n_good']:7.3f} "
              f"{np.median(t['lens']):8.1f} {t['jumps']:6d}")

    if args.render:
        (HERE / "videos").mkdir(exist_ok=True)
        for maxn, c, video, outs in sorted(crowd, key=lambda x: -x[0])[:args.render]:
            dst = HERE / "videos" / f"{c.stem}.mp4"
            render(c.stem, video, outs, dst)
            print("wrote", dst)


if __name__ == "__main__":
    main()
