"""End-to-end anchor test: the whole deployed pipeline (detector -> SU-T tracker -> D6a per track
-> species/genus/family verdict) on every test clip, scored on the expert's box at frame 60.

No new labels: each clip's expert box (stage3/phase0/baseline_boxes.csv, 113 species, 321-class
decisions) is the one fish we know the truth for. Per expert box, the funnel:

  detected   a detection at frame 60 (conf >= the pipeline's) overlaps it, IoU >= thr
  tracked    a confirmed track sits on it at frame 60 (Hungarian, IoU >= thr)
  top1       that track's averaged D6a probabilities put the right species first (no threshold)
  species_ok the deployed verdict says species level AND names the right species
  genus_ok   the verdict is right at species or genus level (genus = first word of the name)
  level      what the verdict committed to: species / genus / family / unidentified

The funnel is reported at thr 0.5 and 0.3: frame60_diag.py showed OzFish expert boxes and
detector boxes often overlap only 0.3-0.5 on a fish the detector did find (96% of expert fish
have a detection touching them), so 0.5 understates the pipeline. Matches are kept down to 0.3
with their IoU in the csv, so any threshold >= 0.3 can be applied afterwards.

Compare top1 with D6a on the expert's own box (d6a_baseline.py, all321_tau1, since the
pipeline runs tau 1): the gap is what detection + tracking cost; the frames averaged over a
track can also win some back.

Run on the VM from the Junior-Project root (config.yaml paths are relative to it):
    python ozfish-temporal/e2e/anchor_test.py --clips ~/Desktop/Senior_Group/stage3_clips --name cfd --limit 20
    python ozfish-temporal/e2e/anchor_test.py --clips ~/Desktop/Senior_Group/stage3_clips --name cfd
    python ozfish-temporal/e2e/anchor_test.py --clips ... --name ft1 --detector ozfish-temporal/detector_ft/runs/ft1/checkpoint_best_total.pth
Writes ozfish-temporal/e2e/anchor_<name>.csv (one row per expert box; resumable) and prints the summary.
"""
import argparse
import collections
import csv
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FRAME = 60
STEPS = ("detected", "tracked", "top1", "species_ok", "genus_ok")
MIN_IOU = 0.3          # matches below this are dropped; the summary also gates at 0.5


def iou(a, b):
    """xyxy (N,4) x (M,4) -> (N,M)."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def assign(gt, pred):
    """(index of the pred matched to each gt or -1, its IoU or 0); one-to-one, IoU >= MIN_IOU."""
    out, val = np.full(len(gt), -1), np.zeros(len(gt))
    if len(gt) and len(pred):
        m = iou(gt, pred)
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= MIN_IOU
        out[r[ok]], val[r[ok]] = c[ok], m[r[ok], c[ok]]
    return out, val


def xyxy(xywh):
    b = np.asarray(xywh, float).reshape(-1, 4)
    return np.hstack([b[:, :2], b[:, :2] + b[:, 2:]])


def frames_of(path):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)   # MJPEG needs FFMPEG
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()


def score_clip(gt, s1, s2, s2_argmax, conf):
    """gt (K,4) xyxy expert boxes at FRAME -> one dict per box."""
    at = (s1.frame_ids == FRAME) & (s1.confidences >= conf)
    _, det_iou = assign(gt, xyxy(s1.bboxes[at]))
    on = [(i, b) for i, t in enumerate(s2.tracks) for f, b in t["frames"] if f == FRAME]
    trk, trk_iou = assign(gt, xyxy([b for _, b in on]))
    rows = []
    for k in range(len(gt)):
        r = dict(det_iou=round(det_iou[k], 4), trk_iou=round(trk_iou[k], 4), level="", name="", top1_name="",
                 confidence="", n_frames="")
        if trk[k] >= 0:
            i = on[trk[k]][0]
            v, va = s2.track_verdicts[i], s2_argmax.track_verdicts[i]
            r.update(level=v["level"], name=v["name"], confidence=v["confidence"], n_frames=v["n_frames"],
                     top1_name=va["name"])
        rows.append(r)
    return rows


def summary(rows, thr):
    """Funnel with a fish counted as found when its detection / track overlaps it at IoU >= thr."""
    rows = [{**r, "detected": int(r["det_iou"] >= thr), "tracked": int(r["trk_iou"] >= thr),
             **{k: int(r["trk_iou"] >= thr and r[k]) for k in STEPS[2:]}} for r in rows]

    def rate(rs, k):
        return np.mean([r[k] for r in rs]) if rs else float("nan")

    def macro(rs, k):
        by = collections.defaultdict(list)
        for r in rs:
            by[r["species"]].append(r[k])
        return np.mean([np.mean(v) for v in by.values()]) if by else float("nan")

    out = [f"IoU >= {thr}: {len(rows)} expert boxes, {len({r['path'] for r in rows})} clips, "
           f"{len({r['species'] for r in rows})} species",
           f"{'step':10} {'all':>7} {'macro':>7} {'d6a_test':>9}"]
    for k in STEPS:
        out.append(f"{k:10} {rate(rows, k):7.3f} {macro(rows, k):7.3f} "
                   f"{rate([r for r in rows if r['d6a_split'] == 'test'], k):9.3f}")
    tr = [r for r in rows if r["tracked"]]
    lv = collections.Counter(r["level"] for r in tr)
    out.append("verdict level on tracked fish: " + ", ".join(f"{l} {lv[l] / max(len(tr), 1):.3f}"
                                                         for l in ("species", "genus", "family", "unidentified")))
    sp = [r for r in tr if r["level"] == "species"]
    out.append(f"precision of species-level verdicts: {rate(sp, 'species_ok'):.3f} (n={len(sp)})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="stage3_clips folder")
    ap.add_argument("--boxes", default=str(REPO / "ozfish-temporal/stage3/phase0/baseline_boxes.csv"))
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--detector", help="detector weights (default: config.yaml's, i.e. CFD)")
    ap.add_argument("--name", required=True, help="output tag, e.g. cfd or ft1")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    a = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from bioreef._9_pipeline.config import InferenceConfig
    from bioreef._9_pipeline.models import load_models
    from bioreef._9_pipeline._92_detect import run_stage1
    from bioreef._9_pipeline._93_track import run_stage2

    cfg = InferenceConfig.from_yaml(a.config)
    if a.detector:
        cfg.detection_ckpt = a.detector
    cfg_argmax = dataclasses.replace(cfg, species_thresh=0.0)   # same tracks, verdict = plain argmax species
    models = load_models(cfg)

    boxes = list(csv.DictReader(open(a.boxes, newline="")))
    by_clip = collections.OrderedDict()
    for b in boxes:
        by_clip.setdefault(b["path"], []).append(b)
    paths = list(by_clip)[:a.limit or None]

    out = HERE / f"anchor_{a.name}.csv"
    done = {r["path"] for r in csv.DictReader(open(out, newline=""))} if out.exists() else set()
    cols = ["path", "species", "group", "fold", "d6a_split", "box_index",
            "det_iou", "trk_iou", "top1", "species_ok", "genus_ok", "level", "name", "top1_name",
            "confidence", "n_frames"]
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if not done:
        w.writeheader()

    t0, n = time.time(), 0
    for p in paths:
        if p in done:
            continue
        s1 = run_stage1(frames_of(Path(a.clips).expanduser() / p), models, cfg, video_id=p)
        if not len(s1):
            s2 = s2a = None
        else:
            s2 = run_stage2(s1, models, cfg)
            s2a = run_stage2(s1, models, cfg_argmax)
            assert s2.track_verdicts is not None, f"no verdicts: is csv_path ({cfg.csv_path}) present?"
        gt = np.array([[float(b[k]) for k in ("x0", "y0", "x1", "y1")] for b in by_clip[p]])
        scored = (score_clip(gt, s1, s2, s2a, cfg.conf_threshold) if s2 is not None else
                  [dict(det_iou=0, trk_iou=0, level="", name="", top1_name="", confidence="", n_frames="")] * len(gt))
        for b, r in zip(by_clip[p], scored):
            sp = b["species"]
            w.writerow({**{c: b[c] for c in cols[:6]}, **r,
                        "top1": int(r["top1_name"] == sp),
                        "species_ok": int(r["level"] == "species" and r["name"] == sp),
                        "genus_ok": int((r["level"] == "species" and r["name"] == sp) or
                                        (r["level"] == "genus" and r["name"] == sp.split()[0]))})
        fh.flush()
        n += 1
        if n % 25 == 0:
            print(f"{n + len(done)}/{len(paths)} clips  {time.time() - t0:.0f}s", flush=True)
    fh.close()

    keep = set(paths)
    rows = [r for r in csv.DictReader(open(out, newline="")) if r["path"] in keep]
    for r in rows:
        r["det_iou"], r["trk_iou"] = float(r["det_iou"]), float(r["trk_iou"])
        for k in STEPS[2:]:
            r[k] = int(r[k])
    text = summary(rows, 0.5) + "\n\n" + summary(rows, 0.3)
    print(f"\n[{a.name}] " + text)
    (HERE / f"anchor_{a.name}_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
