"""Why does the anchor test detect only ~41% of expert fish at frame 60? Detector only, no tracker.

Per expert box, the best-overlapping detection under looser and looser rules:
  base      frame 60, conf >= 0.3, IoU >= 0.5      (= the anchor test's "detected")
  iou03     frame 60, conf >= 0.3, IoU >= 0.3      -> box-shape disagreement, fish was found
  cover     frame 60, conf >= 0.3, detection covers >= 50% of the expert box
  conf005   frame 60, conf >= 0.05, IoU >= 0.5     -> found, but below the pipeline threshold
  near      frames 57..63, conf >= 0.3, IoU >= 0.5 -> label sits a frame or two off
Plus base recall by expert-box size (share of the frame). Saves frame-60 images of the first
--save misses to e2e/diag_misses/ (expert box green, detections >= 0.3 red), to look at.

    python ozfish-temporal/e2e/frame60_diag.py --clips ~/Desktop/Senior_Group/stage3_clips
    python ozfish-temporal/e2e/frame60_diag.py --clips ... --detector <ft1 ckpt> --name ft1
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from anchor_test import FRAME, HERE, REPO, frames_of, iou

NEAR = range(FRAME - 3, FRAME + 4)


def best(gt, d, conf):
    """Per gt box: (max IoU, max share of the gt box covered) over detections d (K,5) with conf."""
    d = d[d[:, 4] >= conf]
    if not len(d):
        return np.zeros(len(gt)), np.zeros(len(gt))
    m = iou(gt, d[:, :4])
    ix = np.clip(np.minimum(gt[:, None, 2], d[None, :, 2]) - np.maximum(gt[:, None, 0], d[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(gt[:, None, 3], d[None, :, 3]) - np.maximum(gt[:, None, 1], d[None, :, 1]), 0, None)
    cover = ix * iy / ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))[:, None]
    return m.max(1), cover.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--boxes", default=str(REPO / "ozfish-temporal/stage3/phase0/baseline_boxes.csv"))
    ap.add_argument("--detector", default=str(REPO / "models/rfdetr_medium_cfd.pth"))
    ap.add_argument("--name", default="cfd")
    ap.add_argument("--save", type=int, default=40, help="miss images to save")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from bioreef._2_stage1._21_detector import build_detector
    det = build_detector("rfdetr", weights=a.detector)

    by_clip = collections.OrderedDict()
    for b in csv.DictReader(open(a.boxes, newline="")):
        by_clip.setdefault(b["path"], []).append(b)
    paths = list(by_clip)[:a.limit or None]
    shots = HERE / "diag_misses"
    shots.mkdir(exist_ok=True)

    rows, saved = [], 0
    for n, p in enumerate(paths, 1):
        gt = np.array([[float(b[k]) for k in ("x0", "y0", "x1", "y1")] for b in by_clip[p]])
        dets, img60 = {}, None
        for i, f in frames_of(Path(a.clips).expanduser() / p):
            if i in NEAR:
                d = det.predict(f, conf=0.05)
                dets[i] = np.hstack([d.xyxy, d.conf[:, None]])
                if i == FRAME:
                    img60 = f
            if i >= NEAR[-1]:
                break
        if FRAME not in dets:
            print("could not decode", p)
            continue
        H, W = img60.shape[:2]
        i30, c30 = best(gt, dets[FRAME], 0.3)
        i05, _ = best(gt, dets[FRAME], 0.05)
        near = np.max([best(gt, dets[f], 0.3)[0] for f in dets], axis=0)
        for k, b in enumerate(by_clip[p]):
            rows.append(dict(path=p, species=b["species"], box_index=b["box_index"],
                             area=(gt[k, 2] - gt[k, 0]) * (gt[k, 3] - gt[k, 1]) / (W * H),
                             iou=i30[k], cover=c30[k], iou_conf005=i05[k], iou_near=near[k]))
        if saved < a.save and (i30 < 0.5).any():
            im = img60.copy()
            for x0, y0, x1, y1, c in dets[FRAME][dets[FRAME][:, 4] >= 0.3]:
                cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            for x0, y0, x1, y1 in gt:
                cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)
            cv2.imwrite(str(shots / (p.replace("/", "__")[:-4] + ".jpg")), im)
            saved += 1
        if n % 100 == 0:
            print(f"{n}/{len(paths)}", flush=True)

    with open(HERE / f"frame60_diag_{a.name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = {k: np.array([x[k] for x in rows]) for k in rows[0] if k not in ("path", "species", "box_index")}
    out = [f"[{a.name}] {len(rows)} expert boxes, {len(paths)} clips",
           f"base     (f60, conf .3, IoU .5)   {np.mean(r['iou'] >= 0.5):.3f}",
           f"iou03    (f60, conf .3, IoU .3)   {np.mean(r['iou'] >= 0.3):.3f}",
           f"cover    (f60, conf .3, >=50% of expert box covered) {np.mean(r['cover'] >= 0.5):.3f}",
           f"conf005  (f60, conf .05, IoU .5)  {np.mean(r['iou_conf005'] >= 0.5):.3f}",
           f"near     (f57-63, conf .3, IoU .5) {np.mean(r['iou_near'] >= 0.5):.3f}",
           f"no detection touching the box at all (f60, conf .05): {np.mean(r['iou_conf005'] == 0):.3f}",
           "base recall by expert-box size (share of frame):"]
    for lo, hi in ((0, .001), (.001, .005), (.005, .02), (.02, 1)):
        s = (r["area"] >= lo) & (r["area"] < hi)
        out.append(f"  {lo:.3f}-{hi:.3f}: n={s.sum():5d}  recall {np.mean(r['iou'][s] >= 0.5) if s.any() else float('nan'):.3f}")
    text = "\n".join(out)
    print(text)
    (HERE / f"frame60_diag_{a.name}.txt").write_text(text + "\n")
    print(f"miss images: {shots} ({saved})")


if __name__ == "__main__":
    main()
