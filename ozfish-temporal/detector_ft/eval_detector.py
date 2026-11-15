"""Score a detector on the exhaustively labelled test frames (dataset/test): before vs after fine-tune.

Per frame, predictions are matched one-to-one to labelled fish (IoU >= 0.5, Hungarian). Reports, at
each confidence threshold: recall, precision, F1, and per-frame count error (|predicted - true| fish),
plus recall on small fish (< 0.5% of the frame, the distant ones CFD misses most). 0.3 is the
pipeline's detector threshold, 0.4 the tracker's start-a-track threshold.

    python eval_detector.py                                            # CFD baseline
    python eval_detector.py --weights runs/ft1/checkpoint_best_total.pth --name ft1
Appends one row per threshold to eval_results.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(gt, pred):
    """Indices of gt boxes matched to some prediction (IoU >= 0.5, one-to-one)."""
    if not len(gt) or not len(pred):
        return np.zeros(len(gt), bool)
    m = iou(gt, pred)
    r, c = linear_sum_assignment(-m)
    hit = np.zeros(len(gt), bool)
    hit[r[m[r, c] >= 0.5]] = True
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset"))
    ap.add_argument("--weights", default=str(REPO / "models" / "rfdetr_medium_cfd.pth"))
    ap.add_argument("--name", default="cfd")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from bioreef._2_stage1._21_detector import build_detector
    det = build_detector("rfdetr", weights=args.weights, device=args.device)

    test = Path(args.dataset) / "test"
    coco = json.loads((test / "_annotations.coco.json").read_text())
    gt = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append([x, y, x + w, y + h])

    stats = {t: dict(tp=0, n_gt=0, n_pred=0, small_tp=0, small_gt=0, count_err=[]) for t in THRESHOLDS}
    for im in coco["images"]:
        g = np.array(gt[im["id"]], float).reshape(-1, 4)
        d = det.predict(Image.open(test / im["file_name"]).convert("RGB"), conf=min(THRESHOLDS))
        small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 0.005 * im["width"] * im["height"]
        for t, s in stats.items():
            p = d.xyxy[d.conf >= t]
            hit = match(g, p)
            s["tp"] += hit.sum(); s["n_gt"] += len(g); s["n_pred"] += len(p)
            s["small_tp"] += hit[small].sum(); s["small_gt"] += small.sum()
            s["count_err"].append(abs(len(p) - len(g)))

    rows = []
    print(f"{args.name}: {len(coco['images'])} frames, {sum(map(len, gt.values()))} fish")
    print(f"{'conf':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'small_rec':>9} {'count_MAE':>9}")
    for t, s in stats.items():
        rec, prec = s["tp"] / max(s["n_gt"], 1), s["tp"] / max(s["n_pred"], 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        row = dict(name=args.name, conf=t, recall=rec, precision=prec, f1=f1,
                   small_recall=s["small_tp"] / max(s["small_gt"], 1), count_mae=np.mean(s["count_err"]),
                   frames=len(coco["images"]), fish=s["n_gt"], weights=args.weights)
        rows.append(row)
        print(f"{t:5.2f} {rec:7.3f} {prec:7.3f} {f1:7.3f} {row['small_recall']:9.3f} {row['count_mae']:9.2f}")

    out = HERE / "eval_results.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
