"""Score a detector on the exhaustively labelled test frames (dataset/test): before vs after fine-tune.

Per frame, predictions are matched one-to-one to labelled fish (IoU >= 0.5, Hungarian). Reports, at
each confidence threshold: recall, precision, F1, and per-frame count error (|predicted - true| fish),
plus recall on small fish (< 0.5% of the frame, the distant ones CFD misses most). 0.3 is the
pipeline's detector threshold, 0.4 the tracker's start-a-track threshold.

    python eval_detector.py                                            # CFD baseline
    python eval_detector.py --weights runs/ft1/checkpoint_best_total.pth --name ft1
    python eval_detector.py --tile 2 --name cfd_tile2                  # tiled: full frame + 2x2 crops
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


def predict_tiled(det, img, conf, n, overlap=0.2):
    """Detect on the full frame plus an n x n grid of overlapping crops (each crop is upscaled to the
    detector's input size, so small fish get ~n times more pixels), then merge.
    A crop's box touching an inner crop edge is dropped: the fish is cut there, and the neighbouring
    crop (for fish smaller than the overlap) or the full frame (for big ones) has it whole.
    Duplicates across passes: NMS at IoU 0.5. Returns (xyxy, conf).
    ponytail: plain NMS can keep a part-box inside a big fish's box; switch to IoS merging if
    precision drops."""
    import torch
    from torchvision.ops import nms
    W, H = img.size
    d = det.predict(img, conf=conf)
    boxes, confs = [d.xyxy], [d.conf]
    tw, th = W / (n - (n - 1) * overlap), H / (n - (n - 1) * overlap)
    for i in range(n):
        for j in range(n):
            x0, y0 = round(i * tw * (1 - overlap)), round(j * th * (1 - overlap))
            x1, y1 = min(W, round(x0 + tw)), min(H, round(y0 + th))
            d = det.predict(img.crop((x0, y0, x1, y1)), conf=conf)
            b = d.xyxy + np.array([x0, y0, x0, y0])
            cut = (((b[:, 0] <= x0 + 2) & (x0 > 0)) | ((b[:, 1] <= y0 + 2) & (y0 > 0)) |
                   ((b[:, 2] >= x1 - 2) & (x1 < W)) | ((b[:, 3] >= y1 - 2) & (y1 < H)))
            boxes.append(b[~cut]); confs.append(d.conf[~cut])
    b, c = np.concatenate(boxes).astype(np.float32), np.concatenate(confs).astype(np.float32)
    keep = nms(torch.from_numpy(b), torch.from_numpy(c), 0.5).numpy() if len(b) else np.zeros(0, int)
    return b[keep], c[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset"))
    ap.add_argument("--weights", default=str(REPO / "models" / "rfdetr_medium_cfd.pth"))
    ap.add_argument("--name", default="cfd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", help="valid: pick the conf threshold here, never on test")
    ap.add_argument("--tile", type=int, default=0, help="n: also detect on an n x n grid of crops (0 = off)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from bioreef._2_stage1._21_detector import build_detector
    det = build_detector("rfdetr", weights=args.weights, device=args.device)

    test = Path(args.dataset) / args.split
    coco = json.loads((test / "_annotations.coco.json").read_text())
    gt = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append([x, y, x + w, y + h])

    stats = {t: dict(tp=0, n_gt=0, n_pred=0, small_tp=0, small_gt=0, count_err=[]) for t in THRESHOLDS}
    for im in coco["images"]:
        g = np.array(gt[im["id"]], float).reshape(-1, 4)
        img = Image.open(test / im["file_name"]).convert("RGB")
        if args.tile:
            xyxy, conf = predict_tiled(det, img, min(THRESHOLDS), args.tile)
        else:
            d = det.predict(img, conf=min(THRESHOLDS))
            xyxy, conf = d.xyxy, d.conf
        small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 0.005 * im["width"] * im["height"]
        for t, s in stats.items():
            p = xyxy[conf >= t]
            hit = match(g, p)
            s["tp"] += hit.sum(); s["n_gt"] += len(g); s["n_pred"] += len(p)
            s["small_tp"] += hit[small].sum(); s["small_gt"] += small.sum()
            s["count_err"].append(abs(len(p) - len(g)))

    rows = []
    print(f"{args.name} [{args.split}]: {len(coco['images'])} frames, {sum(map(len, gt.values()))} fish")
    print(f"{'conf':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'small_rec':>9} {'count_MAE':>9}")
    for t, s in stats.items():
        rec, prec = s["tp"] / max(s["n_gt"], 1), s["tp"] / max(s["n_pred"], 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        row = dict(name=args.name, split=args.split, tile=args.tile, conf=t, recall=rec, precision=prec, f1=f1,
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
