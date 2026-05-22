"""
Run a detector (RF-DETR or YOLO) on a directory of images, save predictions
in COCO format, and optionally render visualization PNGs.

Backends are dispatched through `bioreef.detection.build_detector`, which is
the single source of truth used by the live pipeline (infer_stage1.py,
demo_video.py, eval_pipeline.py) too. All backends produce an identical
predictions.json schema, so the same downstream scoring script can grade
either.

Usage:
    # RF-DETR (Community Fish Detector) — default per #6
    python run_detector.py

    # YOLO baseline (current trained checkpoint)
    python run_detector.py --backend yolo \\
        --weights models/best.pt \\
        --out_dir outputs/yolo_baseline

    # Other variants
    python run_detector.py --backend rfdetr --model nano \\
        --weights weights/rfdetr_nano_cfd.pth
    python run_detector.py --device cpu --conf 0.5

Outputs (in --out_dir):
    predictions.json        COCO-format predictions
    overlays/frame_*.png    Per-image visualizations
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from PIL import Image

from bioreef.detection import build_detector


# --- Defaults --------------------------------------------------------------
DEFAULT_BACKEND = "rfdetr"
DEFAULT_YOLO_WEIGHTS = "models/best.pt"
DEFAULT_IMAGES = "annotations/images"
DEFAULT_GT = "annotations/instances_default.json"
DEFAULT_OUT_RFDETR = "outputs/rfdetr_cfd"
DEFAULT_OUT_YOLO = "outputs/yolo_baseline"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["rfdetr", "yolo"], default=DEFAULT_BACKEND,
                   help=f"Detector backend. Default: {DEFAULT_BACKEND}")
    p.add_argument("--weights", default=None,
                   help="Checkpoint path. rfdetr defaults to the repo's "
                        "weights/rfdetr_medium_cfd.pth; yolo requires this arg.")
    p.add_argument("--model", default="medium",
                   choices=["medium", "small", "nano"],
                   help="RF-DETR variant (ignored for yolo). Default: medium")
    p.add_argument("--resolution", type=int, default=None,
                   help="RF-DETR inference resolution. Default: 1024 for "
                        "medium/small, 640 for nano.")
    p.add_argument("--imgsz", type=int, default=960,
                   help="YOLO inference imgsz (ignored for rfdetr). Default: 960.")
    p.add_argument("--images", default=DEFAULT_IMAGES,
                   help=f"Directory of frame PNGs. Default: {DEFAULT_IMAGES}")
    p.add_argument("--gt", default=DEFAULT_GT,
                   help=f"COCO ground-truth JSON (for image_id mapping). "
                        f"Default: {DEFAULT_GT}")
    p.add_argument("--out_dir", default=None,
                   help="Output directory. Default depends on backend: "
                        f"rfdetr -> {DEFAULT_OUT_RFDETR}, yolo -> {DEFAULT_OUT_YOLO}.")
    p.add_argument("--conf", type=float, default=0.3,
                   help="Confidence threshold. Default: 0.3")
    p.add_argument("--device", default="cuda",
                   help="'cuda' or 'cpu'. Default: cuda (falls back to cpu if unavailable).")
    p.add_argument("--no_overlays", action="store_true",
                   help="Skip per-image visualization PNGs.")
    args = p.parse_args()

    # Default weights / output dir keyed off backend
    if args.weights is None and args.backend == "yolo":
        args.weights = DEFAULT_YOLO_WEIGHTS
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_RFDETR if args.backend == "rfdetr" else DEFAULT_OUT_YOLO
    return args


def load_image_id_map(gt_path: str) -> Dict[str, int]:
    """Build {file_name -> image_id} from the COCO GT so predictions share its IDs."""
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    return {im["file_name"]: im["id"] for im in gt.get("images", [])}


def main() -> int:
    args = parse_args()

    # --- Device resolution (cuda fallback messaging) ---
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available; falling back to cpu")
        device = "cpu"
    print(f"backend: {args.backend} | device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Inputs ---
    images = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not images:
        print(f"error: no images found in {args.images}", file=sys.stderr)
        return 2
    print(f"images: {len(images)} in {args.images}")

    file_to_id = load_image_id_map(args.gt)
    if file_to_id:
        print(f"image_id map: {len(file_to_id)} entries from {args.gt}")
    else:
        print(f"note: no GT image_id map ({args.gt} missing); using filename ordinal")

    # --- Build detector via the shared abstraction ---
    try:
        detector = build_detector(
            args.backend,
            weights=args.weights,
            model_size=args.model,
            resolution=args.resolution,
            imgsz=args.imgsz,
            device=device,
        )
    except Exception as e:
        print(f"error: failed to load {args.backend} detector: {e}", file=sys.stderr)
        return 2

    # --- Output dirs ---
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "overlays")
    if not args.no_overlays:
        os.makedirs(overlay_dir, exist_ok=True)
        import supervision as sv
        box_annot = sv.BoxAnnotator(thickness=2)
        label_annot = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    # --- Inference loop ---
    predictions: List[Dict] = []
    next_id = 1
    total_dets = 0
    t0 = time.time()

    for i, fname in enumerate(images):
        pil = Image.open(os.path.join(args.images, fname)).convert("RGB")

        try:
            dets = detector.predict(pil, conf=args.conf)
        except torch.cuda.OutOfMemoryError:
            print(f"\nCUDA OOM at {fname} — retry with --device cpu "
                  f"(or rfdetr --model nano).", file=sys.stderr)
            return 3

        n = len(dets)
        total_dets += n
        image_id = file_to_id.get(fname, i + 1)

        xywh = dets.xywh
        for j in range(n):
            x, y, w, h = xywh[j]
            predictions.append({
                "id": next_id,
                "image_id": int(image_id),
                "category_id": int(dets.cls[j]) + 1,  # COCO is 1-indexed
                "bbox": [round(float(x), 2), round(float(y), 2),
                         round(float(w), 2), round(float(h), 2)],
                "score": round(float(dets.conf[j]), 4),
                "area": round(float(w * h), 2),
                "iscrowd": 0,
            })
            next_id += 1

        if not args.no_overlays:
            scene = np.array(pil)
            if n > 0:
                import supervision as sv
                sv_dets = sv.Detections(
                    xyxy=dets.xyxy.astype(float),
                    confidence=dets.conf.astype(float),
                    class_id=dets.cls.astype(int),
                )
                scene = box_annot.annotate(scene=scene, detections=sv_dets)
                labels = [f"fish {c:.2f}" for c in dets.conf]
                scene = label_annot.annotate(
                    scene=scene, detections=sv_dets, labels=labels,
                )
            Image.fromarray(scene).save(os.path.join(overlay_dir, fname))

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(images) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(images)}] {fname}  dets={n:2d}  "
                  f"{rate:4.1f} img/s  eta {eta:5.1f}s", end="\r")

    print()
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
    print(f"total detections: {total_dets} ({total_dets / len(images):.2f} per frame)")

    out_json = os.path.join(args.out_dir, "predictions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"predictions -> {out_json}")
    if not args.no_overlays:
        print(f"overlays    -> {overlay_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
