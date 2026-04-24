"""
BioReef.ai — Grounding DINO Pseudo-Labeling
==============================================
Produces a cleaned version of the OzFish YOLO dataset by running Grounding
DINO on each training frame and merging its detections with the existing
labels (union, dedupe by IoU >= iou_thresh).

Output layout (mirrors src_dataset):
    out_dataset/
        images/train/ -> symlink to src_dataset/images/train/
        images/val/   -> symlink to src_dataset/images/val/
        images/test/  -> symlink to src_dataset/images/test/
        labels/train/<hash>/*.txt   (NEW — merged GT + pseudo-labels)
        labels/val/   -> symlink to src_dataset/labels/val/
        labels/test/  -> symlink to src_dataset/labels/test/
        train.txt  (paths rewritten to point to cleaned images dir)
        val.txt    (same)
        test.txt   (same)
        data.yaml  (nc: 1, names: [fish])

Usage (dry-run on 50 frames):
    python pseudo_label_gdino.py --max_frames 50 --device cuda:0

Usage (full):
    python pseudo_label_gdino.py --device cuda:0
"""

import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pseudo")


# =============================================================================
# Helpers
# =============================================================================

def image_to_label_path(img_path, images_root, labels_root):
    rel = os.path.relpath(img_path, images_root)
    stem, _ = os.path.splitext(rel)
    return os.path.join(labels_root, stem + ".txt")


def load_yolo_labels(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _cls, cx, cy, bw, bh = map(float, parts[:5])
            x = (cx - bw / 2) * img_w
            y = (cy - bh / 2) * img_h
            boxes.append([x, y, bw * img_w, bh * img_h])
    return boxes


def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def write_yolo_label(path, boxes_xywh, img_w, img_h, class_id=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x, y, w, h in boxes_xywh:
            cx = max(0.0, min(1.0, (x + w / 2) / img_w))
            cy = max(0.0, min(1.0, (y + h / 2) / img_h))
            bw = max(0.001, min(1.0, w / img_w))
            bh = max(0.001, min(1.0, h / img_h))
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def ensure_symlink(src, dst):
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            return
        raise RuntimeError(f"{dst} exists and is not the expected symlink to {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


def rewrite_split_txt(src_txt, dst_txt, src_images_root, dst_images_root):
    src_abs = os.path.abspath(src_images_root)
    dst_abs = os.path.abspath(dst_images_root)
    with open(src_txt) as f_in, open(dst_txt, "w") as f_out:
        for line in f_in:
            p = line.strip()
            if not p:
                continue
            p_abs = os.path.abspath(p)
            if p_abs.startswith(src_abs):
                p = dst_abs + p_abs[len(src_abs):]
            f_out.write(p + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dataset", type=str, default="datasets/ozfish")
    p.add_argument("--out_dataset", type=str, default="datasets/ozfish_cleaned")
    p.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--prompt", type=str, default="a fish.")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument("--iou_thresh", type=float, default=0.5,
                   help="GDINO detections with IoU >= this to any GT are treated as duplicates and dropped.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=None,
                   help="Limit frames processed (dry-run).")
    p.add_argument("--save_viz", action="store_true",
                   help="Save annotated frames to <out_dataset>/viz/ (green=existing GT, red=GDINO addition).")
    p.add_argument("--splits", type=str, nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="Which splits to pseudo-label. Others get symlinked. Default: train.")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Output tree ---
    out = args.out_dataset
    os.makedirs(out, exist_ok=True)
    src_images = os.path.join(args.src_dataset, "images")
    src_labels = os.path.join(args.src_dataset, "labels")
    dst_images = os.path.join(out, "images")
    dst_labels = os.path.join(out, "labels")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    for split in ("train", "val", "test"):
        s_img = os.path.join(src_images, split)
        if os.path.isdir(s_img):
            ensure_symlink(s_img, os.path.join(dst_images, split))

    # For splits we're PROCESSING: remove any existing label symlink so we can write fresh files.
    # For splits we're NOT processing: symlink to the original labels.
    for split in ("train", "val", "test"):
        d_lbl = os.path.join(dst_labels, split)
        s_lbl = os.path.join(src_labels, split)
        if split in args.splits:
            # Processing this split — remove a stale symlink if present so we can create a real dir
            if os.path.islink(d_lbl):
                os.unlink(d_lbl)
        else:
            if os.path.isdir(s_lbl):
                ensure_symlink(s_lbl, d_lbl)

    # --- Load Grounding DINO ---
    logger.info(f"Loading {args.model} on {device} ...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    viz_dir = os.path.join(out, "viz") if args.save_viz else None
    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    total_kept = total_added = total_frames_with_adds = total_frames = 0

    for split in args.splits:
        split_txt = os.path.join(args.src_dataset, f"{split}.txt")
        if not os.path.exists(split_txt):
            logger.warning(f"{split_txt} not found — skipping split '{split}'")
            continue
        with open(split_txt) as f:
            img_paths = [ln.strip() for ln in f if ln.strip()]
        if args.max_frames:
            img_paths = img_paths[: args.max_frames]

        logger.info(f"Pseudo-labeling split={split!r}: {len(img_paths)} frames (threshold={args.threshold})")
        kept = added = frames_with_adds = 0

        split_images_src = os.path.join(src_images, split)
        split_labels_src = os.path.join(src_labels, split)
        split_labels_dst = os.path.join(dst_labels, split)

        for img_path in tqdm(img_paths, desc=split):
            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                continue
            h, w = frame_bgr.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            existing = image_to_label_path(img_path, split_images_src, split_labels_src)
            gt_boxes = load_yolo_labels(existing, w, h)

            inputs = processor(images=pil, text=args.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([[h, w]], device=device)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=args.threshold,
                text_threshold=args.text_threshold,
                target_sizes=target_sizes,
            )[0]

            gdino = []
            for box in results["boxes"].cpu().numpy():
                x1, y1, x2, y2 = box
                gdino.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            gt_found_by_gdino = [False] * len(gt_boxes)
            additions = []
            for gd in gdino:
                matched_any = False
                for gi, g in enumerate(gt_boxes):
                    if iou_xywh(gd, g) >= args.iou_thresh:
                        gt_found_by_gdino[gi] = True
                        matched_any = True
                if not matched_any:
                    additions.append(gd)

            merged = list(gt_boxes) + additions
            kept += len(gt_boxes)
            added += len(additions)
            if additions:
                frames_with_adds += 1

            out_label = image_to_label_path(img_path, split_images_src, split_labels_dst)
            write_yolo_label(out_label, merged, w, h)

            if viz_dir is not None:
                canvas = frame_bgr.copy()
                for gi, g in enumerate(gt_boxes):
                    color = (255, 255, 0) if gt_found_by_gdino[gi] else (0, 200, 0)
                    x, y, bw, bh = map(int, g)
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
                for a in additions:
                    x, y, bw, bh = map(int, a)
                    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(viz_dir, f"{split}_{os.path.basename(img_path)}"), canvas)

        logger.info(
            f"  {split}: frames={len(img_paths)}, kept={kept}, added={added}, "
            f"frames_with_adds={frames_with_adds}"
        )
        total_kept += kept
        total_added += added
        total_frames_with_adds += frames_with_adds
        total_frames += len(img_paths)

    # --- Rewrite split txt files and write data.yaml ---
    for split in ("train", "val", "test"):
        src_txt = os.path.join(args.src_dataset, f"{split}.txt")
        if os.path.exists(src_txt):
            rewrite_split_txt(src_txt, os.path.join(out, f"{split}.txt"), src_images, dst_images)

    data_yaml = os.path.join(out, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n")
        f.write("nc: 1\n")
        f.write("names: [fish]\n")

    print("\n" + "=" * 60)
    print(f"Splits processed        : {args.splits}")
    print(f"Frames processed        : {total_frames}")
    print(f"Frames with additions   : {total_frames_with_adds}")
    print(f"Existing labels kept    : {total_kept}")
    print(f"GDINO pseudo-labels     : {total_added}")
    print(f"Total labels            : {total_kept + total_added}  (+{(total_added / max(1, total_kept) * 100):.1f}%)")
    print(f"Output dataset          : {out}")
    print(f"data.yaml               : {data_yaml}")
    print("=" * 60)
    print("\nRetrain command:")
    print(
        f"  yolo detect train data={data_yaml} model=yolo11m.pt "
        f"epochs=100 imgsz=960 batch=12 device=0,1 "
        f"close_mosaic=0 label_smoothing=0.05 patience=15 copy_paste=0.3"
    )


if __name__ == "__main__":
    main()
