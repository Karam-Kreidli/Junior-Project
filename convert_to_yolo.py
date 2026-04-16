"""
Convert OzFish CSV annotations to YOLO format.

Creates the directory layout Ultralytics expects:
    datasets/ozfish/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        data.yaml

Uses the SAME deterministic 80/10/10 split as detection_dataset.py
so that training and evaluation are on identical splits.

Usage:
    python convert_to_yolo.py
    python convert_to_yolo.py --csv_path data_oz/metadata/frame_metadata.csv --out datasets/ozfish
"""

import argparse
import os
import random
import shutil
import logging
from collections import Counter

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FILTER_LABELS = {
    "Unidentified", "Fish", "Unknown", "unidentifiable",
    "fish", "unknown", "unidentified", "other", "Other",
    "spp", "sp1", "sp2", "sp3", "sp6", "sp10",
}


def load_and_group(csv_path: str, img_dirs: list):
    df = pd.read_csv(csv_path).dropna(subset=["species"])
    df = df[~df["species"].isin(FILTER_LABELS)]

    unique_sp = sorted(df["species"].unique().tolist())
    sp_to_idx = {sp: i for i, sp in enumerate(unique_sp)}

    frame_dict = {}
    for _, row in df.iterrows():
        fname = row["file_name"]
        if fname not in frame_dict:
            img_path = ""
            for d in img_dirs:
                candidate = os.path.join(d, fname)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                continue
            frame_dict[fname] = {"img_path": img_path, "annotations": []}

        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        cls_idx = sp_to_idx[row["species"]]
        frame_dict[fname]["annotations"].append((cls_idx, x0, y0, x1, y1))

    return list(frame_dict.values()), unique_sp, sp_to_idx


def split_frames(frames, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = list(frames)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return shuffled[:n_train], shuffled[n_train:n_val], shuffled[n_val:]


def get_image_dims(img_path):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]


def write_split(frames, out_dir, split_name):
    img_dir = os.path.join(out_dir, "images", split_name)
    lbl_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_frames = 0
    n_annots = 0
    skipped = 0

    for frame in frames:
        img_path = frame["img_path"]
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]

        w, h = get_image_dims(img_path)
        if w is None:
            skipped += 1
            continue

        # Symlink image (saves disk space)
        dst_img = os.path.join(img_dir, basename)
        if not os.path.exists(dst_img):
            abs_src = os.path.abspath(img_path)
            try:
                os.symlink(abs_src, dst_img)
            except OSError:
                shutil.copy2(img_path, dst_img)

        # Write YOLO label: class cx cy w h (normalized)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        with open(lbl_path, "w") as f:
            for cls_idx, x0, y0, x1, y1 in frame["annotations"]:
                cx = (x0 + x1) / 2.0 / w
                cy = (y0 + y1) / 2.0 / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                n_annots += 1

        n_frames += 1

    logger.info(f"  {split_name}: {n_frames} frames, {n_annots} annotations, {skipped} skipped")
    return n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dir", default="/media/openuae/UUI/frames_waternet")
    parser.add_argument("--out", default="datasets/ozfish")
    args = parser.parse_args()

    img_dirs = [
        args.img_dir,
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
        "/media/openuae/UUI/frames_waternet_3",
    ]

    logger.info("Loading CSV and grouping by frame...")
    frames, class_names, sp_to_idx = load_and_group(args.csv_path, img_dirs)
    logger.info(f"  {len(frames)} frames, {len(class_names)} classes")

    logger.info("Splitting (80/10/10, seed=42 — matches detection_dataset.py)...")
    train, val, test = split_frames(frames)

    logger.info("Writing YOLO format...")
    write_split(train, args.out, "train")
    write_split(val, args.out, "val")
    write_split(test, args.out, "test")

    # Write data.yaml
    abs_out = os.path.abspath(args.out)
    data_yaml = {
        "path": abs_out,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote {yaml_path} ({len(class_names)} classes)")

    logger.info("Done! Train with:")
    logger.info(f"  yolo detect train data={yaml_path} model=yolo11m.pt epochs=100 imgsz=640 batch=8 device=0,1")


if __name__ == "__main__":
    main()
