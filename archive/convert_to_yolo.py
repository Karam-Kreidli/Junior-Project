"""
Convert OzFish CSV annotations to YOLO format (single-class: "fish").

All species are collapsed to a single class. Species identification is
handled downstream by MCEAM; the detector's job is class-agnostic fish
localization, which eliminates long-tail classification overfitting and
gives the detector ~260x more positive examples per class.

Storage-efficient: writes only label .txt files and path-list files.
No images are copied or symlinked. Ultralytics reads image paths from
train.txt / val.txt / test.txt, and finds labels by replacing the
nearest 'images' path component with 'labels' and .png → .txt.

Uses the SAME deterministic 80/10/10 split as detection_dataset.py.

Usage:
    python convert_to_yolo.py
"""

import argparse
import os
import random
import hashlib
import logging

import cv2
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
                    img_path = os.path.abspath(candidate)
                    break
            if not img_path:
                continue
            frame_dict[fname] = {"img_path": img_path, "annotations": []}

        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        # Single-class detector: all species collapse to class 0 ("fish").
        # MCEAM handles species ID downstream.
        frame_dict[fname]["annotations"].append((0, x0, y0, x1, y1))

    return list(frame_dict.values()), unique_sp, sp_to_idx


def split_frames(frames, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = list(frames)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return shuffled[:n_train], shuffled[n_train:n_val], shuffled[n_val:]


def write_split(frames, out_dir, split_name):
    """Write label files next to a path-list .txt for this split.

    Ultralytics label-path logic: for an image at /a/b/images/x/foo.png,
    it looks for /a/b/labels/x/foo.txt.  We exploit this by organizing:
        <out_dir>/images/<split>/foo.png  →  listed in <split>.txt as absolute path
        <out_dir>/labels/<split>/foo.txt  →  the YOLO label

    But we DON'T copy images.  Instead, <split>.txt lists the real absolute
    image paths, and we write labels into <out_dir>/labels/<split>/ with
    matching basenames.  Ultralytics resolves labels by swapping the last
    'images' component in the path with 'labels'.

    So we need the image path to contain an 'images' component.  We achieve
    this by creating one symlink per SOURCE DIRECTORY (not per file):
        <out_dir>/images/<split>/ → symlink to source dir
    That way the listed path <out_dir>/images/<split>/foo.png has 'images'
    in it, and label lookup goes to <out_dir>/labels/<split>/foo.txt.

    If multiple source directories exist, we create subdirs per source.
    """
    lbl_dir = os.path.join(out_dir, "labels", split_name)
    img_dir = os.path.join(out_dir, "images", split_name)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Group frames by their source directory
    by_source = {}
    for frame in frames:
        src_dir = os.path.dirname(frame["img_path"])
        by_source.setdefault(src_dir, []).append(frame)

    path_list = []
    n_frames = 0
    n_annots = 0
    skipped = 0

    for src_dir, src_frames in by_source.items():
        # Create a short stable name for this source directory
        dir_hash = hashlib.md5(src_dir.encode()).hexdigest()[:8]
        sub_img_dir = os.path.join(img_dir, dir_hash)
        sub_lbl_dir = os.path.join(lbl_dir, dir_hash)
        os.makedirs(sub_lbl_dir, exist_ok=True)

        # Symlink the source directory (ONE symlink, not per-file)
        if not os.path.exists(sub_img_dir):
            try:
                os.symlink(src_dir, sub_img_dir)
            except OSError:
                logger.warning(f"Cannot symlink {src_dir} → {sub_img_dir}, skipping")
                skipped += len(src_frames)
                continue

        for frame in src_frames:
            basename = os.path.basename(frame["img_path"])
            stem = os.path.splitext(basename)[0]

            # Read image dimensions for normalization
            img = cv2.imread(frame["img_path"])
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]

            # Image path that Ultralytics will see (via the dir symlink)
            listed_path = os.path.join(sub_img_dir, basename)
            path_list.append(os.path.abspath(listed_path))

            # Write YOLO label
            lbl_path = os.path.join(sub_lbl_dir, stem + ".txt")
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

    # Write path list
    list_path = os.path.join(out_dir, f"{split_name}.txt")
    with open(list_path, "w") as f:
        for p in sorted(path_list):
            f.write(p + "\n")

    logger.info(f"  {split_name}: {n_frames} frames, {n_annots} annotations, {skipped} skipped")
    return n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dirs", nargs="+", default=[
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
    ])
    parser.add_argument("--out", default="datasets/ozfish")
    args = parser.parse_args()

    img_dirs = args.img_dirs

    logger.info("Loading CSV and grouping by frame...")
    frames, class_names, sp_to_idx = load_and_group(args.csv_path, img_dirs)
    logger.info(
        f"  {len(frames)} frames, {len(class_names)} species → "
        f"collapsed to 1 class ('fish') for class-agnostic detection"
    )

    logger.info("Splitting (80/10/10, seed=42 — matches detection_dataset.py)...")
    train, val, test = split_frames(frames)

    logger.info("Writing YOLO format (labels only, images stay in place)...")
    write_split(train, args.out, "train")
    write_split(val, args.out, "val")
    write_split(test, args.out, "test")

    # Write data.yaml (single class: "fish")
    abs_out = os.path.abspath(args.out)
    data_yaml = {
        "path": abs_out,
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "nc": 1,
        "names": ["fish"],
    }
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote {yaml_path} (1 class: 'fish')")

    logger.info("Done! Train with:")
    logger.info(
        f"  yolo detect train data={yaml_path} model=yolo11m.pt "
        f"epochs=60 imgsz=640 batch=8 device=0,1 "
        f"close_mosaic=10 label_smoothing=0.05 patience=15"
    )


if __name__ == "__main__":
    main()
