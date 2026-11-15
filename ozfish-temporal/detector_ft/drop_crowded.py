"""Remove frames too crowded to label completely from cvat_tasks/*/ (in place; rerun
prelabel_frames.py to get them back). A half-labelled frame teaches the detector that the
unboxed fish are background, so a pile of fish on the bait is better left out than guessed.

Crowded = >= --max_overlap pre-label boxes overlapping another box by >30% of the smaller one
(the bait-pile look), or >= --max_boxes boxes. Dropped frames go to dropped_crowded.csv.

    python drop_crowded.py
"""
import argparse
import csv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def crowding(boxes):
    """(n boxes, n boxes overlapping another by >30% of the smaller box's area)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(b) < 2:
        return len(b), 0
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    np.fill_diagonal(inter, 0)
    return len(b), int((inter / np.minimum(a[:, None], a[None, :]) > 0.3).any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(HERE / "cvat_tasks"))
    ap.add_argument("--max_overlap", type=int, default=10)
    ap.add_argument("--max_boxes", type=int, default=40)
    args = ap.parse_args()

    log = []
    for xml in sorted(Path(args.tasks).glob("*/annotations.xml")):
        root = ET.parse(xml).getroot()
        drop = set()
        for im in list(root.iter("image")):
            boxes = [[float(bx.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] for bx in im.iter("box")]
            n, ov = crowding(boxes)
            if ov >= args.max_overlap or n >= args.max_boxes:
                drop.add(im.get("name"))
                log.append((xml.parent.name, im.get("name"), n, ov))
                root.remove(im)
        if not drop:
            continue
        for i, im in enumerate(root.iter("image")):
            im.set("id", str(i))
        ET.ElementTree(root).write(xml, encoding="utf-8", xml_declaration=True)
        zp = xml.parent / "images.zip"
        tmp = zp.with_suffix(".tmp")
        with zipfile.ZipFile(zp) as src, zipfile.ZipFile(tmp, "w") as dst:
            for name in src.namelist():
                if name not in drop:
                    dst.writestr(name, src.read(name))
        tmp.replace(zp)
        print(f"{xml.parent.name}: dropped {len(drop)}, {sum(1 for _ in root.iter('image'))} left")

    with open(Path(args.tasks) / "dropped_crowded.csv", "a", newline="") as f:
        csv.writer(f).writerows(log)
    print(f"dropped {len(log)} frames in total")


if __name__ == "__main__":
    main()
