"""CVAT exports -> RF-DETR COCO dataset (train/ valid/ test/, each with _annotations.coco.json).

Reads every annotations.xml it finds under --labels (CVAT for images 1.1: a folder of the
labellers' export zips, unzipped or not, or cvat_tasks/ itself for a smoke test on the
pre-labels). Task name decides the split: test_* -> test. Train tasks are split by camera drop
into train / valid (--valid_frac of drops), so valid never shares a scene with train.

Category: one class `fish` with id 0, so it lands on CFD's own fish output (slot 0) and the
fine-tune keeps CFD's trained head instead of starting a new one.

    python build_dataset.py --labels returned/            # the team's exports
    python build_dataset.py --labels cvat_tasks --out dataset_smoke   # pipeline check on pre-labels
"""
import argparse
import csv
import io
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def xml_sources(root):
    """(task name, xml bytes) for every annotations.xml under root, inside zips too."""
    for p in sorted(Path(root).rglob("*")):
        if p.name == "annotations.xml":
            yield p.parent.name, p.read_bytes()
        elif p.suffix == ".zip" and p.name != "images.zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith("annotations.xml"):
                        yield p.stem.removesuffix("_done"), z.read(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", default=str(HERE / "frames"))
    ap.add_argument("--out", default=str(HERE / "dataset"))
    ap.add_argument("--valid_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = Path(args.frames)
    drop_of = {r["image"]: r["drop"] for r in csv.DictReader(open(frames / "manifest.csv"))}
    split_dir = {r["image"]: r["split"] for r in csv.DictReader(open(frames / "manifest.csv"))}

    images = {}  # name -> (task, w, h, boxes); a later export of the same frame replaces an earlier one
    for task, data in xml_sources(args.labels):
        for im in ET.parse(io.BytesIO(data)).getroot().iter("image"):
            name = Path(im.get("name")).name
            boxes = [[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
                     for b in im.iter("box") if b.get("label") == "fish"]
            images[name] = (task, int(im.get("width")), int(im.get("height")), boxes)
    if not images:
        raise SystemExit(f"no annotations.xml found under {args.labels}")

    train_drops = sorted({drop_of[n] for n, v in images.items() if not v[0].startswith("test")})
    random.Random(args.seed).shuffle(train_drops)
    valid_drops = set(train_drops[:round(len(train_drops) * args.valid_frac)])

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 0, "name": "fish", "supercategory": "none"}]}
            for s in ("train", "valid", "test")}
    empty = []
    for name, (task, w, h, boxes) in sorted(images.items()):
        s = "test" if task.startswith("test") else ("valid" if drop_of[name] in valid_drops else "train")
        c = coco[s]
        img_id = len(c["images"])
        c["images"].append({"id": img_id, "file_name": name, "width": w, "height": h})
        for x0, y0, x1, y1 in boxes:
            c["annotations"].append({"id": len(c["annotations"]), "image_id": img_id, "category_id": 0,
                                     "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                     "iscrowd": 0})
        (out / s).mkdir(parents=True, exist_ok=True)
        shutil.copy(frames / split_dir[name] / name, out / s / name)
        if not boxes:
            empty.append(f"{task}/{name}")
    for s, c in coco.items():
        (out / s).mkdir(parents=True, exist_ok=True)
        (out / s / "_annotations.coco.json").write_text(json.dumps(c))
        print(f"{s:5}: {len(c['images']):4d} frames, {len(c['annotations']):5d} boxes")
    if empty:
        print(f"{len(empty)} frames have no box (every frame was picked to contain a fish: check), e.g. {empty[:3]}")


if __name__ == "__main__":
    main()
