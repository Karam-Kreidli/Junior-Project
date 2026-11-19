"""CVAT exports -> RF-DETR COCO dataset (train/ valid/ test/, each with _annotations.coco.json).

Reads every annotations.xml it finds under --labels (CVAT for images 1.1, per-task or whole-project
exports, zipped or not; or cvat_tasks/ itself for a smoke test on the pre-labels). The frame's
split in frames/manifest.csv decides test vs train, so a project export holding a test task and
train tasks together is split correctly. Train frames are split by camera drop into train / valid
(--valid_frac of drops), so valid never shares a scene with train.

Frames a labeller deleted in CVAT are dropped. Depending on how they exported, a deleted frame is
either missing from the export, or still in it (usually with no boxes). So:
  * a frame with no boxes is dropped (every frame was picked to contain a fish);
  * deletedFrames_*.txt notes under --labels ("test_000:" then "frame 12" lines, CVAT frame
    numbers) drop those frames from per-task exports. Not applied to project exports, whose image
    ids are offset across tasks; those leave deleted frames out anyway.

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
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def xml_sources(root):
    """xml bytes of every annotations.xml under root, inside zips too (not images.zip)."""
    for p in sorted(Path(root).rglob("*")):
        if p.name == "annotations.xml":
            yield p.read_bytes()
        elif p.suffix == ".zip" and p.name != "images.zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith("annotations.xml"):
                        yield z.read(n)


def deleted_notes(root):
    """{(task, cvat frame number)} from the labellers' deletedFrames_*.txt notes."""
    out = set()
    for p in Path(root).rglob("deletedFrames*.txt"):
        lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        task = lines[0].rstrip(":").strip()
        out |= {(task, int(m.group(1))) for l in lines[1:] if (m := re.match(r"frame\s+(\d+)", l))}
    return out


def parse(data, deleted):
    """One export -> {frame name: (task, w, h, boxes)}, deleted frames left out; and the dropped names."""
    root = ET.parse(io.BytesIO(data)).getroot()
    names = {t.findtext("id"): t.findtext("name") for t in root.iter("task")}
    single = next(iter(names.values())) if len(names) == 1 else None
    out, dropped = {}, []
    for im in root.iter("image"):
        name = Path(im.get("name")).name
        task = names.get(im.get("task_id")) or single or "?"
        boxes = [[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
                 for b in im.iter("box") if b.get("label") == "fish"]
        noted = im.get("task_id") is None and (task, int(im.get("id"))) in deleted
        if noted or not boxes:
            dropped.append(f"{task}/{name}" + (" (deleted note)" if noted else " (no boxes)"))
            continue
        out[name] = (task, int(im.get("width")), int(im.get("height")), boxes)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", default=str(HERE / "frames"))
    ap.add_argument("--out", default=str(HERE / "dataset"))
    ap.add_argument("--valid_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = Path(args.frames)
    manifest = {r["image"]: r for r in csv.DictReader(open(frames / "manifest.csv"))}
    deleted = deleted_notes(args.labels)

    images, dropped, twice = {}, [], set()
    for data in xml_sources(args.labels):
        got, d = parse(data, deleted)
        twice |= {n for n in got if n in images and images[n][3] != got[n][3]}
        images.update(got)
        dropped += d
    if twice:  # e.g. the pre-label bundle we sent left in the folder next to the corrected export
        raise SystemExit(f"{len(twice)} frames appear in two exports with different boxes, e.g. "
                         f"{sorted(twice)[:3]}: keep one export per task under {args.labels}")
    if not images:
        raise SystemExit(f"no annotations.xml found under {args.labels}")

    train_drops = sorted({manifest[n]["drop"] for n in images if manifest[n]["split"] != "test"})
    random.Random(args.seed).shuffle(train_drops)
    valid_drops = set(train_drops[:round(len(train_drops) * args.valid_frac)])

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 0, "name": "fish", "supercategory": "none"}]}
            for s in ("train", "valid", "test")}
    tasks = {s: set() for s in coco}
    for name, (task, w, h, boxes) in sorted(images.items()):
        m = manifest[name]
        s = "test" if m["split"] == "test" else ("valid" if m["drop"] in valid_drops else "train")
        tasks[s].add(task)
        c = coco[s]
        img_id = len(c["images"])
        c["images"].append({"id": img_id, "file_name": name, "width": w, "height": h})
        for x0, y0, x1, y1 in boxes:
            c["annotations"].append({"id": len(c["annotations"]), "image_id": img_id, "category_id": 0,
                                     "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                     "iscrowd": 0})
        (out / s).mkdir(parents=True, exist_ok=True)
        shutil.copy(frames / m["split"] / name, out / s / name)
    for s, c in coco.items():
        (out / s).mkdir(parents=True, exist_ok=True)
        (out / s / "_annotations.coco.json").write_text(json.dumps(c))
        print(f"{s:5}: {len(c['images']):4d} frames, {len(c['annotations']):5d} boxes  from {sorted(tasks[s])}")
    print(f"dropped {len(dropped)} deleted frames:")
    for d in dropped:
        print("  " + d)


if __name__ == "__main__":
    main()
