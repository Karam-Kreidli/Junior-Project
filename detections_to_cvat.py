"""
Convert a Stage-1 detections .npz (output of infer_stage1.py) into a
CVAT-for-Video 1.1 XML of BOXES ONLY — no track-identity linking.

Why boxes-only (issue #17, #11/#8 finding): on Gulf-of-Oman footage the
detector's per-frame recall is too inconsistent for the tracker to produce
clean persistent IDs (empirically, tuning the tracker bottomed out at
~17-57 fragmented IDs for 8 real fish, and lowering the detector conf made
it worse via false positives). The detector DOES find the fish, just
intermittently — so the useful, trustworthy product is the per-frame boxes.
The human draws the persistent track IDs by hand in CVAT (the part the
tracker fails at), keeping the tedious box-drawing the detector did well.

Each detection becomes one single-frame box. In CVAT these import as
independent boxes you then assign to tracks yourself (Group / Merge in the
track tools), or relabel/delete.

This reads the detections .npz (frame_ids, bboxes xywh, confidences), NOT
the tracklets .npz — it deliberately bypasses the tracker.

Usage:
    python detections_to_cvat.py \
        --detections outputs/detections/clip01_mp4.npz \
        --video Khorfakkan/.../clip01.mp4 \
        --out clip01_boxes_cvat.xml
    # filter weak boxes:
    python detections_to_cvat.py --detections ... --video ... --min_conf 0.3
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--detections", required=True,
                   help="Stage-1 detections .npz from infer_stage1.py.")
    p.add_argument("--video", default=None,
                   help="Source video — num_frames is read from it (OpenCV).")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Total frames in the source video. Required if "
                        "--video not given.")
    p.add_argument("--label", default="fish",
                   help="Label name (must match the CVAT task's label).")
    p.add_argument("--min_conf", type=float, default=0.0,
                   help="Drop detections below this confidence. Default 0 "
                        "(keep all the detector emitted).")
    p.add_argument("--out", default="boxes_cvat.xml",
                   help="Output XML path.")
    p.add_argument("--task_name", default="khorfakkan_boxes",
                   help="Cosmetic task name in <meta>.")
    return p.parse_args()


def num_frames_from_video(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f"video reported {n} frames: {path}")
    return n


def load_detections(npz_path: str):
    """Return (frame_ids (N,), bboxes (N,4 xywh), confidences (N,))."""
    if not os.path.exists(npz_path):
        raise SystemExit(f"detections not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    fids = np.asarray(data["frame_ids"], dtype=int)
    bxs = np.asarray(data["bboxes"], dtype=float)
    confs = np.asarray(data["confidences"], dtype=float) \
        if "confidences" in data else np.ones(len(fids))
    return fids, bxs, confs


def build_xml(fids, bxs, confs, num_frames, label_name, task_name, min_conf):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(num_frames)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "created").text = datetime.utcnow().isoformat()
    ET.SubElement(task, "updated").text = datetime.utcnow().isoformat()
    ET.SubElement(task, "subset").text = "default"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(num_frames - 1)
    ET.SubElement(task, "frame_filter").text = ""

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = label_name
    ET.SubElement(label, "color").text = "#33ddff"
    ET.SubElement(label, "type").text = "rectangle"
    ET.SubElement(label, "attributes")

    ET.SubElement(meta, "dumped").text = datetime.utcnow().isoformat()

    # Boxes-only: emit per-frame <track>s each containing a single box plus
    # its outside=1 terminator. One track per detection — no cross-frame
    # identity. CVAT shows each as an independent box to group/merge by hand.
    written = 0
    track_id = 0
    for fid, (x, y, w, h), c in zip(fids.tolist(), bxs.tolist(), confs.tolist()):
        if c < min_conf:
            continue
        fid = int(fid)
        xtl, ytl, xbr, ybr = x, y, x + w, y + h
        track_el = ET.SubElement(root, "track", {
            "id": str(track_id),
            "label": label_name,
            "source": "auto",
        })
        ET.SubElement(track_el, "box", {
            "frame": str(fid),
            "outside": "0", "occluded": "0", "keyframe": "1",
            "xtl": f"{xtl:.2f}", "ytl": f"{ytl:.2f}",
            "xbr": f"{xbr:.2f}", "ybr": f"{ybr:.2f}",
            "z_order": "0",
        })
        # Terminator one frame later so CVAT doesn't extrapolate the box.
        term = min(fid + 1, num_frames - 1)
        if term > fid:
            ET.SubElement(track_el, "box", {
                "frame": str(term),
                "outside": "1", "occluded": "0", "keyframe": "1",
                "xtl": f"{xtl:.2f}", "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}", "ybr": f"{ybr:.2f}",
                "z_order": "0",
            })
        track_id += 1
        written += 1

    return ET.ElementTree(root), written


def main() -> int:
    args = parse_args()

    if args.video:
        num_frames = num_frames_from_video(args.video)
    elif args.num_frames is not None:
        num_frames = args.num_frames
    else:
        raise SystemExit("provide --video or --num_frames")

    fids, bxs, confs = load_detections(args.detections)
    total = len(fids)

    tree, written = build_xml(fids, bxs, confs, num_frames,
                              args.label, args.task_name, args.min_conf)

    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    with open(args.out, "wb") as f:
        f.write(pretty)

    dropped = total - written
    print(f"wrote {written} boxes -> {args.out}")
    if dropped:
        print(f"  ({dropped} dropped below --min_conf {args.min_conf})")
    print(f"  label: '{args.label}'  num_frames: {num_frames}")
    print()
    print("In CVAT: Actions -> Upload annotations -> 'CVAT for video 1.1'.")
    print("Each box is independent (no track IDs). Use the track tools to")
    print("group boxes of the same fish into a track and assign IDs by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
