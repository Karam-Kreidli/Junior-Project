"""
Convert a TrackletWriter .npz (output of track_stage2.py) into a
CVAT-for-Video 1.1 XML file you can import as **Tracks** into a CVAT task.

This is the pre-labeling shortcut for issue #17:

    1. Run the live pipeline offline (infer_stage1.py -> track_stage2.py)
       on a Khorfakkan video. You get tracklets.npz with persistent IDs.
    2. Run this script:
            python tracklets_to_cvat.py \\
                --tracklets outputs/tracklets/<video>.npz \\
                --num_frames 14355 \\
                --out tracks_cvat.xml
    3. In CVAT, open the matching task -> Actions -> Upload annotations
       -> "CVAT for video 1.1" -> select tracks_cvat.xml.
    4. Tracks appear as fully-editable annotations. You scrub through and
       correct: delete false-positive tracks (boxes on coral), merge
       split tracks (one fish given two IDs), and *especially* add
       missed fish the tracker never saw (false negatives — the one
       failure mode pre-labeling does not help with).

What the XML contains:
    - One <track> per tracker output, label="fish", source="auto".
    - One <box keyframe="1"> per frame the track was active, with the
      bbox in xtl/ytl/xbr/ybr pixel coordinates.
    - An extra <box outside="1"> keyframe at the frame *after* the
      track's last visible frame, so CVAT does not extrapolate a phantom
      box past the end of the track.
    - A <meta> block declaring the label and the total frame count.

Caveats:
    - Frame numbering: this writes 0-indexed frames (CVAT's convention).
      Tracklet.frame_ids from track_stage2.py are already 0-indexed.
    - Label name: defaults to "fish". MUST match the label name configured
      in the CVAT task, or the import silently drops every track.
    - --num_frames: the total frame count in the source video. Required —
      CVAT's <size> field is mandatory and used to validate box frame
      indices. Pass the exact number of frames in the .mp4 you uploaded
      to the task.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracklets", required=True,
                   help="Path to tracklets.npz produced by track_stage2.py")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Total frame count in the source video. Required for "
                        "a valid CVAT <size>. If omitted, falls back to "
                        "(max-keyframe + 1) which only works if the tracker "
                        "saw the last frame of the video.")
    p.add_argument("--video", default=None,
                   help="Optional: source video path. If provided, "
                        "num_frames is auto-read via OpenCV (overrides "
                        "--num_frames).")
    p.add_argument("--label", default="fish",
                   help="Label name (must match the CVAT task's label). "
                        "Default: 'fish'.")
    p.add_argument("--out", default="tracks_cvat.xml",
                   help="Output XML path. Default: tracks_cvat.xml")
    p.add_argument("--task_name", default="khorfakkan_prelabeled",
                   help="Cosmetic — appears in the <meta>/<task>/<name>. "
                        "Does not affect import.")
    p.add_argument("--min_track_length", type=int, default=1,
                   help="Drop tracks shorter than this many frames. "
                        "Useful for filtering tracker noise. Default: 1.")
    p.add_argument("--interp_gap", type=int, default=10,
                   help="Max detection-dropout gap (frames) to let CVAT "
                        "interpolate across instead of marking the fish "
                        "outside. Short gaps (<= this) become smooth "
                        "interpolated boxes (removes flicker); longer gaps "
                        "get an outside=1 terminator (probable real exit). "
                        "Default: 10. Set 0 for the old always-terminate "
                        "behavior.")
    return p.parse_args()


def num_frames_from_video(path: str) -> int:
    """Read total frame count from a video file via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f"video reported {n} frames; verify {path} is readable")
    return n


def load_tracklets(npz_path: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Read the TrackletWriter .npz and return [(track_id, frame_ids, bboxes), ...].

    `frame_ids` is (T,) int, `bboxes` is (T, 4) float in [x, y, w, h] (the
    Tracklet convention used throughout the pipeline).
    """
    if not os.path.exists(npz_path):
        raise SystemExit(f"tracklets file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    track_ids = data["track_ids"]
    frame_ids_list = data["frame_ids"]
    bboxes_list = data["bboxes"]

    out = []
    for tid, fids, bxs in zip(track_ids, frame_ids_list, bboxes_list):
        out.append((int(tid), np.asarray(fids, dtype=int),
                    np.asarray(bxs, dtype=float)))
    return out


def build_xml(
    tracklets: List[Tuple[int, np.ndarray, np.ndarray]],
    num_frames: int,
    label_name: str,
    task_name: str,
    interp_gap: int = 10,
) -> ET.ElementTree:
    """
    Build the CVAT-for-Video 1.1 ElementTree.

    Track bbox format: Tracklet stores [x, y, w, h] (top-left + size).
    CVAT XML expects xtl, ytl, xbr, ybr (top-left + bottom-right).
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- meta -----------------------------------------------------------
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    # 'id' is informational; CVAT assigns the real one on import.
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(num_frames)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "5"
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
    ET.SubElement(label, "attributes")  # empty — single-class, no attrs

    ET.SubElement(meta, "dumped").text = datetime.utcnow().isoformat()

    # --- tracks ---------------------------------------------------------
    for tid, fids, boxes_xywh in tracklets:
        track_el = ET.SubElement(root, "track", {
            "id": str(int(tid)),
            "label": label_name,
            "source": "auto",
        })

        # Sort frames just in case (Tracklet is supposed to be chronological,
        # but defensive — out-of-order frames would silently corrupt CVAT).
        order = np.argsort(fids)
        fids_sorted = fids[order]
        boxes_sorted = boxes_xywh[order]

        prev_fid = None
        last_box = None  # remember last visible box for terminator coords
        for fid, (x, y, w, h) in zip(fids_sorted.tolist(), boxes_sorted.tolist()):
            fid = int(fid)
            xtl, ytl, xbr, ybr = x, y, x + w, y + h

            # Handle gaps (frames the detector dropped this fish; the track
            # survived with the same ID but recorded no box). Two regimes:
            #
            #   short gap (<= interp_gap): almost certainly a brief detection
            #     dropout, not a real exit. Do NOT emit outside=1 — leave the
            #     hole so CVAT linearly interpolates the box between this real
            #     keyframe and the previous one. This removes the box "flicker"
            #     a brief miss would otherwise cause. The human verifies/nudges
            #     the interpolated boxes (and deletes them if the fish really
            #     did leave for those few frames).
            #
            #   long gap (> interp_gap): probably a real exit/occlusion. Keep
            #     the original behavior — emit an outside=1 terminator at
            #     prev+1 so CVAT does NOT draw a (likely wrong) interpolated
            #     box marching across the whole gap.
            if prev_fid is not None and fid > prev_fid + 1:
                gap_len = fid - prev_fid - 1
                if gap_len > interp_gap:
                    gap_end = min(prev_fid + 1, num_frames - 1)
                    px, py, pw, ph = last_box
                    ET.SubElement(track_el, "box", {
                        "frame": str(gap_end),
                        "outside": "1",
                        "occluded": "0",
                        "keyframe": "1",
                        "xtl": f"{px:.2f}",
                        "ytl": f"{py:.2f}",
                        "xbr": f"{(px + pw):.2f}",
                        "ybr": f"{(py + ph):.2f}",
                        "z_order": "0",
                    })

            ET.SubElement(track_el, "box", {
                "frame": str(fid),
                "outside": "0",
                "occluded": "0",
                "keyframe": "1",
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}",
                "z_order": "0",
            })

            prev_fid = fid
            last_box = (x, y, w, h)

        # Final terminator one frame past the last visible frame, so CVAT
        # does not extrapolate a phantom box forward to the end of the video.
        if prev_fid is not None:
            end_frame = min(prev_fid + 1, num_frames - 1)
            if end_frame > prev_fid:
                px, py, pw, ph = last_box
                ET.SubElement(track_el, "box", {
                    "frame": str(end_frame),
                    "outside": "1",
                    "occluded": "0",
                    "keyframe": "1",
                    "xtl": f"{px:.2f}",
                    "ytl": f"{py:.2f}",
                    "xbr": f"{(px + pw):.2f}",
                    "ybr": f"{(py + ph):.2f}",
                    "z_order": "0",
                })

    return ET.ElementTree(root)


def main() -> int:
    args = parse_args()

    # Resolve num_frames
    if args.video:
        num_frames = num_frames_from_video(args.video)
        print(f"video frame count from {args.video}: {num_frames}")
    elif args.num_frames is not None:
        num_frames = args.num_frames
    else:
        num_frames = None  # may infer below

    tracklets = load_tracklets(args.tracklets)
    if args.min_track_length > 1:
        before = len(tracklets)
        tracklets = [t for t in tracklets if len(t[1]) >= args.min_track_length]
        print(f"min_track_length={args.min_track_length}: kept "
              f"{len(tracklets)}/{before} tracks")

    if num_frames is None:
        if not tracklets:
            raise SystemExit("--num_frames required (no tracklets to infer from)")
        num_frames = max(int(fids.max()) for _, fids, _ in tracklets) + 1
        print(f"--num_frames not given; inferring from tracks: {num_frames} "
              f"(only correct if tracks span to the last frame of the video)")

    if not tracklets:
        print("warning: no tracks to write", file=sys.stderr)

    tree = build_xml(tracklets, num_frames, args.label, args.task_name,
                     interp_gap=args.interp_gap)

    # Pretty-print
    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    with open(args.out, "wb") as f:
        f.write(pretty)

    n_boxes = sum(len(fids) for _, fids, _ in tracklets)
    print(f"wrote {len(tracklets)} tracks ({n_boxes} keyframes) -> {args.out}")
    print(f"  label: '{args.label}'  num_frames: {num_frames}")
    print()
    print("Next: in CVAT, open the task, Actions -> Upload annotations")
    print("      -> 'CVAT for video 1.1' -> select this XML.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
