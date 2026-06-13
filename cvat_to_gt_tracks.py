"""
Convert a CVAT-for-Video 1.1 export (with HUMAN-assigned track IDs) into the
ground-truth track JSON the HOTAEvaluator / track_stage2.py --gt_tracks wants.

This closes the HOTA loop for issue #4 / #17:

    1. Pre-label boxes-only (detections_to_cvat.py) — detector boxes, no IDs.
    2. In CVAT, YOU draw the persistent track IDs (group boxes into tracks).
       The IDs must be human-assigned, detector/tracker-uninfluenced, or the
       HOTA eval is circular (you'd be scoring the tracker against its own
       cleaned-up output — see problems.md #17 test-split rule).
    3. Export from CVAT as "CVAT for Video 1.1".
    4. Run THIS to turn that export into gt_tracks.json.
    5. HOTA = raw (UNEDITED) tracker .npz  vs  this GT:
         python track_stage2.py --no_frames \
             --detections outputs/detections/clip01_mp4.npz \
             --gt_tracks gt_tracks.json

GT JSON format (consumed by track_stage2.load_gt_tracks / HOTAEvaluator):
    { "<video_id>": { "<frame_id>": [ {"track_id": int,
                                       "bbox": [x, y, w, h]}, ... ] } }
    bbox is [x, y, w, h] (top-left + size). CVAT stores xtl/ytl/xbr/ybr, so
    we convert. outside="1" boxes (track terminators / gaps) are skipped —
    they mark frames where the object is NOT present.

Usage:
    python cvat_to_gt_tracks.py \
        --cvat clip01_gt_from_cvat.xml \
        --video_id clip01_mp4 \
        --out gt_tracks.json

    # --video_id MUST match the prediction's video_id so HOTA pairs them.
    # The tracker's .npz for clip01.mp4 is saved as clip01_mp4.npz, whose
    # internal sequence id is "clip01_mp4" — use that.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cvat", required=True,
                   help="CVAT-for-Video 1.1 XML exported from the GT task.")
    p.add_argument("--video_id", required=True,
                   help="Sequence id to key the GT under. MUST match the "
                        "prediction's video_id (e.g. 'clip01_mp4').")
    p.add_argument("--out", default="gt_tracks.json",
                   help="Output GT JSON path. If it already contains other "
                        "video_ids, this one is merged in (multi-clip GT).")
    p.add_argument("--label", default=None,
                   help="If set, only import tracks with this label "
                        "(default: import all labels).")
    p.add_argument("--include_outside", action="store_true",
                   help="Also emit boxes flagged outside='1' (default: skip "
                        "them — they mark frames where the object is absent).")
    return p.parse_args()


def parse_cvat_tracks(xml_path: str, label_filter, include_outside):
    """Parse CVAT-for-Video XML -> {frame_id: [{track_id, bbox xywh}, ...]}."""
    if not os.path.exists(xml_path):
        raise SystemExit(f"CVAT XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    per_frame = defaultdict(list)
    n_tracks = 0
    n_boxes = 0
    n_skipped_outside = 0
    seen_ids = set()

    for track_el in root.findall("track"):
        label = track_el.get("label")
        if label_filter is not None and label != label_filter:
            continue
        tid = int(track_el.get("id"))
        n_tracks += 1
        seen_ids.add(tid)

        for box in track_el.findall("box"):
            outside = box.get("outside", "0") == "1"
            if outside and not include_outside:
                n_skipped_outside += 1
                continue
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            w = xbr - xtl
            h = ybr - ytl
            per_frame[frame].append({
                "track_id": tid,
                "bbox": [round(xtl, 2), round(ytl, 2),
                         round(w, 2), round(h, 2)],
            })
            n_boxes += 1

    # CVAT can also store per-frame shapes outside <track> (image-mode). Warn
    # if there are <image> shapes — those have no track_id and can't be GT.
    n_imageshapes = sum(len(img.findall("box")) for img in root.findall("image"))
    if n_imageshapes:
        print(f"  WARNING: {n_imageshapes} box(es) found under <image> "
              f"(track-less shapes). These have NO track_id and are SKIPPED "
              f"— HOTA needs track IDs. Make sure you grouped boxes into "
              f"tracks in CVAT, not left them as individual shapes.",
              file=sys.stderr)

    return per_frame, n_tracks, n_boxes, n_skipped_outside, len(seen_ids)


def main() -> int:
    args = parse_args()

    per_frame, n_tracks, n_boxes, n_skip, n_ids = parse_cvat_tracks(
        args.cvat, args.label, args.include_outside,
    )

    if n_tracks == 0:
        print("ERROR: no <track> elements found in the CVAT XML. Did you "
              "export as 'CVAT for Video 1.1' AND group boxes into tracks "
              "(not leave them as individual image shapes)?", file=sys.stderr)
        return 1

    # Build this video's GT block: {frame_id(str): [ {track_id, bbox} ]}
    video_block = {str(f): items for f, items in sorted(per_frame.items())}

    # Merge into existing GT file if present (supports multi-clip GT).
    gt = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                gt = json.load(f)
            if not isinstance(gt, dict):
                gt = {}
        except (json.JSONDecodeError, OSError):
            gt = {}
    if args.video_id in gt:
        print(f"  note: overwriting existing GT for video_id "
              f"'{args.video_id}' in {args.out}")
    gt[args.video_id] = video_block

    with open(args.out, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"wrote GT for '{args.video_id}': {n_ids} track IDs, {n_boxes} "
          f"boxes across {len(video_block)} frames -> {args.out}")
    if n_skip:
        print(f"  ({n_skip} outside=1 terminator boxes skipped)")
    print()
    print("Next — score the UNEDITED tracker output against this GT:")
    print(f"  python track_stage2.py --no_frames \\")
    print(f"      --detections outputs/detections/{args.video_id}.npz \\")
    print(f"      --gt_tracks {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
