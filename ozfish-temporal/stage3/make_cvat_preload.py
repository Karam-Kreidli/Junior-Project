"""
Write the CVAT import files for every clip to be labelled (ranks 001-100).

For each clip, next to the video:
    <rank>__<clip>.xml     CVAT 1.1 annotation with the expert's box(es) on frame 60, label
                           `reference`. Upload it into the clip's task: Actions -> Upload
                           annotations -> "CVAT 1.1". It marks WHICH fish to track; the labeller
                           draws the `fish` track themselves. Nothing is propagated.

In stage3_clips/:
    cvat_labels.json       paste into the CVAT project's label constructor (Raw tab)
    preload_boxes.csv      every pre-loaded box, so exported tracks can be matched back to the
                           expert annotation at frame 60
    <Species>/RARE.txt     marker in each rare-species folder (extra-fish rule applies there)

Frame numbering: CVAT and OpenCV both count from 0. The expert's frame is clip index 60
(start + 60 in the original video), so the box goes on CVAT frame 60.

Which boxes are pre-loaded (agreed 2026-10-04):
    abundant species: exactly one, chosen at random when the expert boxed several
    rare species:     all of them, up to 5, chosen at random beyond that
The random choice is seeded by the clip name, so reruns give the same boxes.
"""
import collections
import csv
import json
import os
import random
from xml.sax.saxutils import quoteattr

STAGE3 = os.path.dirname(os.path.abspath(__file__))       # Junior-Project/ozfish-temporal/stage3
PROJ = os.path.dirname(os.path.dirname(STAGE3))              # Junior-Project
ROOT = os.path.dirname(PROJ)                                 # University/Junior (sibling repos)
DST = os.path.join(PROJ, "data/results/stage3_clips")
META = os.path.join(PROJ, "frame_metadata.csv")
SPECIES = os.path.join(STAGE3, "phase0/species_113.csv")
CENTRE = 60
W, H = 1920, 1080
RARE_CAP = 5

LABELS = [
    {"name": "fish", "color": "#2ec4b6", "type": "rectangle", "attributes": [
        {"name": "source", "input_type": "radio", "mutable": False,
         "values": ["expert", "inferred"], "default_value": "expert"}]},
    {"name": "reference", "color": "#ff3b30", "type": "rectangle", "attributes": []},
]


def load_boxes():
    """(station, absolute frame) -> [(species, x0, y0, x1, y1)] from the OzFish metadata."""
    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(META, newline="")):
        fn = r["file_name"]
        base = fn[:-4] if fn.endswith(".png") else fn
        stem, _, frame = base.rpartition(".")
        if not frame.isdigit():
            continue
        station = stem.split(".")[0]
        boxes[(station, int(frame))].append((
            "{} {}".format(r["genus"], r["species"]),
            float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))
    return boxes


def clamp(x0, y0, x1, y1):
    x0, x1 = sorted((min(max(x0, 0), W), min(max(x1, 0), W)))
    y0, y1 = sorted((min(max(y0, 0), H), min(max(y1, 0), H)))
    return x0, y0, x1, y1


def xml_for(task_name, frame_boxes):
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>",
             "  <version>1.1</version>",
             "  <meta><task><name>%s</name></task></meta>" % task_name.replace("&", "&amp;"),
             '  <image id="%d" name="frame_%06d" width="%d" height="%d">' % (CENTRE, CENTRE, W, H)]
    for x0, y0, x1, y1 in frame_boxes:
        parts.append('    <box label="reference" occluded="0" source="manual" '
                     'xtl="%.2f" ytl="%.2f" xbr="%.2f" ybr="%.2f" z_order="0"></box>'
                     % (x0, y0, x1, y1))
    parts += ["  </image>", "</annotations>", ""]
    return "\n".join(parts)


def main():
    group = {r["species"]: r["group"] for r in csv.DictReader(open(SPECIES, newline=""))}
    boxes = load_boxes()
    rows = [r for r in csv.DictReader(open(os.path.join(DST, "clip_paths.csv"), newline=""))
            if r["stage"] in ("50", "75", "100")]

    out_rows, n_xml, n_box, n_multi_cut, missing = [], 0, 0, 0, []
    for r in rows:
        sp = r["species"]
        start = int(r["clip"].rsplit(".", 2)[-2])
        cand = [b[1:] for b in boxes.get((r["station"], start + CENTRE), []) if b[0] == sp]
        if not cand:
            missing.append(r["path"])
            continue
        cand.sort()                                   # stable order before the seeded shuffle
        random.Random(r["clip"]).shuffle(cand)
        keep = cand[:1] if group[sp] == "abundant" else cand[:RARE_CAP]
        n_multi_cut += len(cand) > len(keep)
        keep = [clamp(*b) for b in keep]

        video = os.path.join(DST, r["path"])
        with open(video[:-4] + ".xml", "w", encoding="utf-8", newline="\n") as f:
            f.write(xml_for(os.path.basename(r["path"])[:-4], keep))
        n_xml += 1
        for i, b in enumerate(keep):
            out_rows.append({"path": r["path"], "species": sp, "group": group[sp],
                             "box_index": i, "frame": CENTRE,
                             "x0": round(b[0], 2), "y0": round(b[1], 2),
                             "x1": round(b[2], 2), "y1": round(b[3], 2),
                             "expert_boxes_at_frame60": len(cand)})
            n_box += 1

    with open(os.path.join(DST, "preload_boxes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with open(os.path.join(DST, "cvat_labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)
    for sp, g in group.items():
        marker = os.path.join(DST, sp.replace(" ", "_"), "RARE.txt")
        if g == "rare":
            with open(marker, "w") as f:
                f.write("Rare species: track every pre-loaded fish (up to 5), and you may add other "
                        "fish you are sure are the same species, with source = inferred. "
                        "See LABELLING_GUIDE.md, Rule 2.\n")
        elif os.path.exists(marker):
            os.remove(marker)

    print("xml files written: %d  (clips to label: %d)" % (n_xml, len(rows)))
    print("reference boxes:   %d   clips where boxes were dropped by the 1 / 5 rule: %d"
          % (n_box, n_multi_cut))
    print("clips with no expert box found at frame 60: %d %s" % (len(missing), missing[:5]))


if __name__ == "__main__":
    main()
