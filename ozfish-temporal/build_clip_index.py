"""Build clip_index.csv: one row per OzFish temporal clip, joined to frame_metadata.csv.

Clips are 120-frame MJPEG windows named <station>.<ext>.<start>.mp4, with the
labelled frame at start+60. Verified: 10313/10313 clips match that rule exactly.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)                     # ozfish-temporal lives inside Junior-Project
META = os.path.join(PROJ, "frame_metadata.csv")
ROOT = os.path.join(PROJ, "data", "results")
BAD = os.path.join(HERE, "manifests", "bad_clips.txt")
OUT = os.path.join(HERE, "manifests", "clip_index.csv")

STATION = re.compile(r"([ABEG]\d{6}_[LR])\.(?:avi|AVI|MP4|mp4|mpeg|MPEG)\.(\d+)\.")
CLIP = re.compile(r".*\.(\d+)\.mp4$")
WINDOW = 120
CENTER = 60


def load_meta():
    rows = {}
    with open(META, newline="") as f:
        for r in csv.DictReader(f):
            m = STATION.match(r["file_name"])
            if m:
                rows.setdefault(m.group(1), []).append((int(m.group(2)), r))
    for k in rows:
        rows[k].sort(key=lambda t: t[0])
    return rows


def load_bad():
    if not os.path.exists(BAD):
        return set()
    with open(BAD) as f:
        return {l.strip().replace(os.sep, "/") for l in f if l.strip()}


def main():
    meta = load_meta()
    bad = load_bad()
    n = ok = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "clip", "start", "center", "n_label_frames",
                    "label_offsets", "n_boxes", "species", "ok"])
        for d in sorted(os.listdir(ROOT)):
            p = os.path.join(ROOT, d)
            if not os.path.isdir(p):
                continue
            rows = meta.get(d, [])
            for fn in sorted(os.listdir(p)):
                m = CLIP.match(fn)
                if not m:
                    continue
                s = int(m.group(1))
                inw = [(fr, rr) for fr, rr in rows if s <= fr < s + WINDOW]
                offs = sorted({fr - s for fr, _ in inw})
                sp = sorted({(rr["genus"] + " " + rr["species"]).strip() for _, rr in inw})
                good = 0 if (d + "/" + fn) in bad else 1
                ok += good
                n += 1
                w.writerow([d, fn, s, s + CENTER, len(offs),
                            ";".join(map(str, offs)), len(inw), ";".join(sp), good])
    print("wrote %s" % OUT)
    print("clips %d  usable %d  excluded %d" % (n, ok, n - ok))


if __name__ == "__main__":
    sys.exit(main())
