"""
Arrange the Stage 3 clips one folder per species, inside Junior-Project/data/results/stage3_clips/:

    stage3_clips/<Genus_species>/<rank>__<original clip name>.mp4

<rank> is the clip's labelling rank within its species (phase0/clips_113.csv, label_rank), zero
padded so a name sort gives the labelling order:
    001-050  round 1 (cap 50)
    051-100  rounds 2 and 3 (caps 75, 100), only if the Phase 2 / Phase 4 gates say keep going
    101+     baseline only (testable clips beyond cap 100), never labelled
The original clip name is kept after the "__", so the station and start frame stay readable.

Moves only (same drive, a rename). Finds each clip wherever an earlier run left it: the original
data/results/<station>/ path, the old group folders, or already in place. Writes
stage3_clips/clip_paths.csv mapping every file to its station, clip, rank, stage and fold.
"""
import csv
import os
import shutil

STAGE3 = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(STAGE3)), "data", "results")
DST = os.path.join(SRC, "stage3_clips")
CLIPS = os.path.join(STAGE3, "phase0", "clips_113.csv")
OLD_GROUPS = ("1_label_now", "2_label_later", "3_no_labelling")


def needed(r):
    return r["stage"] != "0" or r["test_eligible"] == "1"


def folder(species):
    return species.replace(" ", "_")


rows = [r for r in csv.DictReader(open(CLIPS, newline="")) if needed(r)]
moved = already = 0
missing = []
index = []
for r in rows:
    rel = r["station"] + "/" + r["clip"]
    name = "%03d__%s" % (int(r["label_rank"]), r["clip"])
    dst = os.path.join(DST, folder(r["species"]), name)
    if os.path.exists(dst):
        already += 1
    else:
        candidates = [os.path.join(DST, g, rel) for g in OLD_GROUPS] + [os.path.join(SRC, rel)]
        src = next((c for c in candidates if os.path.exists(c)), None)
        if src is None:
            missing.append(rel)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        moved += 1
    index.append({
        "path": folder(r["species"]) + "/" + name, "species": r["species"],
        "label_rank": r["label_rank"], "stage": r["stage"], "fold": r["fold"],
        "test_eligible": r["test_eligible"], "station": r["station"], "clip": r["clip"],
    })

# remove the emptied old group folders and their file lists
for g in OLD_GROUPS:
    p = os.path.join(DST, g)
    if os.path.isdir(p):
        for root, dirs, files in os.walk(p, topdown=False):
            if not files:
                for d in dirs:
                    dp = os.path.join(root, d)
                    if os.path.isdir(dp) and not os.listdir(dp):
                        os.rmdir(dp)
        if not os.listdir(p):
            os.rmdir(p)
    lst = os.path.join(DST, g + ".txt")
    if os.path.exists(lst):
        os.remove(lst)

with open(os.path.join(DST, "clip_paths.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(index[0]))
    w.writeheader()
    w.writerows(index)
for e in (os.path.join(STAGE3, "LABELLING_GUIDE.md"), CLIPS):
    shutil.copy2(e, DST)

species_dirs = [d for d in os.listdir(DST) if os.path.isdir(os.path.join(DST, d))]
in_place = sum(os.path.exists(os.path.join(DST, x["path"])) for x in index)
print("moved %d, already in place %d, missing %d" % (moved, already, len(missing)))
print("species folders: %d   files in place: %d/%d" % (len(species_dirs), in_place, len(rows)))
print("round 1 (rank <= 50): %d   rounds 2-3: %d   baseline only: %d" % (
    sum(x["stage"] == "50" for x in index),
    sum(x["stage"] in ("75", "100") for x in index),
    sum(x["stage"] == "0" for x in index)))
if missing:
    print("MISSING:", missing[:10])
