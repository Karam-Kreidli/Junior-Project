"""
Stage 3, Phase 0: freeze the species list, the per-species labelling order and the CV folds.

Writes to stage3/phase0/:
    species_113.csv   one row per species: taxonomy, group, clip and deployment counts,
                      how many clips each labelling cap takes, how many are test-eligible
    clips_113.csv     one row per usable clip of those species: deployment, D6a split,
                      labelling rank and stage, CV fold

Two rules that shape the folds:

1. Group by DEPLOYMENT, not camera folder. Most drops have a left and a right camera
   (A000001_L / A000001_R) filming the same fish at the same moment. Splitting by folder
   would put one view in train and the other in test.

2. Only clips from deployments D6a did NOT train on can be tested. The labelled centre
   frame of every clip is an OzFish still, so for D6a's train deployments it was very
   likely a D6a training image. Those clips are training-only (fold = -1). The clips from
   D6a's val/test deployments are spread over 5 folds with StratifiedGroupKFold.
   Deployed D6a is seed 1, so its split is ozfish_split_seed1.csv.

Labelling order per species (fixed seed): test-eligible clips first, then the rest; inside
each part, deployments are interleaved round-robin so no single drop fills a quota. The
first 50 are labelled at cap 50, the next 25 at cap 75, the next 25 at cap 100.

    python build_phase0.py
"""
import collections
import csv
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

STAGE3 = os.path.dirname(os.path.abspath(__file__))       # Junior-Project/ozfish-temporal/stage3
PROJ = os.path.dirname(os.path.dirname(STAGE3))              # Junior-Project
ROOT = os.path.dirname(PROJ)                                 # University/Junior (sibling repos)
CLIP_INDEX = os.path.join(PROJ, "ozfish-temporal/manifests/clip_index.csv")
META = os.path.join(PROJ, "frame_metadata.csv")
D6A_SPLIT = os.path.join(ROOT, "ozfish-finegrained-staging/splits/ozfish_split_seed1.csv")
OUT = os.path.join(STAGE3, "phase0")

MIN_CLIPS = 20
ABUNDANT = 50                      # > 50 clips = abundant
CAPS = (50, 75, 100)
N_FOLDS = 5
SEED = 0
PLACEHOLDERS = {"Nemipterus spp", "Scomberomorus spp", "Lethrinus spp",
                "Plectropomus spp", "Scarus sp3", "sp10"}


def deployment(station):
    """A000001_L -> A000001. Folders without a camera suffix are their own deployment."""
    head, _, tail = station.rpartition("_")
    return head if tail in ("L", "R") and head else station


def interleave(clips, rng):
    """Round-robin across deployments, each deployment's clips in random order."""
    by_dep = collections.defaultdict(list)
    for c in clips:
        by_dep[c["deployment"]].append(c)
    deps = list(by_dep)
    rng.shuffle(deps)
    for d in deps:
        rng.shuffle(by_dep[d])
    out = []
    while any(by_dep[d] for d in deps):
        for d in deps:
            if by_dep[d]:
                out.append(by_dep[d].pop())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)

    d6a = {}
    for r in csv.DictReader(open(D6A_SPLIT, newline="")):
        d6a[r["deployment"]] = r["split"]

    taxo = {}
    for r in csv.DictReader(open(META, newline="")):
        taxo.setdefault("{} {}".format(r["genus"], r["species"]), (r["genus"], r["family"]))

    clips = []
    for r in csv.DictReader(open(CLIP_INDEX, newline="")):
        if r["ok"] != "1":
            continue
        dep = deployment(r["station"])
        clips.append({
            "species": r["species"], "station": r["station"], "deployment": dep,
            "clip": r["clip"], "start": int(r["start"]),
            "n_label_frames": int(r["n_label_frames"]),
            "d6a_split": d6a.get(dep, "absent"),
        })

    n_clips = collections.Counter(c["species"] for c in clips)
    keep = sorted(s for s, n in n_clips.items() if n >= MIN_CLIPS and s not in PLACEHOLDERS)
    keep_set = set(keep)
    clips = [c for c in clips if c["species"] in keep_set]
    for c in clips:
        # D6a never trained on val/test deployments (val picked its checkpoint, so it is
        # not pristine, but none of its frames were trained on).
        c["test_eligible"] = int(c["d6a_split"] in ("val", "test"))

    # ---- labelling order and stage -------------------------------------------------
    by_sp = collections.defaultdict(list)
    for c in clips:
        by_sp[c["species"]].append(c)
    for sp in keep:
        rng = random.Random("{}|{}".format(SEED, sp))
        held = [c for c in by_sp[sp] if c["test_eligible"]]
        rest = [c for c in by_sp[sp] if not c["test_eligible"]]
        order = interleave(held, rng) + interleave(rest, rng)
        for i, c in enumerate(order, 1):
            c["label_rank"] = i
            c["stage"] = next((cap for cap in CAPS if i <= cap), 0)

    # ---- folds: test-eligible clips only, grouped by deployment ---------------------
    elig = [c for c in clips if c["test_eligible"]]
    y = np.array([keep.index(c["species"]) for c in elig])
    g = np.array([c["deployment"] for c in elig])
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(elig)), y, g)):
        for i in te:
            elig[i]["fold"] = k
    for c in clips:
        c.setdefault("fold", -1)       # -1 = always training

    # sanity: no deployment in two folds, no deployment both test-eligible and train-only
    dep_folds = collections.defaultdict(set)
    for c in clips:
        dep_folds[c["deployment"]].add(c["fold"])
    bad = [d for d, f in dep_folds.items() if len(f) > 1]
    assert not bad, "deployments split across folds: {}".format(bad[:5])

    # ---- write ---------------------------------------------------------------------
    clips.sort(key=lambda c: (c["species"], c["label_rank"]))
    cols = ["species", "label_rank", "stage", "fold", "test_eligible", "d6a_split",
            "deployment", "station", "clip", "start", "n_label_frames"]
    with open(os.path.join(OUT, "clips_113.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(clips)

    rows = []
    for sp in keep:
        cs = by_sp[sp]
        n = len(cs)
        lab50 = [c for c in cs if c["stage"] == 50]
        genus, family = taxo.get(sp, (sp.split(" ")[0], ""))
        rows.append({
            "species": sp, "genus": genus, "family": family,
            "group": "abundant" if n > ABUNDANT else "rare",
            "clips": n,
            "deployments": len({c["deployment"] for c in cs}),
            "test_eligible_clips": sum(c["test_eligible"] for c in cs),
            "label_cap50": min(n, 50), "label_cap75": min(n, 75), "label_cap100": min(n, 100),
            "test_eligible_in_cap50": sum(c["test_eligible"] for c in lab50),
            "folds_with_test": len({c["fold"] for c in cs if c["fold"] >= 0}),
        })
    rows.sort(key=lambda r: -r["clips"])
    with open(os.path.join(OUT, "species_113.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary -------------------------------------------------------------------
    ab = [r for r in rows if r["group"] == "abundant"]
    ra = [r for r in rows if r["group"] == "rare"]
    print("species {}  (abundant {}, rare {})".format(len(rows), len(ab), len(ra)))
    print("clips {}  deployments {}".format(len(clips), len({c['deployment'] for c in clips})))
    for cap in CAPS:
        print("  cap {:>3}: label {:>5} clips".format(cap, sum(min(r['clips'], cap) for r in rows)))
    te = [r["test_eligible_clips"] for r in rows]
    print("test-eligible clips {} (D6a val/test deployments); per species min {} median {}".format(
        sum(te), min(te), int(np.median(te))))
    print("species with < 5 test-eligible clips: {}".format(sum(t < 5 for t in te)))
    fc = collections.Counter(c["fold"] for c in clips)
    print("clips per fold: " + "  ".join("{}:{}".format(k, fc[k]) for k in sorted(fc)))
    print("wrote {}".format(OUT))


if __name__ == "__main__":
    main()
