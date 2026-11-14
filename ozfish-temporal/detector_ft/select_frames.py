"""Pick frames for the detector fine-tune (#30): one per camera drop, each guaranteed to hold a fish.

Every OzFish clip is cut around frames an AIMS expert annotated (clip_index.csv `label_offsets`,
always including 60), so picking one of THOSE frames guarantees at least one fish, with no
detector needed. The detector/GDINO count comes later in prelabel_frames.py.

Rules:
  * drop = station without _L/_R (the stereo pair is one drop: same fish, same scene)
  * drops used by the tracker benchmark (stage3/labels/tracks.csv) are excluded entirely,
    so the post-fine-tune tracker re-score stays clean
  * whole drops go to train OR test (~15% test), never both
  * up to --per_drop clips per drop, one expert frame per clip, spread over all drops

    python select_frames.py --n_train 1000 --n_test 200      # on the VM (clips live there)
Writes frames/<split>/<clip>_f<NNN>.jpg and frames/manifest.csv.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INDEX = HERE.parent / "manifests" / "clip_index.csv"
BAD = HERE.parent / "manifests" / "bad_clips.txt"
TRACKS = HERE.parent / "stage3" / "labels" / "tracks.csv"
CLIPS = REPO / "data" / "results"


def read_frame(path, idx):
    """Decode sequentially to frame idx (MJPEG, 120 frames: cheap and exact, no seek drift)."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    for _ in range(idx):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--test_frac", type=float, default=0.15, help="share of drops held out")
    ap.add_argument("--per_drop", type=int, default=2, help="max clips per drop")
    ap.add_argument("--clips", default=str(CLIPS), help="folder holding the clips (searched recursively)")
    ap.add_argument("--out", default=str(HERE / "frames"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    bad = {Path(line.strip()).name for line in BAD.read_text().splitlines() if line.strip()}
    bench = {r["path"].split("/")[-1].split("__", 1)[1] for r in csv.DictReader(open(TRACKS))}
    rows = [r for r in csv.DictReader(open(INDEX)) if r["ok"] == "1" and r["clip"] not in bad]
    drop = lambda r: r["station"].rsplit("_", 1)[0]
    bench_drops = {drop(r) for r in rows if r["clip"] in bench}

    by_drop = {}
    for r in rows:
        if drop(r) not in bench_drops:
            by_drop.setdefault(drop(r), []).append(r)
    drops = sorted(by_drop)
    rng.shuffle(drops)
    n_test = round(len(drops) * args.test_frac)
    split_of = {d: ("test" if i < n_test else "train") for i, d in enumerate(drops)}
    print(f"{len(rows)} clips, {len(bench_drops)} benchmark drops excluded, "
          f"{len(drops)} drops left ({n_test} test / {len(drops) - n_test} train)")

    # Round-robin over drops: every drop gives its 1st clip before any gives a 2nd.
    queues = {d: rng.sample(cs, len(cs))[:args.per_drop] for d, cs in by_drop.items()}
    order = [q[i] for i in range(args.per_drop) for d in drops for q in [queues[d]] if i < len(q)]

    # stage3 clips were moved to stage3_clips/<Species>/<rank>__<clip>: find each clip wherever it is
    where = {f.name.split("__", 1)[-1]: f for f in Path(args.clips).rglob("*.mp4")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = {"train": args.n_train, "test": args.n_test}
    got = {"train": 0, "test": 0}
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "drop", "station", "clip", "frame", "expert_species"])
        for r in order:
            s = split_of[drop(r)]
            if got[s] >= want[s]:
                continue
            frame = int(rng.choice(r["label_offsets"].split(";")))
            img = read_frame(where[r["clip"]], frame) if r["clip"] in where else None
            if img is None:
                print(f"  unreadable: {r['station']}/{r['clip']} frame {frame}")
                continue
            name = f"{Path(r['clip']).stem}_f{frame:03d}.jpg"
            (out / s).mkdir(exist_ok=True)
            cv2.imwrite(str(out / s / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.writerow([s, name, drop(r), r["station"], r["clip"], frame, r["species"]])
            got[s] += 1
            if got == want:
                break
    print(f"wrote {got['train']} train + {got['test']} test frames -> {out}")
    if got != want:
        print(f"  short of target {want}: raise --per_drop")


if __name__ == "__main__":
    main()
