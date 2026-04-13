"""Pick the best video for tracker sanity-testing and stage its frames."""
import argparse
import os
import shutil
from collections import defaultdict

import pandas as pd


TOP_SPECIES = {
    "affinis", "bleekeri", "choirocephalus", "corallicola", "flagellifer",
    "hedlandensis", "hispidus", "lunaris", "macarellus", "macrorhinus",
    "milii", "pallimaculatus", "punctatissimus", "xanthochilus",
    "qenie", "monostigma", "nigrofasciata", "multistriatus", "virescens",
    "multinotatus",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_oz/metadata/frame_metadata.csv")
    ap.add_argument("--frames_root", action="append", default=None,
                    help="Directory containing .png frames (repeatable). "
                         "If omitted, uses the same dirs as train_detection.py.")
    ap.add_argument("--out_dir", default="test_vid")
    ap.add_argument("--min_frames", type=int, default=50)
    args = ap.parse_args()

    frame_roots = args.frames_root or [
        "/media/openuae/UUI/frames_waternet",
        "data_oz/frames_waternet_1",
        "data_oz/frames_waternet_2",
        "/media/openuae/UUI/frames_waternet_3",
    ]

    df = pd.read_csv(args.csv)
    df["video_id"] = df["file_name"].str.split(".avi").str[0] + ".avi"
    df["frame_num"] = df["file_name"].str.split(".avi.").str[1].str.replace(".png", "").astype(int)

    grouped = df.groupby("video_id").agg(
        n_frames=("file_name", "nunique"),
        top_hits=("species", lambda s: sum(sp in TOP_SPECIES for sp in s)),
    ).reset_index()
    grouped = grouped[grouped["n_frames"] >= args.min_frames]
    grouped = grouped.sort_values(["top_hits", "n_frames"], ascending=False)

    print(grouped.head(10).to_string(index=False))
    best = grouped.iloc[0]["video_id"]
    print(f"\nPicked: {best}")

    frames = sorted(df[df["video_id"] == best]["file_name"].unique(),
                    key=lambda f: int(f.split(".avi.")[1].replace(".png", "")))

    os.makedirs(args.out_dir, exist_ok=True)
    copied = 0
    missing = 0
    for fname in frames:
        src = None
        for root in frame_roots:
            candidate = os.path.join(root, fname)
            if os.path.exists(candidate):
                src = candidate
                break
        if src is None:
            missing += 1
            continue
        shutil.copy(src, os.path.join(args.out_dir, fname))
        copied += 1
    print(f"Copied {copied}/{len(frames)} frames to {args.out_dir} "
          f"({missing} missing across roots)")


if __name__ == "__main__":
    main()
