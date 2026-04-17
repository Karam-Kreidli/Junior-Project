"""
BioReef.ai — Best 2-Folder Subset Selector
=============================================
Evaluates multiple min-sample thresholds in a single run, writing all
results into one report so you can compare before committing.

For each threshold it:
  1. Picks the best 2-of-3 folder combination
  2. Lists kept/dropped species and distribution stats
  3. Produces a file transfer plan (archive dropped-species frames,
     import all donor frames)

Usage:
    python check_ssd_species.py
    python check_ssd_species.py --thresholds 10 20 30 50
"""

import argparse
import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

ALL_DIRS = [
    "data_oz/frames_waternet_1",
    "data_oz/frames_waternet_2",
    "/media/openuae/UUI/frames_waternet_3",
]


def scan_dir(d):
    if not os.path.isdir(d):
        return set()
    return {f for f in os.listdir(d) if f.endswith(".png")}


def w(out, text):
    print(text)
    out.write(text + "\n")


def evaluate_threshold(out, min_samples, combos, frame_annotations, frame_families,
                       dir_files, img_dirs, archive_dir):
    """Evaluate and report for a single threshold. Appends to `out`."""

    w(out, "")
    w(out, "#" * 75)
    w(out, f"#  THRESHOLD: min_samples = {min_samples}")
    w(out, "#" * 75)

    # ------------------------------------------------------------------
    # 1. Score each combo at this threshold
    # ------------------------------------------------------------------
    scored = []
    for combo in combos:
        sp_counts = combo["sp_counts"]
        kept = {sp: c for sp, c in sp_counts.items() if c >= min_samples}
        dropped = {sp: c for sp, c in sp_counts.items() if c < min_samples}
        kept_ann = sum(kept.values())
        dropped_ann = sum(dropped.values())

        scored.append({
            **combo,
            "kept": kept,
            "dropped": dropped,
            "kept_species": len(kept),
            "dropped_species": len(dropped),
            "kept_annotations": kept_ann,
            "dropped_annotations": dropped_ann,
        })

    # ------------------------------------------------------------------
    # 2. Comparison table
    # ------------------------------------------------------------------
    w(out, f"\n{'=' * 75}")
    w(out, f"Comparison (min_samples={min_samples})")
    w(out, f"{'=' * 75}")
    w(out, f"  {'Combo':<45s} {'Frames':>7s} {'KeptSpp':>8s} {'DrpSpp':>7s} {'KeptAnn':>9s} {'DrpAnn':>8s}")
    w(out, "  " + "-" * 86)

    for c in scored:
        w(out, f"  {c['label']:<45s} {c['total_frames']:>7,} "
               f"{c['kept_species']:>8} {c['dropped_species']:>7} "
               f"{c['kept_annotations']:>9,} {c['dropped_annotations']:>8,}")

    # ------------------------------------------------------------------
    # 3. Winner
    # ------------------------------------------------------------------
    winner = max(scored, key=lambda x: (x["kept_annotations"], x["kept_species"]))

    w(out, f"\n>>> WINNER: {winner['label']}")
    w(out, f"    Kept species:      {winner['kept_species']}")
    w(out, f"    Kept annotations:  {winner['kept_annotations']:,}")
    w(out, f"    Dropped species:   {winner['dropped_species']}")
    w(out, f"    Dropped annot:     {winner['dropped_annotations']:,}")
    w(out, f"    Total frames:      {winner['total_frames']:,}")

    # ------------------------------------------------------------------
    # 4. Kept species table
    # ------------------------------------------------------------------
    sp_family = {}
    for f in winner["files"]:
        fam_map = frame_families.get(f, {})
        for sp, fam in fam_map.items():
            if sp not in sp_family:
                sp_family[sp] = fam

    sorted_kept = sorted(winner["kept"].items(), key=lambda x: -x[1])

    w(out, f"\n{'=' * 75}")
    w(out, f"Kept Species ({winner['kept_species']} species, {winner['kept_annotations']:,} annotations)")
    w(out, f"{'=' * 75}")
    w(out, f"  {'#':<4s} {'Species':<35s} {'Family':<20s} {'Count':>6s}")
    w(out, "  " + "-" * 67)

    for i, (sp, count) in enumerate(sorted_kept, 1):
        fam = sp_family.get(sp, "?")
        w(out, f"  {i:<4d} {sp:<35s} {fam:<20s} {count:>6d}")

    w(out, "  " + "-" * 67)
    w(out, f"  {'':4s} {'TOTAL':<35s} {'':20s} {winner['kept_annotations']:>6,}")

    # ------------------------------------------------------------------
    # 5. Dropped species table
    # ------------------------------------------------------------------
    sorted_dropped = sorted(winner["dropped"].items(), key=lambda x: -x[1])

    w(out, f"\n{'=' * 75}")
    w(out, f"Dropped Species ({winner['dropped_species']} species, {winner['dropped_annotations']:,} annotations)")
    w(out, f"{'=' * 75}")

    for sp, count in sorted_dropped:
        fam = sp_family.get(sp, "?")
        w(out, f"  {sp:<35s} {fam:<20s} {count:>4d}")

    # ------------------------------------------------------------------
    # 6. Distribution stats
    # ------------------------------------------------------------------
    if sorted_kept:
        counts = np.array([c for _, c in sorted_kept])
        w(out, f"\n{'=' * 75}")
        w(out, "Distribution Summary (kept species)")
        w(out, f"{'=' * 75}")
        w(out, f"  Classes:    {len(counts)}")
        w(out, f"  Min:        {counts.min()}")
        w(out, f"  Max:        {counts.max()}")
        w(out, f"  Mean:       {counts.mean():.1f}")
        w(out, f"  Median:     {int(np.median(counts))}")
        w(out, f"  Std:        {counts.std():.1f}")
        w(out, f"  p10:        {int(np.percentile(counts, 10))}")
        w(out, f"  p25:        {int(np.percentile(counts, 25))}")
        w(out, f"  p75:        {int(np.percentile(counts, 75))}")
        w(out, f"  p90:        {int(np.percentile(counts, 90))}")

        fam_counts = Counter()
        for sp, c in sorted_kept:
            fam = sp_family.get(sp, "?")
            fam_counts[fam] += 1

        w(out, f"\n  Families represented: {len(fam_counts)}")
        for fam, n in sorted(fam_counts.items(), key=lambda x: -x[1]):
            w(out, f"    {fam:<25s} {n:>3} species")

    # ------------------------------------------------------------------
    # 7. File transfer plan
    # ------------------------------------------------------------------
    donor_dir = [d for d in img_dirs if d not in (winner["d1"], winner["d2"])][0]
    donor_files = dir_files[donor_dir]

    kept_species_set = set(winner["kept"].keys())
    archive_frames = []
    for f in winner["files"]:
        anns = frame_annotations.get(f, [])
        if not anns:
            archive_frames.append(f)
            continue
        if not any(sp in kept_species_set for sp in anns):
            archive_frames.append(f)

    archive_with_src = []
    for f in archive_frames:
        if f in dir_files[winner["d1"]]:
            archive_with_src.append((f, winner["d1"]))
        elif f in dir_files[winner["d2"]]:
            archive_with_src.append((f, winner["d2"]))

    d1_after = len(dir_files[winner["d1"]]) - sum(1 for _, d in archive_with_src if d == winner["d1"])
    d2_after = len(dir_files[winner["d2"]]) - sum(1 for _, d in archive_with_src if d == winner["d2"])
    import_dest = winner["d1"] if d1_after <= d2_after else winner["d2"]

    w(out, f"\n{'=' * 75}")
    w(out, f"File Transfer Plan (threshold={min_samples})")
    w(out, f"{'=' * 75}")
    w(out, f"  Step 1: Archive {len(archive_frames):,} dropped-species frames")
    w(out, f"          FROM: {winner['d1']}, {winner['d2']}")
    w(out, f"          TO:   {archive_dir}")
    w(out, f"")
    w(out, f"  Step 2: Import {len(donor_files):,} frames from donor")
    w(out, f"          FROM: {donor_dir}")
    w(out, f"          TO:   {import_dest}")
    w(out, f"")
    w(out, f"  Net SSD change: {len(donor_files) - len(archive_frames):+,} frames")
    w(out, f"  SSD after: ~{winner['total_frames'] - len(archive_frames) + len(donor_files):,} frames")

    # Write file lists
    suffix = f"_t{min_samples}"
    archive_list_path = f"ssd_species_report_archive{suffix}.txt"
    import_list_path = f"ssd_species_report_import{suffix}.txt"

    with open(archive_list_path, "w") as al:
        al.write(f"# Move these frames to archive: {archive_dir}\n")
        al.write(f"# Total: {len(archive_with_src)} files\n")
        al.write(f"# Threshold: {min_samples} (frames with only below-threshold species)\n\n")
        for f, src_dir in sorted(archive_with_src, key=lambda x: x[0]):
            al.write(f"{os.path.join(src_dir, f)}\n")

    with open(import_list_path, "w") as il:
        il.write(f"# Move all frames from donor to SSD: {import_dest}\n")
        il.write(f"# Total: {len(donor_files)} files\n\n")
        for f in sorted(donor_files):
            il.write(f"{os.path.join(donor_dir, f)}\n")

    w(out, f"\n  Archive list: {archive_list_path} ({len(archive_with_src)} files)")
    w(out, f"  Import list:  {import_list_path} ({len(donor_files)} files)")

    # Shell commands
    w(out, f"\n  Shell commands:")
    w(out, f'    mkdir -p "{archive_dir}"')
    w(out, f"    # Archive:")
    w(out, f"    while IFS= read -r src; do")
    w(out, f'      [[ "$src" == "#"* || -z "$src" ]] && continue')
    w(out, f'      mv "$src" "{archive_dir}/"')
    w(out, f"    done < {archive_list_path}")
    w(out, f"    # Import:")
    w(out, f"    while IFS= read -r src; do")
    w(out, f'      [[ "$src" == "#"* || -z "$src" ]] && continue')
    w(out, f'      mv "$src" "{import_dest}/"')
    w(out, f"    done < {import_list_path}")


def main():
    parser = argparse.ArgumentParser(description="Best 2-folder subset selector")
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dirs", type=str, nargs="+", default=ALL_DIRS)
    parser.add_argument("--output", type=str, default="ssd_species_report.txt")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[20, 30, 50],
                        help="Min-sample thresholds to evaluate (all in one report).")
    parser.add_argument("--archive_dir", type=str,
                        default="/media/openuae/UUI/frames_waternet_archive",
                        help="4th folder on HDD to archive dropped-species frames.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Scan directories & load metadata (once)
    # ------------------------------------------------------------------
    dir_files = {}
    for d in args.img_dirs:
        dir_files[d] = scan_dir(d)

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["species"])

    frame_annotations = defaultdict(list)
    for _, row in df.iterrows():
        frame_annotations[row["file_name"]].append(row["species"])

    frame_families = {}
    for _, row in df.iterrows():
        frame_families.setdefault(row["file_name"], {})[row["species"]] = row["family"]

    # ------------------------------------------------------------------
    # 2. Pre-compute per-combo species counts (threshold-independent)
    # ------------------------------------------------------------------
    combos = []
    for d1, d2 in combinations(args.img_dirs, 2):
        files = dir_files[d1] | dir_files[d2]
        sp_counts = Counter()
        for f in files:
            for sp in frame_annotations.get(f, []):
                sp_counts[sp] += 1

        short1 = os.path.basename(d1)
        short2 = os.path.basename(d2)
        combos.append({
            "label": f"{short1} + {short2}",
            "d1": d1, "d2": d2,
            "total_frames": len(files),
            "sp_counts": sp_counts,
            "files": files,
        })

    # ------------------------------------------------------------------
    # 3. Write report — all thresholds in one file
    # ------------------------------------------------------------------
    with open(args.output, "w", encoding="utf-8") as out:

        w(out, "=" * 75)
        w(out, "BioReef.ai — Best 2-Folder Subset Selector")
        w(out, f"  Thresholds: {args.thresholds}")
        w(out, f"  Directories: {[os.path.basename(d) for d in args.img_dirs]}")
        w(out, "=" * 75)

        # Per-directory summary (once)
        w(out, f"\n{'=' * 75}")
        w(out, "Per-Directory Summary")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Directory':<45s} {'Frames':>8s} {'Annot':>8s} {'Spp':>5s}")
        w(out, "  " + "-" * 68)

        for d in args.img_dirs:
            files = dir_files[d]
            sp_c = Counter()
            for f in files:
                for sp in frame_annotations.get(f, []):
                    sp_c[sp] += 1
            ann = sum(sp_c.values())
            exists = "ok" if os.path.isdir(d) else "MISSING"
            w(out, f"  {d:<45s} {len(files):>8,} {ann:>8,} {len(sp_c):>5}  [{exists}]")

        # Threshold sensitivity overview (once)
        # Use the combo with the most total annotations as reference
        ref = max(combos, key=lambda x: sum(x["sp_counts"].values()))
        sp_ref = ref["sp_counts"]

        w(out, f"\n{'=' * 75}")
        w(out, f"Threshold Sensitivity ({ref['label']})")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Threshold':>10s} {'Species':>8s} {'Annotations':>12s} {'Dropped Spp':>12s}")
        w(out, "  " + "-" * 44)

        for t in sorted(set([5, 10, 15] + args.thresholds + [75, 100])):
            k = sum(1 for c in sp_ref.values() if c >= t)
            a = sum(c for c in sp_ref.values() if c >= t)
            d = len(sp_ref) - k
            marker = " <--" if t in args.thresholds else ""
            w(out, f"  {t:>10d} {k:>8} {a:>12,} {d:>12}{marker}")

        # Evaluate each threshold
        for t in args.thresholds:
            evaluate_threshold(
                out, t, combos, frame_annotations, frame_families,
                dir_files, args.img_dirs, args.archive_dir,
            )

    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
