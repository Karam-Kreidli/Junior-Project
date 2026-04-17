"""
BioReef.ai — Best 2-Folder Subset Selector
=============================================
Given a minimum sample threshold, finds which 2-of-3 folder combination
maximizes annotations for species that meet the threshold.

Strategy:
  1. For each 2-of-3 combo, count per-species annotations
  2. Drop species below --min_samples
  3. Compare: which combo has the most annotations / species after filtering
  4. Report the winning combo with full per-species breakdown

Usage:
    python check_ssd_species.py --min_samples 20
    python check_ssd_species.py --min_samples 50
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


def main():
    parser = argparse.ArgumentParser(description="Best 2-folder subset selector")
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dirs", type=str, nargs="+", default=ALL_DIRS)
    parser.add_argument("--output", type=str, default="ssd_species_report.txt")
    parser.add_argument("--min_samples", type=int, default=20,
                        help="Drop species with fewer samples than this in the chosen combo.")
    parser.add_argument("--archive_dir", type=str,
                        default="/media/openuae/UUI/frames_waternet_archive",
                        help="4th folder on HDD to archive dropped-species frames.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Scan directories & load metadata
    # ------------------------------------------------------------------
    dir_files = {}
    for d in args.img_dirs:
        dir_files[d] = scan_dir(d)

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["species"])

    # file_name -> list of species
    frame_annotations = defaultdict(list)
    for _, row in df.iterrows():
        frame_annotations[row["file_name"]].append(row["species"])

    # file_name -> list of (species, family)
    frame_families = {}
    for _, row in df.iterrows():
        frame_families.setdefault(row["file_name"], {})[row["species"]] = row["family"]

    all_species_global = sorted(df["species"].unique())

    with open(args.output, "w", encoding="utf-8") as out:

        w(out, "=" * 75)
        w(out, "BioReef.ai — Best 2-Folder Subset Selector")
        w(out, f"  Min samples threshold: {args.min_samples}")
        w(out, "=" * 75)

        # ==============================================================
        # 2. Evaluate each 2-of-3 combination
        # ==============================================================
        combos = []
        for d1, d2 in combinations(args.img_dirs, 2):
            files = dir_files[d1] | dir_files[d2]

            # Count species in this combo
            sp_counts = Counter()
            for f in files:
                for sp in frame_annotations.get(f, []):
                    sp_counts[sp] += 1

            # Filter to species meeting threshold
            kept = {sp: c for sp, c in sp_counts.items() if c >= args.min_samples}
            dropped = {sp: c for sp, c in sp_counts.items() if c < args.min_samples}

            # Count annotations that belong to kept species
            kept_annotations = sum(kept.values())
            dropped_annotations = sum(dropped.values())

            # Count frames that have at least one kept-species annotation
            useful_frames = set()
            for f in files:
                if any(sp in kept for sp in frame_annotations.get(f, [])):
                    useful_frames.add(f)

            short1 = os.path.basename(d1)
            short2 = os.path.basename(d2)
            label = f"{short1} + {short2}"

            combos.append({
                "label": label, "d1": d1, "d2": d2,
                "total_frames": len(files),
                "useful_frames": len(useful_frames),
                "total_species": len(sp_counts),
                "kept_species": len(kept),
                "dropped_species": len(dropped),
                "kept_annotations": kept_annotations,
                "dropped_annotations": dropped_annotations,
                "kept": kept,
                "dropped": dropped,
                "sp_counts": sp_counts,
                "files": files,
            })

        # ==============================================================
        # 3. Comparison table
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, f"Comparison (min_samples={args.min_samples})")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Combo':<45s} {'Frames':>7s} {'KeptSpp':>8s} {'DrpSpp':>7s} {'KeptAnn':>9s} {'DrpAnn':>8s}")
        w(out, "  " + "-" * 86)

        for c in combos:
            w(out, f"  {c['label']:<45s} {c['total_frames']:>7,} "
                   f"{c['kept_species']:>8} {c['dropped_species']:>7} "
                   f"{c['kept_annotations']:>9,} {c['dropped_annotations']:>8,}")

        # ==============================================================
        # 4. Pick winner (most kept annotations)
        # ==============================================================
        winner = max(combos, key=lambda x: (x["kept_annotations"], x["kept_species"]))

        w(out, f"\n>>> WINNER: {winner['label']}")
        w(out, f"    Kept species:      {winner['kept_species']}")
        w(out, f"    Kept annotations:  {winner['kept_annotations']:,}")
        w(out, f"    Dropped species:   {winner['dropped_species']}")
        w(out, f"    Dropped annot:     {winner['dropped_annotations']:,}")
        w(out, f"    Total frames:      {winner['total_frames']:,}")
        w(out, f"    Useful frames:     {winner['useful_frames']:,}")

        # ==============================================================
        # 5. Kept species table (sorted by count)
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, f"Kept Species ({winner['kept_species']} species, {winner['kept_annotations']:,} annotations)")
        w(out, f"{'=' * 75}")
        w(out, f"  {'#':<4s} {'Species':<35s} {'Family':<20s} {'Count':>6s}")
        w(out, "  " + "-" * 67)

        # Get family mapping from the winner's files
        sp_family = {}
        for f in winner["files"]:
            fam_map = frame_families.get(f, {})
            for sp, fam in fam_map.items():
                if sp not in sp_family:
                    sp_family[sp] = fam

        sorted_kept = sorted(winner["kept"].items(), key=lambda x: -x[1])
        for i, (sp, count) in enumerate(sorted_kept, 1):
            fam = sp_family.get(sp, "?")
            w(out, f"  {i:<4d} {sp:<35s} {fam:<20s} {count:>6d}")

        w(out, "  " + "-" * 67)
        w(out, f"  {'':4s} {'TOTAL':<35s} {'':20s} {winner['kept_annotations']:>6,}")

        # ==============================================================
        # 6. Dropped species table
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, f"Dropped Species ({winner['dropped_species']} species, {winner['dropped_annotations']:,} annotations)")
        w(out, f"{'=' * 75}")

        sorted_dropped = sorted(winner["dropped"].items(), key=lambda x: -x[1])
        for sp, count in sorted_dropped:
            fam = sp_family.get(sp, "?")
            w(out, f"  {sp:<35s} {fam:<20s} {count:>4d}")

        # ==============================================================
        # 7. Distribution stats for kept species
        # ==============================================================
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

        # Family distribution
        fam_counts = Counter()
        for sp, c in sorted_kept:
            fam = sp_family.get(sp, "?")
            fam_counts[fam] += 1

        w(out, f"\n  Families represented: {len(fam_counts)}")
        for fam, n in sorted(fam_counts.items(), key=lambda x: -x[1]):
            w(out, f"    {fam:<25s} {n:>3} species")

        # ==============================================================
        # 8. Compare thresholds side-by-side
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, "Threshold Sensitivity (winner combo)")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Threshold':>10s} {'Species':>8s} {'Annotations':>12s} {'Dropped Spp':>12s}")
        w(out, "  " + "-" * 44)

        sp_all = winner["sp_counts"]
        for t in [5, 10, 15, 20, 30, 50, 75, 100]:
            k = sum(1 for c in sp_all.values() if c >= t)
            a = sum(c for c in sp_all.values() if c >= t)
            d = len(sp_all) - k
            marker = " <--" if t == args.min_samples else ""
            w(out, f"  {t:>10d} {k:>8} {a:>12,} {d:>12}{marker}")

        # ==============================================================
        # 9. File transfer plan
        # ==============================================================
        # Identify the donor (3rd folder not in the winner)
        donor_dir = [d for d in args.img_dirs if d not in (winner["d1"], winner["d2"])][0]
        donor_files = dir_files[donor_dir]

        # Frames on SSD that ONLY have dropped-species annotations
        # (no kept-species annotations at all) -> archive these
        kept_species_set = set(winner["kept"].keys())
        archive_frames = []
        for f in winner["files"]:
            anns = frame_annotations.get(f, [])
            if not anns:
                # No annotations -> archive (not useful for training)
                archive_frames.append(f)
                continue
            if not any(sp in kept_species_set for sp in anns):
                archive_frames.append(f)

        # Determine which SSD dir each archive frame is in
        archive_with_src = []
        for f in archive_frames:
            if f in dir_files[winner["d1"]]:
                archive_with_src.append((f, winner["d1"]))
            elif f in dir_files[winner["d2"]]:
                archive_with_src.append((f, winner["d2"]))

        # Destination for donor files: whichever SSD dir will have fewer
        # frames after archiving
        d1_after = len(dir_files[winner["d1"]]) - sum(1 for _, d in archive_with_src if d == winner["d1"])
        d2_after = len(dir_files[winner["d2"]]) - sum(1 for _, d in archive_with_src if d == winner["d2"])
        import_dest = winner["d1"] if d1_after <= d2_after else winner["d2"]

        w(out, f"\n{'=' * 75}")
        w(out, "File Transfer Plan")
        w(out, f"{'=' * 75}")
        w(out, f"  Step 1: Archive {len(archive_frames):,} dropped-species frames")
        w(out, f"          FROM: {winner['d1']}, {winner['d2']}")
        w(out, f"          TO:   {args.archive_dir}")
        w(out, f"")
        w(out, f"  Step 2: Import {len(donor_files):,} frames from donor")
        w(out, f"          FROM: {donor_dir}")
        w(out, f"          TO:   {import_dest}")
        w(out, f"")
        w(out, f"  Net SSD change: {len(donor_files) - len(archive_frames):+,} frames")
        w(out, f"  SSD after: ~{winner['total_frames'] - len(archive_frames) + len(donor_files):,} frames")

        # Write archive list
        archive_list_path = args.output.replace(".txt", "_archive.txt")
        with open(archive_list_path, "w") as al:
            al.write(f"# Move these frames to archive: {args.archive_dir}\n")
            al.write(f"# Total: {len(archive_with_src)} files\n")
            al.write(f"# These frames only contain dropped species (< {args.min_samples} samples)\n\n")
            for f, src_dir in sorted(archive_with_src, key=lambda x: x[0]):
                al.write(f"{os.path.join(src_dir, f)}\n")

        # Write import list (all donor files)
        import_list_path = args.output.replace(".txt", "_import.txt")
        with open(import_list_path, "w") as il:
            il.write(f"# Copy all frames from donor to SSD: {import_dest}\n")
            il.write(f"# Total: {len(donor_files)} files\n\n")
            for f in sorted(donor_files):
                il.write(f"{os.path.join(donor_dir, f)}\n")

        w(out, f"\n  Archive list: {archive_list_path} ({len(archive_with_src)} files)")
        w(out, f"  Import list:  {import_list_path} ({len(donor_files)} files)")

        # Shell commands
        w(out, f"\n{'=' * 75}")
        w(out, "Shell Commands")
        w(out, f"{'=' * 75}")
        w(out, f"  # Step 0: Create archive folder")
        w(out, f'  mkdir -p "{args.archive_dir}"')
        w(out, f"")
        w(out, f"  # Step 1: Move dropped-species frames from SSD to archive")
        w(out, f"  while IFS= read -r src; do")
        w(out, f'    [[ "$src" == "#"* || -z "$src" ]] && continue')
        w(out, f'    mv "$src" "{args.archive_dir}/"')
        w(out, f"  done < {archive_list_path}")
        w(out, f"")
        w(out, f"  # Step 2: Move all donor frames from HDD to SSD")
        w(out, f"  while IFS= read -r src; do")
        w(out, f'    [[ "$src" == "#"* || -z "$src" ]] && continue')
        w(out, f'    mv "$src" "{import_dest}/"')
        w(out, f"  done < {import_list_path}")
        w(out, f"")
        w(out, f"  # Step 3: Verify")
        w(out, f'  echo "SSD dir 1: $(ls {winner["d1"]}/*.png 2>/dev/null | wc -l) frames"')
        w(out, f'  echo "SSD dir 2: $(ls {winner["d2"]}/*.png 2>/dev/null | wc -l) frames"')
        w(out, f'  echo "Archive:   $(ls {args.archive_dir}/*.png 2>/dev/null | wc -l) frames"')

    print(f"\nReport saved to: {args.output}")
    print(f"Archive list: {archive_list_path}")
    print(f"Import list: {import_list_path}")


if __name__ == "__main__":
    main()
