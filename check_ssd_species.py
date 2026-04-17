"""
BioReef.ai — SSD Subset Species Audit
=======================================
Scans frames_waternet_1 and frames_waternet_2 (SSD), cross-references
with frame_metadata.csv, and reports which species are available, their
annotation counts, and what gets lost compared to the full CSV.

Usage:
    python check_ssd_species.py
    python check_ssd_species.py --csv_path data_oz/metadata/frame_metadata.csv
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

SSD_DIRS = [
    "data_oz/frames_waternet_1",
    "data_oz/frames_waternet_2",
]


def main():
    parser = argparse.ArgumentParser(description="Check species in SSD frame subset")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data_oz/metadata/frame_metadata.csv",
        help="Path to frame_metadata.csv",
    )
    parser.add_argument(
        "--img_dirs",
        type=str,
        nargs="+",
        default=SSD_DIRS,
        help="Directories to scan for frames.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ssd_species_report.txt",
        help="Output file for the report.",
    )
    args = parser.parse_args()

    # Tee all output to both console and file
    report_file = open(args.output, "w", encoding="utf-8")
    original_stdout = sys.stdout

    class Tee:
        def __init__(self, *targets):
            self.targets = targets
        def write(self, text):
            for t in self.targets:
                t.write(text)
        def flush(self):
            for t in self.targets:
                t.flush()

    sys.stdout = Tee(original_stdout, report_file)

    # ------------------------------------------------------------------
    # 1. Discover all frame filenames on SSD
    # ------------------------------------------------------------------
    ssd_files: set[str] = set()
    for d in args.img_dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Directory not found, skipping: {d}")
            continue
        for fname in os.listdir(d):
            if fname.endswith(".png"):
                ssd_files.add(fname)

    print(f"Frames on SSD: {len(ssd_files):,}")

    # ------------------------------------------------------------------
    # 2. Load metadata CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["species"])
    total_annotations = len(df)
    all_species = sorted(df["species"].unique())
    print(f"Total CSV annotations (with species): {total_annotations:,}")
    print(f"Total species in CSV: {len(all_species)}")

    # ------------------------------------------------------------------
    # 3. Filter to SSD-available frames
    # ------------------------------------------------------------------
    df["on_ssd"] = df["file_name"].isin(ssd_files)
    df_ssd = df[df["on_ssd"]]
    df_missing = df[~df["on_ssd"]]

    ssd_annotations = len(df_ssd)
    missing_annotations = len(df_missing)
    print(f"\nAnnotations with frames on SSD:  {ssd_annotations:,}  "
          f"({100 * ssd_annotations / total_annotations:.1f}%)")
    print(f"Annotations missing (no frame):  {missing_annotations:,}  "
          f"({100 * missing_annotations / total_annotations:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Species breakdown for SSD subset
    # ------------------------------------------------------------------
    ssd_species_counts = Counter(df_ssd["species"])
    ssd_species = sorted(ssd_species_counts.keys())

    # Species lost entirely
    lost_species = sorted(set(all_species) - set(ssd_species))

    print(f"\nSpecies available on SSD: {len(ssd_species)}")
    print(f"Species lost (no frames on SSD): {len(lost_species)}")

    if lost_species:
        print("\n--- Lost Species ---")
        csv_counts = Counter(df["species"])
        for sp in lost_species:
            print(f"  {sp:40s}  ({csv_counts[sp]} annotations in full CSV)")

    # ------------------------------------------------------------------
    # 5. Per-species table (SSD subset)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"{'Species':<40s} {'Family':<20s} {'Count':>6s}")
    print(f"{'='*70}")

    # Get family for each species (take first occurrence)
    sp_family = df_ssd.groupby("species")["family"].first().to_dict()

    sorted_by_count = sorted(ssd_species_counts.items(), key=lambda x: -x[1])
    for sp, count in sorted_by_count:
        fam = sp_family.get(sp, "?")
        print(f"  {sp:<38s} {fam:<20s} {count:>6d}")

    print(f"{'='*70}")
    print(f"  {'TOTAL':<38s} {'':<20s} {ssd_annotations:>6d}")

    # ------------------------------------------------------------------
    # 6. Distribution summary
    # ------------------------------------------------------------------
    counts = [c for _, c in sorted_by_count]
    import numpy as np

    counts_arr = np.array(counts)
    print(f"\n--- Distribution Summary ---")
    print(f"  Classes:  {len(counts)}")
    print(f"  Min:      {counts_arr.min()}")
    print(f"  Max:      {counts_arr.max()}")
    print(f"  Mean:     {counts_arr.mean():.1f}")
    print(f"  Median:   {int(np.median(counts_arr))}")
    print(f"  Std:      {counts_arr.std():.1f}")

    # Tail classes (< 10 samples)
    tail = [sp for sp, c in sorted_by_count if c < 10]
    if tail:
        print(f"\n  Tail classes (<10 samples): {len(tail)}")
        for sp in tail:
            print(f"    {sp:40s} {ssd_species_counts[sp]:>4d}")

    # ------------------------------------------------------------------
    # 7. Unique videos per directory
    # ------------------------------------------------------------------
    print(f"\n--- Frames per directory ---")
    for d in args.img_dirs:
        if not os.path.isdir(d):
            continue
        n = sum(1 for f in os.listdir(d) if f.endswith(".png"))
        print(f"  {d}: {n:,} frames")

    # Clean up
    sys.stdout = original_stdout
    report_file.close()
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
