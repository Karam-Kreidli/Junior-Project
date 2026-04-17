"""
BioReef.ai — Frame Directory Audit & Swap Planner
====================================================
Scans all three frame directories, cross-references with frame_metadata.csv,
and produces a swap plan:
  - Move OUT frames from the SSD that only contain rare/hopeless species
  - Move IN frames from the HDD that add samples for well-represented species

Goal: maximize training value within a fixed 2-folder SSD budget.

Usage:
    python check_ssd_species.py
    python check_ssd_species.py --drop_threshold 5 --boost_threshold 50
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
    parser = argparse.ArgumentParser(description="Audit & swap-plan for frame directories")
    parser.add_argument("--csv_path", type=str, default="data_oz/metadata/frame_metadata.csv")
    parser.add_argument("--img_dirs", type=str, nargs="+", default=ALL_DIRS)
    parser.add_argument("--output", type=str, default="ssd_species_report.txt")
    parser.add_argument("--drop_threshold", type=int, default=5,
                        help="Species with <= this many total samples (across ALL dirs) "
                             "are considered hopeless. Frames containing ONLY these "
                             "species are candidates for removal.")
    parser.add_argument("--boost_threshold", type=int, default=30,
                        help="Species with >= this many samples in the base pair are "
                             "considered worth boosting. Donor frames with these species "
                             "are candidates for import.")
    parser.add_argument("--max_swap", type=int, default=0,
                        help="Max frames to swap (0 = balance removals with imports).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Scan directories & load metadata
    # ------------------------------------------------------------------
    dir_files = {}
    for d in args.img_dirs:
        dir_files[d] = scan_dir(d)

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["species"])
    total_annotations = len(df)
    all_species = sorted(df["species"].unique())
    total_species = len(all_species)

    # file_name -> [(species, family)]
    frame_annotations = defaultdict(list)
    for _, row in df.iterrows():
        frame_annotations[row["file_name"]].append(row["species"])

    # Global species counts (across ALL frames in ALL dirs)
    all_frames = set()
    for d in args.img_dirs:
        all_frames |= dir_files[d]
    global_sp = Counter()
    for f in all_frames:
        for sp in frame_annotations.get(f, []):
            global_sp[sp] += 1

    with open(args.output, "w", encoding="utf-8") as out:

        w(out, "=" * 75)
        w(out, "BioReef.ai — Frame Directory Audit & Swap Plan")
        w(out, "=" * 75)
        w(out, f"CSV: {args.csv_path}")
        w(out, f"Total annotations: {total_annotations:,}")
        w(out, f"Total species: {total_species}")
        w(out, f"Drop threshold: <= {args.drop_threshold} samples globally")
        w(out, f"Boost threshold: >= {args.boost_threshold} samples in base pair")

        # ==============================================================
        # 2. Per-directory stats
        # ==============================================================
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

        # ==============================================================
        # 3. All 2-of-3 combinations
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, "2-of-3 Combinations")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Combo':<50s} {'Frames':>8s} {'Annot':>8s} {'Spp':>5s}")
        w(out, "  " + "-" * 73)

        combo_stats = []
        for d1, d2 in combinations(args.img_dirs, 2):
            combined = dir_files[d1] | dir_files[d2]
            sp_c = Counter()
            for f in combined:
                for sp in frame_annotations.get(f, []):
                    sp_c[sp] += 1
            ann = sum(sp_c.values())
            short = f"{os.path.basename(d1)} + {os.path.basename(d2)}"
            combo_stats.append({"label": short, "d1": d1, "d2": d2,
                                "sp_counts": sp_c, "annotations": ann,
                                "frames": len(combined)})
            w(out, f"  {short:<50s} {len(combined):>8,} {ann:>8,} {len(sp_c):>5}")

        # All three
        sp_all = Counter()
        for f in all_frames:
            for sp in frame_annotations.get(f, []):
                sp_all[sp] += 1
        w(out, f"  {'ALL THREE':<50s} {len(all_frames):>8,} {sum(sp_all.values()):>8,} {len(sp_all):>5}")

        # Pick best base pair (most annotations)
        best = max(combo_stats, key=lambda x: x["annotations"])
        donor_dir = [d for d in args.img_dirs if d not in (best["d1"], best["d2"])][0]

        w(out, f"\n>>> Base pair: {best['label']}")
        w(out, f">>> Donor: {os.path.basename(donor_dir)}")

        base_files = dir_files[best["d1"]] | dir_files[best["d2"]]
        base_sp = best["sp_counts"]
        donor_only = dir_files[donor_dir] - base_files

        # ==============================================================
        # 4. Identify hopeless species (globally rare)
        # ==============================================================
        hopeless = {sp for sp, c in global_sp.items() if c <= args.drop_threshold}

        w(out, f"\n{'=' * 75}")
        w(out, f"Hopeless Species (<= {args.drop_threshold} samples globally): {len(hopeless)}")
        w(out, f"{'=' * 75}")
        for sp in sorted(hopeless):
            w(out, f"  {sp:<40s} {global_sp[sp]:>4} samples globally, "
                   f"{base_sp.get(sp, 0):>4} in base pair")

        # ==============================================================
        # 5. Frames to REMOVE from SSD
        # ==============================================================
        # A frame is removable if ALL its species annotations are hopeless
        removable = []
        for f in base_files:
            anns = frame_annotations.get(f, [])
            if not anns:
                # No annotations at all — also removable
                removable.append((f, anns))
                continue
            if all(sp in hopeless for sp in anns):
                removable.append((f, anns))

        w(out, f"\n{'=' * 75}")
        w(out, f"Frames to REMOVE from SSD (only hopeless species)")
        w(out, f"{'=' * 75}")
        w(out, f"  Removable frames: {len(removable):,}")

        # What species/annotations are lost
        remove_sp = Counter()
        for f, anns in removable:
            for sp in anns:
                remove_sp[sp] += 1
        if remove_sp:
            w(out, f"  Annotations lost by removal:")
            for sp, c in sorted(remove_sp.items(), key=lambda x: -x[1]):
                w(out, f"    {sp:<40s} -{c}")

        # ==============================================================
        # 6. Frames to IMPORT from donor
        # ==============================================================
        # Worth boosting: species that already have enough samples to be
        # useful, and more samples would help training
        boostable = {sp for sp, c in base_sp.items()
                     if c >= args.boost_threshold and sp not in hopeless}

        # Score donor frames by how many boostable-species annotations they have
        import_candidates = []
        for f in donor_only:
            anns = frame_annotations.get(f, [])
            boost_hits = [sp for sp in anns if sp in boostable]
            if boost_hits:
                import_candidates.append((f, len(boost_hits), boost_hits, anns))

        import_candidates.sort(key=lambda x: -x[1])

        # Budget: import at most as many as we remove (to keep disk usage constant)
        budget = len(removable) if args.max_swap == 0 else args.max_swap
        to_import = import_candidates[:budget]

        w(out, f"\n{'=' * 75}")
        w(out, f"Frames to IMPORT from Donor (boost well-represented species)")
        w(out, f"{'=' * 75}")
        w(out, f"  Boostable species (>= {args.boost_threshold} in base): {len(boostable)}")
        w(out, f"  Donor frames with boostable species: {len(import_candidates):,}")
        w(out, f"  Budget (= removals): {budget:,}")
        w(out, f"  Frames to import: {len(to_import):,}")

        # What species/annotations are gained
        import_sp = Counter()
        for f, _, _, anns in to_import:
            for sp in anns:
                import_sp[sp] += 1

        if import_sp:
            w(out, f"\n  Top species gained by import:")
            for sp, c in sorted(import_sp.items(), key=lambda x: -x[1])[:30]:
                w(out, f"    {sp:<40s} +{c:<6d} (was {base_sp.get(sp, 0)})")

        # ==============================================================
        # 7. Net effect
        # ==============================================================
        sim_sp = Counter(base_sp)
        for sp, c in remove_sp.items():
            sim_sp[sp] -= c
            if sim_sp[sp] <= 0:
                del sim_sp[sp]
        for sp, c in import_sp.items():
            sim_sp[sp] += c

        new_species = len(sim_sp)
        new_ann = sum(sim_sp.values())

        w(out, f"\n{'=' * 75}")
        w(out, "Summary: Before vs After Swap")
        w(out, f"{'=' * 75}")
        w(out, f"  {'Metric':<30s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
        w(out, "  " + "-" * 62)
        w(out, f"  {'Frames':<30s} {len(base_files):>10,} {len(base_files)-len(removable)+len(to_import):>10,} {len(to_import)-len(removable):>+10,}")
        w(out, f"  {'Annotations':<30s} {best['annotations']:>10,} {new_ann:>10,} {new_ann-best['annotations']:>+10,}")
        w(out, f"  {'Species':<30s} {len(base_sp):>10} {new_species:>10} {new_species-len(base_sp):>+10}")
        w(out, f"  {'Removed frames':<30s} {'':>10s} {len(removable):>10,}")
        w(out, f"  {'Imported frames':<30s} {'':>10s} {len(to_import):>10,}")

        # ==============================================================
        # 8. Write move lists
        # ==============================================================
        remove_list_path = args.output.replace(".txt", "_remove.txt")
        import_list_path = args.output.replace(".txt", "_import.txt")

        # Figure out which base dir each removable frame is in
        with open(remove_list_path, "w") as rl:
            rl.write(f"# Move these frames OUT of the SSD to {donor_dir}\n")
            rl.write(f"# Total: {len(removable)} files\n\n")
            for f, _ in sorted(removable, key=lambda x: x[0]):
                if f in dir_files[best["d1"]]:
                    src = os.path.join(best["d1"], f)
                else:
                    src = os.path.join(best["d2"], f)
                rl.write(f"{src}\n")

        dest_dir = best["d1"] if len(dir_files[best["d1"]]) <= len(dir_files[best["d2"]]) else best["d2"]
        with open(import_list_path, "w") as il:
            il.write(f"# Copy these frames from {donor_dir} to {dest_dir}\n")
            il.write(f"# Total: {len(to_import)} files\n\n")
            for f, _, _, _ in sorted(to_import, key=lambda x: x[0]):
                src = os.path.join(donor_dir, f)
                il.write(f"{src}\n")

        w(out, f"\n  Remove list: {remove_list_path}")
        w(out, f"  Import list: {import_list_path}")

        # ==============================================================
        # 9. Shell commands
        # ==============================================================
        w(out, f"\n{'=' * 75}")
        w(out, "Shell Commands to Execute")
        w(out, f"{'=' * 75}")
        w(out, f"  # Step 1: Move hopeless-species frames to HDD")
        w(out, f"  while IFS= read -r src; do")
        w(out, f'    [[ "$src" == "#"* || -z "$src" ]] && continue')
        w(out, f'    mv "$src" "{donor_dir}/"')
        w(out, f"  done < {remove_list_path}")
        w(out, "")
        w(out, f"  # Step 2: Copy valuable frames from HDD to SSD")
        w(out, f"  while IFS= read -r src; do")
        w(out, f'    [[ "$src" == "#"* || -z "$src" ]] && continue')
        w(out, f'    cp "$src" "{dest_dir}/"')
        w(out, f"  done < {import_list_path}")

    print(f"\nReport saved to: {args.output}")
    print(f"Remove list: {remove_list_path}")
    print(f"Import list: {import_list_path}")


if __name__ == "__main__":
    main()
