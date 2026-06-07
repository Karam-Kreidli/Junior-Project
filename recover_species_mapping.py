"""
Recover the Stage-1 (#24) species mapping from the on-disk training images.
==========================================================================
`bioreef_stage1.pt` has a 256-class head but does not store its species
mapping. The CSV alone can't reproduce that 256 (CSV@min_samples=20 gives
307) because train_stage1.py's split_dataset ALSO (a) drops placeholder
species and (b) keeps only rows whose image actually exists in the
frames_waternet_* folders — and only THEN applies min_samples.

This script replays split_dataset's filtering exactly, using the images
present in --img_dirs as the ground truth of what the model was trained on,
and writes:

  1. <out_csv>            — frame_metadata.csv filtered to exactly the rows
                            that survived (the "final subset"), so future
                            runs can point --csv_path here and get a CSV that
                            matches the checkpoint.
  2. <out_mapping>.npz    — {idx_to_sp, sp_to_idx} drop-in replacement for
                            outputs/detections/species_mapping.npz, with the
                            SAME alphabetical-sort index order the head uses.

CRITICAL: the recovery is only valid if the recovered species count equals
the checkpoint's head size (256). The script loads the checkpoint (if given)
and refuses to write unless they match — so you can trust the result or know
immediately that the folders are not the exact training subset.

Filtering replicated from train_stage1.split_dataset:
  1. drop NaN species
  2. drop placeholder species (is_placeholder_species)
  3. keep only rows whose file_name exists in one of --img_dirs
  4. count per species; keep species with >= --min_samples surviving rows
  5. class index = position in sorted(kept_species)   <-- order matters

Usage (run on the VM, where data_oz/ exists):
    python recover_species_mapping.py \
        --csv_path frame_metadata.csv \
        --img_dirs data_oz/frames_waternet_1 data_oz/frames_waternet_2 \
        --stage1_ckpt bioreef_stage1.pt \
        --out_csv data_oz/metadata/frame_metadata_subset.csv \
        --out_mapping outputs/detections/species_mapping.npz
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Import the REAL placeholder filter so this can never drift from training.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_stage1 import is_placeholder_species  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_path", default="frame_metadata.csv",
                   help="Source frame metadata CSV.")
    p.add_argument("--img_dirs", nargs="+",
                   default=["data_oz/frames_waternet_1",
                            "data_oz/frames_waternet_2"],
                   help="Folders holding the training images. A CSV row is "
                        "kept only if its file_name exists in one of these.")
    p.add_argument("--min_samples", type=int, default=20,
                   help="Min surviving rows per species to keep it as a class. "
                        "Must match train_stage1.py (default 20).")
    p.add_argument("--stage1_ckpt", default="bioreef_stage1.pt",
                   help="Checkpoint to validate the recovered class count "
                        "against its head size. Pass '' to skip the check.")
    p.add_argument("--out_csv", default="data_oz/metadata/frame_metadata_subset.csv",
                   help="Where to write the filtered subset CSV.")
    p.add_argument("--out_mapping",
                   default="outputs/detections/species_mapping.npz",
                   help="Where to write the recovered species_mapping.npz.")
    p.add_argument("--force", action="store_true",
                   help="Write outputs even if the recovered class count does "
                        "not match the checkpoint head (for diagnosis).")
    return p.parse_args()


def index_existing_images(img_dirs):
    """Return a set of basenames present across all img_dirs (first wins is
    irrelevant — we only need membership)."""
    present = set()
    for d in img_dirs:
        if not os.path.isdir(d):
            print(f"  WARNING: image dir not found: {d}", file=sys.stderr)
            continue
        n0 = len(present)
        for fname in os.listdir(d):
            present.add(fname)
        print(f"  {d}: {len(present) - n0} new files "
              f"({len(os.listdir(d))} total in dir)")
    return present


def head_size_from_ckpt(ckpt_path):
    """Read the classifier head's output dim from the checkpoint, or None."""
    if not ckpt_path:
        return None
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt["head"]["weight"].shape[0])


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.csv_path):
        raise SystemExit(f"CSV not found: {args.csv_path}")

    print(f"Reading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  {len(df)} rows, {df['file_name'].nunique()} distinct images")

    print(f"Indexing images in {len(args.img_dirs)} folder(s):")
    present = index_existing_images(args.img_dirs)
    if not present:
        raise SystemExit("no images found in --img_dirs; cannot recover. "
                         "Run this on the VM where data_oz/ exists.")
    print(f"  total distinct image files on disk: {len(present)}")

    # --- Replicate split_dataset filtering, in order --------------------
    # 1+2: drop NaN + placeholder species.  3: keep only rows whose image
    # exists on disk.  Count survivors per species.
    kept_rows_mask = []
    survivor_species = []
    n_nan = n_placeholder = n_missing_img = 0
    for species, fname in zip(df["species"], df["file_name"]):
        if pd.isna(species):
            n_nan += 1
            kept_rows_mask.append(False)
            continue
        if is_placeholder_species(species):
            n_placeholder += 1
            kept_rows_mask.append(False)
            continue
        if fname not in present:
            n_missing_img += 1
            kept_rows_mask.append(False)
            continue
        kept_rows_mask.append(True)
        survivor_species.append(species)

    print("Filtering (replicating train_stage1.split_dataset):")
    print(f"  dropped NaN species        : {n_nan}")
    print(f"  dropped placeholder species: {n_placeholder}")
    print(f"  dropped image-not-on-disk  : {n_missing_img}")
    print(f"  surviving rows             : {sum(kept_rows_mask)}")

    # 4: keep species with >= min_samples surviving rows.
    sp_counter = Counter(survivor_species)
    kept_species = sorted(sp for sp, c in sp_counter.items()
                          if c >= args.min_samples)
    n_dropped_rare = len(sp_counter) - len(kept_species)
    print(f"  distinct species (pre min_samples): {len(sp_counter)}")
    print(f"  dropped < {args.min_samples} samples        : {n_dropped_rare}")
    print(f"  KEPT SPECIES (classes)     : {len(kept_species)}")

    # 5: class index = position in sorted(kept_species) — MUST match training.
    sp_to_idx = {sp: i for i, sp in enumerate(kept_species)}
    idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}

    # --- Validate against the checkpoint head --------------------------
    head_n = head_size_from_ckpt(args.stage1_ckpt)
    if head_n is not None:
        print(f"\nCheckpoint head size: {head_n}")
        if head_n == len(kept_species):
            print("  ✓ MATCH — recovered mapping aligns with the head.")
        else:
            print(f"  ✗ MISMATCH — head={head_n}, recovered="
                  f"{len(kept_species)}. The folders are NOT the exact "
                  f"training subset, or --min_samples is wrong.", file=sys.stderr)
            if not args.force:
                print("  Refusing to write (use --force to write anyway for "
                      "diagnosis).", file=sys.stderr)
                return 1

    # --- Write the filtered subset CSV ---------------------------------
    # Keep only the surviving rows AND only species that made the class cut,
    # so the CSV exactly characterizes the 256-class subset.
    subset = df[pd.Series(kept_rows_mask, index=df.index)].copy()
    subset = subset[subset["species"].isin(sp_to_idx)]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    subset.to_csv(args.out_csv, index=False)
    print(f"\nWrote subset CSV: {args.out_csv}  ({len(subset)} rows, "
          f"{subset['species'].nunique()} species)")

    # --- Write the recovered species_mapping.npz -----------------------
    os.makedirs(os.path.dirname(args.out_mapping) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_mapping,
        sp_to_idx=sp_to_idx,
        idx_to_sp=idx_to_sp,
    )
    print(f"Wrote mapping  : {args.out_mapping}  ({len(idx_to_sp)} species)")
    print("\nDone. Point --csv_path at the subset CSV and the recovered "
          "species_mapping.npz is in place for Stage 2 aggregation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
