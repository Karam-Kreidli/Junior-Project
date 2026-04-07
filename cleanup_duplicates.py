"""
BioReef.ai — Duplicate Frame Cleanup
=====================================
Scans the external hard drive for WaterNet-restored images.
If a restored copy exists on the external drive, deletes the
local copy from data/frames_waternet to free VM disk space.
"""

import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Delete local WaterNet duplicates that exist on external drive")
    parser.add_argument("--local_dir", type=str, default="/media/openuae/UUI/frames_waternet_1",
                        help="Local WaterNet cache directory on the VM")
    parser.add_argument("--external_dir", type=str, default="/media/openuae/UUI/frames_waternet_3",
                        help="External hard drive WaterNet directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview deletions without actually removing files")
    args = parser.parse_args()

    if not os.path.exists(args.local_dir):
        print(f"ERROR: Local directory '{args.local_dir}' does not exist.")
        return

    if not os.path.exists(args.external_dir):
        print(f"ERROR: External directory '{args.external_dir}' does not exist.")
        print("Is the external hard drive plugged in and mounted?")
        return

    # Scan all files in the local waternet cache
    local_files = os.listdir(args.local_dir)
    print(f"Found {len(local_files)} files in local directory: {args.local_dir}")
    print(f"Checking against external directory: {args.external_dir}")

    deleted = 0
    skipped = 0

    for filename in tqdm(local_files, desc="Cleaning duplicates"):
        external_path = os.path.join(args.external_dir, filename)
        local_path = os.path.join(args.local_dir, filename)

        if os.path.exists(external_path):
            if args.dry_run:
                print(f"[DRY RUN] Would delete: {local_path}")
            else:
                try:
                    os.remove(local_path)
                except Exception as e:
                    print(f"Failed to delete {local_path}: {e}")
                    continue
            deleted += 1
        else:
            skipped += 1

    print(f"\n--- Cleanup Summary ---")
    print(f"Deleted (freed from VM) : {deleted}")
    print(f"Kept (not on ext drive) : {skipped}")
    if args.dry_run:
        print("NOTE: This was a dry run. No files were actually deleted.")

if __name__ == "__main__":
    main()
