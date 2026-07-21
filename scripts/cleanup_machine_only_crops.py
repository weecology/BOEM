"""Delete UBFAI_CROPS entries whose parent image was never human-reviewed.

`prepare_USGS.py` previously fed Tallgrass/Normandeau machine predictions into
`split_raster`, leaving ~6,689 per-image crop CSVs (and their tile PNGs) in
UBFAI_CROPS whose only annotation source was a model. The training reader
globs every CSV in UBFAI_CROPS, so even after the source filter lands in
`_load_aws_annotations()`, these stale crops still leak into train.csv.

This script removes them. By default it dry-runs and prints counts;
pass `--apply` to actually delete (with an interactive confirmation).

A crop entry consists of:
  <UBFAI_CROPS>/<stem>.csv
  <UBFAI_CROPS>/<stem>_<n>.png   (one per tile)

A parent stem is "human-reviewed" iff any AWS annotation row for that
bname_parent has source starting with "private.us-east-1." (= Sagemaker job).
"""

import argparse
import glob
import os
import sys

import pandas as pd

AWS_ANN_DIR = "/blue/ewhite/b.weinstein/BOEM/UBFAI Data Collection/annotation_aws"
UBFAI_CROPS = "/blue/ewhite/b.weinstein/BOEM/training/crops"


def load_human_reviewed_bnames() -> set:
    """Return bnames that have any human (private.us-east-1.*) annotation."""
    parts = []
    for f in glob.glob(os.path.join(AWS_ANN_DIR, "*.csv")):
        parts.append(pd.read_csv(f, usecols=["bname_parent", "source"], low_memory=False))
    aws = pd.concat(parts, ignore_index=True)
    human = aws[aws["source"].fillna("").str.startswith("private.")]
    return set(human["bname_parent"].dropna().unique())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (default is dry-run).")
    args = ap.parse_args()

    print(f"Loading human-reviewed bnames from {AWS_ANN_DIR}...")
    human = load_human_reviewed_bnames()
    print(f"  {len(human):,} bnames with any human (private.*) row\n")

    # Aggregate-level outputs we never want to touch
    skip_csvs = {"train.csv", "test.csv", "zero_shot.csv"}

    csv_paths = [
        c for c in glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
        if os.path.basename(c) not in skip_csvs
    ]

    to_delete_csvs = []
    to_delete_tiles = []
    for csv_path in csv_paths:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        if stem in human:
            continue
        to_delete_csvs.append(csv_path)
        # Tile PNGs are <stem>_<n>.png. Glob is fine; UBFAI_CROPS is flat.
        to_delete_tiles.extend(
            glob.glob(os.path.join(UBFAI_CROPS, f"{stem}_*.png"))
        )

    total_bytes = 0
    for p in to_delete_csvs + to_delete_tiles:
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass

    print(f"Scanned {len(csv_paths):,} per-image crop CSVs in {UBFAI_CROPS}")
    print(f"  CSVs to delete (machine-only parent): {len(to_delete_csvs):,}")
    print(f"  Tile PNGs to delete:                  {len(to_delete_tiles):,}")
    print(f"  Total size:                           {total_bytes/1e9:.2f} GB")
    print()

    if not to_delete_csvs:
        print("Nothing to delete.")
        return

    if not args.apply:
        print("Dry-run only. Re-run with --apply to actually delete.")
        # Sample for inspection
        print("\nSample of CSVs that would be deleted:")
        for c in to_delete_csvs[:5]:
            print(f"  {c}")
        return

    resp = input(
        f"Delete {len(to_delete_csvs):,} CSVs and {len(to_delete_tiles):,} "
        f"tiles ({total_bytes/1e9:.2f} GB)? [y/N] "
    )
    if resp.strip().lower() != "y":
        print("Aborted.")
        sys.exit(1)

    n_failed = 0
    for p in to_delete_csvs + to_delete_tiles:
        try:
            os.remove(p)
        except OSError as e:
            print(f"  failed: {p}: {e}")
            n_failed += 1

    print(
        f"\nDeleted {len(to_delete_csvs) + len(to_delete_tiles) - n_failed:,} files "
        f"({n_failed} failed)."
    )


if __name__ == "__main__":
    main()
