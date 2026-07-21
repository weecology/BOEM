#!/usr/bin/env python3
"""
Export all Label Studio annotations (train / validation / review) to a single CSV.

For each image_path, only the annotation from the most recent CSV file is kept.
The output CSV adds a 'split' column indicating which project the annotation came from.

Usage:
    python scripts/export_annotations.py                          # default output path
    python scripts/export_annotations.py -o /path/to/output.csv
"""

import argparse
import glob
import os
import pandas as pd

ANNOTATION_ROOT = "/blue/ewhite/b.weinstein/BOEM/annotations"
DEFAULT_OUTPUT = "/blue/ewhite/b.weinstein/BOEM/annotations/all_annotations.csv"
SPLITS = ["train", "validation", "review"]


def load_split(split_dir, split_name):
    """Load all CSVs under split_dir recursively, tag with split and source file."""
    csvs = glob.glob(os.path.join(split_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        print(f"  [{split_name}] no CSVs found under {split_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  warning: could not read {csv_path}: {e}")
            continue
        if df.empty:
            continue
        df["_csv_path"] = csv_path
        # Timestamp is the stem of the filename (e.g. 20260226_131443)
        df["_timestamp"] = os.path.splitext(os.path.basename(csv_path))[0]
        # Flight name is the parent directory name
        if "flight_name" not in df.columns:
            df["flight_name"] = os.path.basename(os.path.dirname(csv_path))
        df["split"] = split_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  [{split_name}] {len(combined):,} rows from {len(csvs)} CSVs")
    return combined


def deduplicate(df):
    """For each image_path, keep only rows from the most recent CSV file."""
    if df.empty:
        return df
    # Most recent timestamp per image_path (within each split)
    latest = (
        df.groupby(["split", "image_path"])["_timestamp"]
        .max()
        .reset_index()
        .rename(columns={"_timestamp": "_keep_ts"})
    )
    df = df.merge(latest, on=["split", "image_path"])
    df = df[df["_timestamp"] == df["_keep_ts"]]
    df = df.drop(columns=["_csv_path", "_timestamp", "_keep_ts"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Export all Label Studio annotations to one CSV")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--annotation-root", default=ANNOTATION_ROOT)
    args = parser.parse_args()

    all_frames = []
    for split in SPLITS:
        split_dir = os.path.join(args.annotation_root, split)
        if not os.path.isdir(split_dir):
            print(f"  [{split}] directory not found: {split_dir}, skipping")
            continue
        df = load_split(split_dir, split)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No annotations found.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows before deduplication: {len(combined):,}")

    deduped = deduplicate(combined)
    print(f"Total rows after deduplication:  {len(deduped):,}")
    print(f"Unique images: {deduped['image_path'].nunique():,}")
    print(f"Split counts:\n{deduped['split'].value_counts().to_string()}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    deduped.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
