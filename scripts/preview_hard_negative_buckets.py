"""Preview the hard-negative / positive label split that prepare_USGS would
produce, without writing any train.csv / test.csv.

Reads the same inputs as Stage 3 of prepare_USGS.py:
  - per-image crop CSVs in training/crops/ (from the AWS regenerate path)
  - workflow annotations under annotations/{train,validation,review}/<flight>/*.csv

Combines them, applies `is_blacklisted_label` from src/data_processing.py,
and prints the label distribution in each bucket. Useful for sanity-checking
the bucket assignment before re-running prepare_USGS end-to-end.
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import is_blacklisted_label  # noqa: E402

UBFAI_CROPS = "/blue/ewhite/b.weinstein/BOEM/training/crops"
ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"


def load_crops_combined() -> pd.DataFrame:
    """Load per-image AWS crops + label-studio workflow annotations
    into a single DataFrame, the way Stage 3 would see them post-concat."""
    skip = {"train.csv", "test.csv", "zero_shot.csv"}
    crop_csvs = [
        c for c in glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
        if os.path.basename(c) not in skip
    ]
    print(f"Reading {len(crop_csvs):,} per-image crop CSVs from {UBFAI_CROPS}...")
    parts = []
    for c in crop_csvs:
        try:
            parts.append(pd.read_csv(c, low_memory=False))
        except Exception:
            pass
    crops = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"  -> {len(crops):,} crop rows")

    flight_csvs = []
    for sub in ("train", "validation", "review"):
        flight_csvs.extend(
            glob.glob(os.path.join(ANNOTATIONS_BASE, sub, "*", "*.csv"))
        )
    print(f"Reading {len(flight_csvs):,} flight annotation CSVs...")
    fparts = []
    for c in flight_csvs:
        try:
            fparts.append(pd.read_csv(c, low_memory=False))
        except Exception:
            pass
    flights = pd.concat(fparts, ignore_index=True) if fparts else pd.DataFrame()
    print(f"  -> {len(flights):,} flight rows")

    combined = pd.concat([crops, flights], ignore_index=True)
    print(f"Combined: {len(combined):,} rows total\n")
    return combined


def report_buckets(df: pd.DataFrame, name: str):
    if df.empty or "label" not in df.columns:
        print(f"[{name}] empty or missing 'label' column")
        return
    mask = df["label"].apply(is_blacklisted_label)
    print(f"=== {name} ===")
    print(f"Hard-negative bucket ({int(mask.sum()):,} rows) — label counts:")
    print(df.loc[mask, "label"].value_counts(dropna=False).head(25).to_string())
    print()
    print(f"Positive Object bucket ({int((~mask).sum()):,} rows) — label counts:")
    print(df.loc[~mask, "label"].value_counts(dropna=False).head(30).to_string())
    print()


def main():
    combined = load_crops_combined()
    report_buckets(combined, "combined (AWS crops + label-studio workflow)")


if __name__ == "__main__":
    main()
