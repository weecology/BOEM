"""Trace a single annotation (e.g. Tursiops truncatus) from source to classification train/val.

Usage:
  uv run python trace_annotation_to_classification.py "Tursiops truncatus"
  uv run python trace_annotation_to_classification.py "Tursiops truncatus" --annotation-csv /path/to/annotations/train/FLIGHT/IMAGE.csv

Finds the annotation in annotations/, detection/crops/, UBFAI crops, and (if present)
train.csv/val data used by USGS_classification.
"""

import argparse
import glob
import os
import sys

import pandas as pd

ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"
UBFAI_CROPS = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"


def main():
    parser = argparse.ArgumentParser(description="Trace an annotation to classification data")
    parser.add_argument("label", help="Label (or substring) to trace, e.g. 'Tursiops truncatus'")
    parser.add_argument(
        "--annotation-csv",
        default=None,
        help="Optional: specific annotation CSV to trace (e.g. annotations/train/FLIGHT/IMAGE.csv)",
    )
    args = parser.parse_args()

    label = args.label
    print(f"Tracing label (substring): {label!r}\n")

    # 1) Annotations
    print("1) ANNOTATIONS (train/validation/review)")
    if args.annotation_csv:
        paths = [args.annotation_csv] if os.path.isfile(args.annotation_csv) else []
    else:
        paths = []
        for sub in ("train", "validation", "review"):
            parent = os.path.join(ANNOTATIONS_BASE, sub)
            if os.path.isdir(parent):
                for p in glob.glob(os.path.join(parent, "*", "*.csv")):
                    paths.append(p)
    found_ann = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "label" not in df.columns:
                continue
            if df["label"].astype(str).str.contains(label, case=False, na=False).any():
                n = df["label"].astype(str).str.contains(label, case=False, na=False).sum()
                found_ann.append((p, n))
        except Exception:
            continue
    if not found_ann:
        print("   No annotation CSV contains this label.")
        print("   So it never enters detection/crops or UBFAI.")
        return 1
    for p, n in found_ann[:15]:
        print(f"   {p} ({n} rows)")
    if len(found_ann) > 15:
        print(f"   ... and {len(found_ann) - 15} more")
    ann_csv = found_ann[0][0]
    flight = os.path.basename(os.path.dirname(ann_csv))
    image_stem = os.path.splitext(os.path.basename(ann_csv))[0]
    print(f"   Example flight={flight!r}, image_stem={image_stem!r}\n")

    # 2) Detection crops
    print("2) DETECTION CROPS (detection/crops/<flight>/*.csv)")
    flight_crops = os.path.join(DETECTION_CROPS_BASE, flight)
    if not os.path.isdir(flight_crops):
        print("   Flight dir not found. Run prepare_USGS.py --generate-detection-crops first.")
    else:
        dc_csv = os.path.join(flight_crops, f"{image_stem}.csv")
        if os.path.isfile(dc_csv):
            df = pd.read_csv(dc_csv)
            has_label = df["label"].astype(str).str.contains(label, case=False, na=False).any()
            print(f"   {dc_csv} exists. Contains label: {has_label}")
            if has_label:
                print(f"   Rows with label: {df['label'].astype(str).str.contains(label, case=False, na=False).sum()}")
        else:
            print(f"   {dc_csv} not found (crop CSV not generated or different naming).")
    ubfai_csv = os.path.join(UBFAI_CROPS, f"{image_stem}.csv")
    print()

    # 3) UBFAI crops
    print("3) UBFAI CROPS (UBFAI Images with Detection Data/crops/*.csv)")
    if os.path.isfile(ubfai_csv):
        df = pd.read_csv(ubfai_csv)
        has_label = df["label"].astype(str).str.contains(label, case=False, na=False).any()
        print(f"   {ubfai_csv} exists. Contains label: {has_label}")
    else:
        print(f"   {ubfai_csv} not found.")
    print()

    # 4) Classification filters (same logic as USGS_classification.py)
    print("4) CLASSIFICATION INPUT (after USGS_classification.py filters)")
    all_crops = glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
    if not all_crops:
        print("   No UBFAI crop CSVs found.")
        return 0
    combined = pd.concat([pd.read_csv(x) for x in all_crops])
    combined = combined.groupby("label").filter(lambda x: len(x) > 25)
    combined = combined[combined["label"].astype(str).str.contains(" ", na=False)]
    combined = combined[~combined.label.isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"])]
    combined["label"] = combined["label"].astype(str).str.strip().str.replace("/", " ").str.split().str[:2].str.join(" ")
    combined = combined[combined["label"].str.split().str.len() == 2]
    combined = combined[(combined["xmin"] != 0) & (combined["ymin"] != 0) & (combined["xmax"] != 0) & (combined["ymax"] != 0)]
    combined = combined[(combined["xmin"] >= 0) & (combined["ymin"] >= 0) & (combined["xmax"] >= 0) & (combined["ymax"] >= 0)]
    if label not in combined["label"].unique() and not combined["label"].astype(str).str.contains(label, case=False, na=False).any():
        print("   Label dropped by filters (e.g. ≤25 images, not two-word, or excluded list).")
    else:
        n = combined["label"].astype(str).str.contains(label, case=False, na=False).sum()
        print(f"   Label present in combined crop_annotations: {n} rows")
    train_csv = os.path.join(UBFAI_CROPS, "train.csv")
    if os.path.isfile(train_csv):
        tr = pd.read_csv(train_csv)
        if "label" in tr.columns and tr["label"].astype(str).str.contains(label, case=False, na=False).any():
            print(f"   train.csv: {tr['label'].astype(str).str.contains(label, case=False, na=False).sum()} rows with label")
        else:
            print("   train.csv: label not present (prepare_USGS train split may not include this flight/image).")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
