#!/usr/bin/env python3
"""Find which annotation/crop CSVs contain a given image_path and when those files were modified.

Usage:
  uv run python scripts/trace_annotation_history.py 950561-0926191842531-CAM9_24.png
  uv run python scripts/trace_annotation_history.py 950561-0926191842531-CAM9.png

Searches: UBFAI crops/*.csv, detection/crops/**/*.csv, annotations train/validation/review, cumulative CSV.

Duplicate source: USGS_classification.py globs all UBFAI/crops/*.csv (including train.csv, test.csv,
and per-image CSVs). The same row can appear in both a per-image CSV and in train.csv, so concat
produces duplicates; the script dedupes with keep='first' (order depends on glob).
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"
UBFAI_BASE = "/blue/ewhite/b.weinstein/BOEM/training"
UBFAI_CROPS = os.path.join(UBFAI_BASE, "crops")
CUMULATIVE_CSV = os.path.join(UBFAI_BASE, "20260112_annotation_cumulative.csv")


def search_csv(csv_path: str, image_name: str) -> tuple[int, list[dict]]:
    """Return (n_rows, list of row dicts) for rows where image_path contains image_name or equals it."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return 0, [{"_error": str(e)}]
    if "image_path" not in df.columns:
        return 0, []
    # Match exact or basename (image_path might be full path or basename)
    mask = df["image_path"].astype(str).str.endswith(image_name) | (
        df["image_path"].astype(str) == image_name
    )
    n = int(mask.sum())
    if n == 0:
        return 0, []
    rows = df.loc[mask].to_dict("records")
    return n, rows


def main():
    if len(sys.argv) < 2:
        print("Usage: trace_annotation_history.py <image_basename>")
        print("Example: trace_annotation_history.py 950561-0926191842531-CAM9_24.png")
        sys.exit(1)
    image_name = sys.argv[1]
    # Also search for parent patch (without _24) in case image_path is the patch name
    parent_stem = image_name.replace(".png", "").replace(".PNG", "")
    if "_" in parent_stem and parent_stem.split("_")[-1].isdigit():
        parent_image = parent_stem.rsplit("_", 1)[0] + ".png"
    else:
        parent_image = None

    from datetime import datetime

    def mtime_str(path: str) -> str:
        try:
            t = os.path.getmtime(path)
            return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return "?"

    locations = []

    # UBFAI crops: check combined splits first, then any CSV containing the image (quick grep)
    if os.path.isdir(UBFAI_CROPS):
        crop_dir = Path(UBFAI_CROPS)
        # Combined splits (train.csv, test.csv, zero_shot.csv) often contain the same rows as per-image CSVs -> duplicates
        for f in sorted(crop_dir.glob("*.csv")):
            try:
                with open(f, "rb") as fp:
                    if image_name.encode() not in fp.read() and (
                        not parent_image or parent_image.encode() not in fp.read()
                    ):
                        continue
            except OSError:
                continue
            n, rows = search_csv(str(f), image_name)
            if parent_image and n == 0:
                n, rows = search_csv(str(f), parent_image)
                if n:
                    locations.append(("UBFAI/crops", str(f), n, rows, mtime_str(str(f))))
            elif n:
                locations.append(("UBFAI/crops", str(f), n, rows, mtime_str(str(f))))

    # detection/crops/**/*.csv (skip CSV unless it contains the image string)
    if os.path.isdir(DETECTION_CROPS_BASE):
        for f in sorted(Path(DETECTION_CROPS_BASE).rglob("*.csv")):
            try:
                with open(f, "rb") as fp:
                    raw = fp.read()
                    if image_name.encode() not in raw and (
                        not parent_image or parent_image.encode() not in raw
                    ):
                        continue
            except OSError:
                continue
            n, rows = search_csv(str(f), image_name)
            if parent_image and n == 0:
                n, rows = search_csv(str(f), parent_image)
                if n:
                    locations.append(
                        ("detection/crops", str(f), n, rows, mtime_str(str(f)))
                    )
            elif n:
                locations.append(("detection/crops", str(f), n, rows, mtime_str(str(f))))

    # annotations train/validation/review
    for sub in ("train", "validation", "review"):
        ann_dir = os.path.join(ANNOTATIONS_BASE, sub)
        if not os.path.isdir(ann_dir):
            continue
        for flight_dir in sorted(Path(ann_dir).iterdir()):
            if not flight_dir.is_dir():
                continue
            for f in sorted(flight_dir.glob("*.csv")):
                try:
                    with open(f, "rb") as fp:
                        raw = fp.read()
                        if image_name.encode() not in raw and (
                            not parent_image or parent_image.encode() not in raw
                        ):
                            continue
                except OSError:
                    continue
                n, rows = search_csv(str(f), image_name)
                if parent_image and n == 0:
                    n, rows = search_csv(str(f), parent_image)
                    if n:
                        locations.append(
                            (f"annotations/{sub}", str(f), n, rows, mtime_str(str(f)))
                        )
                elif n:
                    locations.append(
                        (f"annotations/{sub}", str(f), n, rows, mtime_str(str(f)))
                    )

    # Cumulative CSV
    if os.path.isfile(CUMULATIVE_CSV):
        n, rows = search_csv(CUMULATIVE_CSV, image_name)
        if parent_image and n == 0:
            n, rows = search_csv(CUMULATIVE_CSV, parent_image)
            if n:
                locations.append(
                    ("cumulative", CUMULATIVE_CSV, n, rows, mtime_str(CUMULATIVE_CSV))
                )
        elif n:
            locations.append(
                ("cumulative", CUMULATIVE_CSV, n, rows, mtime_str(CUMULATIVE_CSV))
            )

    # Report
    print(f"Image: {image_name}")
    if parent_image and parent_image != image_name:
        print(f"Parent patch (for reference): {parent_image}")
    print()
    if not locations:
        print("No annotation rows found for this image in any searched CSV.")
        return
    print("Files containing this image (or its parent patch):")
    print("-" * 80)
    for source, path, n, rows, mtime in locations:
        print(f"  Source: {source}")
        print(f"  File:   {path}")
        print(f"  mtime:  {mtime}")
        print(f"  Rows:   {n}")
        if rows and n <= 5:
            for i, r in enumerate(rows):
                # Show key fields only
                label = r.get("label", "?")
                xmin, ymin = r.get("xmin", r.get("left", "?")), r.get("ymin", r.get("top", "?"))
                xmax, ymax = r.get("xmax", "?"), r.get("ymax", "?")
                img = r.get("image_path", "?")
                print(f"    [{i}] image_path={img} label={label} xmin={xmin} ymin={ymin} xmax={xmax} ymax={ymax}")
        elif rows:
            print(f"    (first row) image_path={rows[0].get('image_path')} label={rows[0].get('label')} ...")
        print()
    print("Duplicate source: same image can appear in multiple CSVs (e.g. per-flight in detection/crops")
    print("and again in UBFAI/crops after prepare_USGS.py collect_workflow_annotations copies them.")
    print("USGS_classification.py reads all UBFAI/crops/*.csv and concats them, then dedupes by (image_path, box, label).")


if __name__ == "__main__":
    main()
