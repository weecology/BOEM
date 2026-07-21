#!/usr/bin/env python3
"""Trace classification crop paths back to parent images and patch geometry.

Classification crops (USGS_classification / write_crops) are named
  {patch_basename}_{row_index}.png
e.g. 950561-0926191842531-CAM9_24_7288.png -> parent patch 950561-0926191842531-CAM9_24.png,
row_index 7288 = 7289-th bbox in annotations. Byte-identical crops mean two rows
yielded the same expanded region (overlapping/duplicate bboxes).

Usage:
  uv run python scripts/investigate_duplicate_crops.py path/to/crop1.png path/to/crop2.png ...
  uv run python scripts/investigate_duplicate_crops.py --csv output/high_loss_crops_top100.csv  # first 5 rows
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Match prepare_USGS.py
UBFAI_BASE = "/blue/ewhite/b.weinstein/BOEM/training"
UBFAI_CROPS = os.path.join(UBFAI_BASE, "crops")
UBFAI_IMAGES = os.path.join(UBFAI_BASE, "images_parent")
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"


def crop_path_to_stem_and_index(crop_path: str) -> tuple[str, int] | None:
    """Parse classification crop '950561-0926191842531-CAM9_24_7288.png' -> (patch_stem, row_index)."""
    basename = os.path.basename(crop_path)
    name, ext = os.path.splitext(basename)
    if "_" not in name:
        return None
    # Last token is the patch index
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    stem, index_str = parts
    if not index_str.isdigit():
        return None
    return stem, int(index_str)


def find_parent_patch(parent_stem: str) -> Path | None:
    """Locate parent patch image (parent_stem.png) from which classification crops were cut."""
    for base in (UBFAI_CROPS, DETECTION_CROPS_BASE):
        if not os.path.isdir(base):
            continue
        p = Path(base) / f"{parent_stem}.png"
        if p.exists():
            return p
        if base == DETECTION_CROPS_BASE:
            for flight_dir in Path(base).iterdir():
                if not flight_dir.is_dir():
                    continue
                q = flight_dir / f"{parent_stem}.png"
                if q.exists():
                    return q
    return None


def find_annotation_csv_for_patch(parent_stem: str):
    """Find CSV that contains rows with image_path == parent_stem.png. Returns Path or None."""
    import pandas as pd

    csv_dir = Path(UBFAI_CROPS)
    if not csv_dir.exists():
        return None
    for csv_path in csv_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "image_path" not in df.columns:
            continue
        if (df["image_path"] == f"{parent_stem}.png").any():
            return csv_path
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trace crops to parent image and patch geometry")
    parser.add_argument("crops", nargs="*", help="Crop image paths (e.g. .../ClassName/foo_7288.png)")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Alternatively read crop_path from this CSV (uses first 5 rows)",
    )
    args = parser.parse_args()

    crop_paths = list(args.crops)
    if args.csv and args.csv.exists():
        import pandas as pd

        df = pd.read_csv(args.csv)
        if "crop_path" in df.columns:
            crop_paths.extend(df["crop_path"].head(5).tolist())
        else:
            print(f"No 'crop_path' column in {args.csv}", file=sys.stderr)
            sys.exit(1)

    if not crop_paths:
        parser.print_help()
        sys.exit(1)

    # Group by parent stem so we load each parent once
    by_stem: dict[str, list[tuple[str, int]]] = {}
    for p in crop_paths:
        parsed = crop_path_to_stem_and_index(str(p))
        if parsed is None:
            print(f"Skip (cannot parse): {p}")
            continue
        stem, idx = parsed
        by_stem.setdefault(stem, []).append((str(p), idx))

    import pandas as pd

    for parent_stem, items in by_stem.items():
        parent_path = find_parent_patch(parent_stem)
        if parent_path is None:
            print(f"\nParent stem: {parent_stem}")
            print("  Parent patch: NOT FOUND in UBFAI/crops or detection/crops/<flight>/")
            for crop_path, idx in items:
                print(f"  Crop row index {idx}: {crop_path}")
            continue

        print(f"\nParent patch: {parent_path}")
        try:
            from PIL import Image
            pil = Image.open(parent_path)
            print(f"  Patch size: {pil.size[0]} x {pil.size[1]}")
        except Exception:
            pass

        # Annotation CSV: try combined train.csv first (same order as write_crops), else per-image CSV
        csv_path = Path(UBFAI_CROPS) / "train.csv"
        if not csv_path.exists() or "image_path" not in pd.read_csv(csv_path, nrows=1).columns:
            csv_path = find_annotation_csv_for_patch(parent_stem)
        full_df = pd.read_csv(csv_path, low_memory=False) if csv_path and csv_path.exists() else None
        if full_df is not None and "image_path" in full_df.columns:
            patch_rows = full_df[full_df["image_path"] == f"{parent_stem}.png"]
            print(f"  Annotations for this patch: {len(patch_rows)} rows in {csv_path.name}")
        else:
            print("  No annotation CSV found with rows for this patch.")
            full_df = None

        for crop_path, idx in items:
            print(f"  Crop row index {idx}: {os.path.basename(crop_path)}")
            if full_df is not None and idx < len(full_df):
                row = full_df.iloc[idx]
                label = row.get("label", "?")
                xmin, ymin = row.get("xmin", row.get("left")), row.get("ymin", row.get("top"))
                xmax, ymax = row.get("xmax", None), row.get("ymax", None)
                if xmax is None and "left" in row.index and "width" in row.index:
                    xmax = row["left"] + row["width"]
                    ymax = row["top"] + row["height"]
                if xmin is not None and xmax is not None:
                    print(f"    bbox (xmin,ymin,xmax,ymax) = ({xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f})  label={label}")
                else:
                    print(f"    label={label}")
            elif full_df is not None:
                print(f"    (row index {idx} >= CSV length {len(full_df)})")
            else:
                print("    (no CSV to show bbox)")

        if len(items) >= 2 and full_df is not None:
            indices = sorted({idx for _, idx in items})
            for i, j in zip(indices[:-1], indices[1:]):
                if i < len(full_df) and j < len(full_df):
                    ri, rj = full_df.iloc[i], full_df.iloc[j]
                    xi, yi = ri.get("xmin", ri.get("left")), ri.get("ymin", ri.get("top"))
                    xj, yj = rj.get("xmin", rj.get("left")), rj.get("ymin", rj.get("top"))
                    if xi is not None and xj is not None:
                        print(f"  Row {i} vs {j}: bbox (~{xi:.0f},{yi:.0f}) vs (~{xj:.0f},{yj:.0f})")
                        if abs(xi - xj) < 50 and abs(yi - yj) < 50:
                            print("    -> Overlapping/near bboxes can yield identical expanded crops.")
                        else:
                            print("    -> Bboxes far apart; if crops are still byte-identical, train row order may differ from CSV or expand_bbox_to_square produced same region.")


if __name__ == "__main__":
    main()
