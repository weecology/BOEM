"""
Load existing USGS/UBFAI annotations from crops/train.csv and crops/test.csv
so the pipeline can exclude those images from the active-learning pool.
"""
import os
import re
from typing import Optional

import pandas as pd

USGS_CROPS_DIR = "/blue/ewhite/b.weinstein/BOEM/training/crops"
TRAIN_CSV = os.path.join(USGS_CROPS_DIR, "train.csv")
TEST_CSV = os.path.join(USGS_CROPS_DIR, "test.csv")

# Crop image_path is like C1_L6_F560_T20241219_173703_737_23.png -> parent stem is C1_L6_F560_T20241219_173703_737
_CROP_SUFFIX_RE = re.compile(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$")


def _crop_path_to_parent_stem(crop_path: str) -> Optional[str]:
    """From crop filename (e.g. C1_L6_F560_T20241219_173703_737_23.png) return parent stem (e.g. C1_L6_F560_T20241219_173703_737)."""
    basename = os.path.basename(crop_path)
    m = _CROP_SUFFIX_RE.match(basename)
    if m is None:
        return None
    return m.group(1)


def _parent_stem_to_image_path(image_dir: str, parent_stem: str) -> Optional[str]:
    """Resolve parent stem to full path using .jpg or .JPG in image_dir. Returns None if not found."""
    for ext in (".jpg", ".JPG"):
        path = os.path.join(image_dir, parent_stem + ext)
        if os.path.isfile(path):
            return path
    return None


def load_usgs_annotated_image_paths(
    image_dir: str,
    train_csv: str = TRAIN_CSV,
    test_csv: str = TEST_CSV,
) -> tuple[list[str], list[str]]:
    """
    Load image_path column from USGS train.csv and test.csv, map crop paths to parent
    image paths that exist in image_dir, and return (train_paths, test_paths).

    Crop paths like C1_L6_F560_T20241219_173703_737_23.png are normalized to the
    parent image (e.g. .../C1_L6_F560_T20241219_173703_737.jpg or .JPG).

    Returns:
        (train_full_paths, test_full_paths) for parent images that exist in image_dir.
    """
    train_paths: list[str] = []
    test_paths: list[str] = []

    for csv_path, out_list in ((train_csv, train_paths), (test_csv, test_paths)):
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "image_path" not in df.columns:
            continue
        seen = set()
        for crop_path in df["image_path"].astype(str).unique():
            stem = _crop_path_to_parent_stem(crop_path)
            if stem is None:
                continue
            full_path = _parent_stem_to_image_path(image_dir, stem)
            if full_path is not None and full_path not in seen:
                seen.add(full_path)
                out_list.append(full_path)

    return train_paths, test_paths
