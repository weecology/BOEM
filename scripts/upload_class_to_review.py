"""Upload USGS classification class instances to Label Studio review for QC.

This script follows the USGS classification data path:
- loads per-image crop CSVs from UBFAI crops (same source style as USGS_classification.py)
- applies the same label cleaning/filtering rules
- recreates train/validation split by parent image
- selects all instances of one class from train and validation
- uploads corresponding crop images + boxes to Label Studio review

Usage:
    uv run python scripts/upload_class_to_review.py "Tursiops truncatus"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotators import get_annotator
from src.label_studio import get_api_key

UBFAI_CROPS_DIR = "/blue/ewhite/b.weinstein/BOEM/training/crops"
EXCLUDE_CSV = frozenset({"train.csv", "test.csv", "zero_shot.csv"})
EXCLUDE_PREFIX = "train_max_empty_"
_CROP_SUFFIX_RE = re.compile(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one USGS class from classification train/val pool to Label Studio review."
    )
    parser.add_argument("target_class", help='Class label to upload, e.g. "Cepphus grylle".')
    parser.add_argument(
        "--ubfai-crops-dir",
        default=UBFAI_CROPS_DIR,
        help=f"UBFAI crops directory (default: {UBFAI_CROPS_DIR})",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=25,
        help="Keep only labels with > N rows before split (default: 25, matching USGS_classification.py).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of parent images used for validation split (default: 0.1).",
    )
    parser.add_argument(
        "--min-test-images",
        type=int,
        default=5,
        help="Minimum per-class validation rows to keep a class (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed (default: 42, matching USGS_classification.py).",
    )
    parser.add_argument(
        "--include-annotated",
        action="store_true",
        help="Include crops already present in existing train/validation/review annotation CSVs.",
    )
    return parser.parse_args()


def _crop_path_to_parent_stem(crop_path: str) -> str:
    basename = os.path.basename(str(crop_path))
    m = _CROP_SUFFIX_RE.match(basename)
    if m is None:
        return basename
    return m.group(1)


def _normalize_label(label: str) -> str:
    s = str(label).strip()
    s = s.replace("/", " ")
    return " ".join(s.split()[:2])


def _load_usgs_pool(crops_dir: str, min_class_count: int) -> pd.DataFrame:
    all_csvs = list(Path(crops_dir).glob("*.csv"))
    per_image_csvs = [
        str(x) for x in all_csvs
        if x.name not in EXCLUDE_CSV and not x.name.startswith(EXCLUDE_PREFIX)
    ]
    if not per_image_csvs:
        raise ValueError(f"No per-image CSVs found in {crops_dir}")

    df = pd.concat([pd.read_csv(x) for x in per_image_csvs], ignore_index=True)

    dup_subset = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    if all(col in df.columns for col in dup_subset):
        df = df.drop_duplicates(subset=dup_subset)

    df = df.groupby("label").filter(lambda x: len(x) > min_class_count)
    df = df[df["label"].astype(str).str.contains(" ", na=False)]
    df = df[
        ~df["label"].isin(
            [0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"]
        )
    ]
    df["label"] = df["label"].apply(_normalize_label)
    df = df[df["label"].apply(lambda x: len(str(x).split()) == 2)]
    df = df[(df["xmin"] != 0) & (df["ymin"] != 0) & (df["xmax"] != 0) & (df["ymax"] != 0)]
    df = df[(df["xmin"] >= 0) & (df["ymin"] >= 0) & (df["xmax"] >= 0) & (df["ymax"] >= 0)]
    if df.empty:
        raise ValueError("USGS pool is empty after filtering.")
    return df


def _split_by_parent_image(
    df: pd.DataFrame,
    test_size: float,
    min_test_images: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["parent_image"] = df["image_path"].map(_crop_path_to_parent_stem)
    unique_parents = df["parent_image"].unique()
    n_test_parents = max(1, int(len(unique_parents) * test_size))
    rng = np.random.default_rng(random_state)
    test_parents = set(rng.choice(unique_parents, size=n_test_parents, replace=False))
    train_parents = set(unique_parents) - test_parents

    train_df = df[df["parent_image"].isin(train_parents)].drop(columns=["parent_image"])
    val_df = df[df["parent_image"].isin(test_parents)].drop(columns=["parent_image"])

    kept = []
    for label in df["label"].unique():
        if (train_df["label"] == label).sum() >= 1 and (val_df["label"] == label).sum() >= min_test_images:
            kept.append(label)

    train_df = train_df[train_df["label"].isin(kept)]
    val_df = val_df[val_df["label"].isin(kept)]
    return train_df, val_df


def _get_existing_basenames(annotator, image_dir: str) -> set[str]:
    existing = set()
    for instance in ("train", "validation", "review"):
        df = annotator.gather_data(instance_name=instance, image_dir=image_dir)
        if df is None or df.empty:
            continue
        existing.update(os.path.basename(str(p)) for p in df["image_path"].unique())
    return existing


def main() -> None:
    args = _parse_args()
    crops_dir = args.ubfai_crops_dir
    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"UBFAI crops dir not found: {crops_dir}")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if api_key is None:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config", overrides=[f"image_dir={crops_dir}"])

    annotator = get_annotator(cfg)

    pool = _load_usgs_pool(crops_dir=crops_dir, min_class_count=args.min_class_count)
    train_df, val_df = _split_by_parent_image(
        df=pool,
        test_size=args.test_size,
        min_test_images=args.min_test_images,
        random_state=args.seed,
    )

    class_train = train_df[train_df["label"] == args.target_class].copy()
    class_val = val_df[val_df["label"] == args.target_class].copy()
    selected = pd.concat([class_train, class_val], ignore_index=True)
    if selected.empty:
        raise ValueError(
            f'No rows found for class "{args.target_class}" in USGS train/validation split.'
        )

    selected["image_basename"] = selected["image_path"].map(lambda p: os.path.basename(str(p)))
    if not args.include_annotated:
        existing_basenames = _get_existing_basenames(annotator=annotator, image_dir=crops_dir)
        selected = selected[~selected["image_basename"].isin(existing_basenames)].copy()
        if selected.empty:
            raise ValueError(
                "All selected class crops are already annotated. Use --include-annotated to force re-upload."
            )

    selected["cropmodel_label"] = selected["label"]
    selected["label"] = "Object"
    selected["score"] = 2.0
    selected["cropmodel_score"] = 2.0
    selected["comet_id"] = "usgs_qc_upload"

    preannotations = {}
    for basename, group in selected.groupby("image_basename"):
        group = group.copy()
        group["image_path"] = basename
        preannotations[basename] = group.drop(columns=["image_basename"])

    image_paths = [os.path.join(crops_dir, b) for b in sorted(preannotations)]
    annotator.upload(images=image_paths, instance_name="review", preannotations=preannotations)

    print(
        f'Uploaded {len(image_paths)} crop images for class "{args.target_class}" '
        f"(train={class_train['image_path'].nunique()}, val={class_val['image_path'].nunique()}) "
        f"to review project: {cfg.annotation.label_studio.instances.review.project_name}"
    )


if __name__ == "__main__":
    main()
