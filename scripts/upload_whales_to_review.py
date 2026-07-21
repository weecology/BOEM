"""Upload all whale and dolphin crop annotations to Label Studio review.

Finds all UBFAI crops labelled as cetaceans (whales, dolphins, porpoises) and
uploads them to the Label Studio review project with bounding-box preannotations.

Usage:
    uv run python scripts/upload_whales_to_review.py
    uv run python scripts/upload_whales_to_review.py --include-annotated
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

# Regex alternation matching cetacean taxa at any taxonomic level
CETACEAN_PATTERN = (
    "Tursiops|Cetacea|Delphinidae|Delphin|Mysticeti|Odontoceti|"
    "Stenella|Megaptera|Balaen|Grampus|Phocoena|Kogia|Ziphius|"
    "Mesoplodon|Physeter|Orcinus|Globicephala|Pseudorca|Feresa|"
    "Peponocephala|Steno|Sotalia|Sousa|Cephalorhynchus|Lissodelphis|"
    "Lagenorhynchus|Lagenodelphis|Inia|Platanista|Pontoporia|Lipotes"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload cetacean crops to Label Studio review.")
    p.add_argument(
        "--ubfai-crops-dir",
        default=UBFAI_CROPS_DIR,
        help=f"UBFAI crops directory (default: {UBFAI_CROPS_DIR})",
    )
    p.add_argument(
        "--include-annotated",
        action="store_true",
        help="Re-upload crops already present in the review project.",
    )
    return p.parse_args()


def _load_cetacean_pool(crops_dir: str) -> pd.DataFrame:
    all_csvs = list(Path(crops_dir).glob("*.csv"))
    per_image_csvs = [
        str(x) for x in all_csvs
        if x.name not in EXCLUDE_CSV and not x.name.startswith(EXCLUDE_PREFIX)
    ]
    if not per_image_csvs:
        raise ValueError(f"No per-image CSVs found in {crops_dir}")

    chunks = []
    for csv_path in per_image_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "label" not in df.columns:
            continue
        df["label"] = df["label"].astype(str)
        mask = df["label"].str.contains(CETACEAN_PATTERN, case=False, na=False)
        if mask.any():
            chunks.append(df[mask])

    if not chunks:
        raise ValueError("No cetacean annotations found in UBFAI crops.")

    df = pd.concat(chunks, ignore_index=True)
    dup_subset = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    if all(c in df.columns for c in dup_subset):
        df = df.drop_duplicates(subset=dup_subset)

    # Drop rows with zero/negative boxes
    df = df[(df["xmin"] >= 0) & (df["ymin"] >= 0) & (df["xmax"] > 0) & (df["ymax"] > 0)]
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])]

    return df.reset_index(drop=True)


def _get_existing_review_basenames(annotator, image_dir: str) -> set[str]:
    df = annotator.gather_data(instance_name="review", image_dir=image_dir)
    if df is None or df.empty:
        return set()
    return {os.path.basename(str(p)) for p in df["image_path"].unique()}


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

    print("Loading cetacean annotations from UBFAI crops...")
    pool = _load_cetacean_pool(crops_dir=crops_dir)
    print(f"Found {len(pool)} annotation rows across {pool['image_path'].nunique()} unique crop paths.")
    print("Label breakdown:")
    print(pool["label"].value_counts().to_string())

    pool["image_basename"] = pool["image_path"].map(lambda p: os.path.basename(str(p)))

    # Filter to crops that actually exist on disk
    pool = pool[pool["image_basename"].map(lambda b: os.path.exists(os.path.join(crops_dir, b)))]
    print(f"\n{pool['image_basename'].nunique()} crop images found on disk.")

    if not args.include_annotated:
        print("Checking for already-uploaded crops in Label Studio review...")
        existing = _get_existing_review_basenames(annotator=annotator, image_dir=crops_dir)
        pool = pool[~pool["image_basename"].isin(existing)].copy()
        print(f"{pool['image_basename'].nunique()} crops not yet in review (skipping {len(existing)} existing).")

    if pool.empty:
        print("Nothing to upload. Use --include-annotated to force re-upload.")
        return

    pool["cropmodel_label"] = pool["label"]
    pool["label"] = "Object"
    pool["score"] = 2.0
    pool["cropmodel_score"] = 2.0
    pool["comet_id"] = "cetacean_qc_upload"

    preannotations: dict[str, pd.DataFrame] = {}
    for basename, group in pool.groupby("image_basename"):
        g = group.copy()
        g["image_path"] = basename
        preannotations[basename] = g.drop(columns=["image_basename"])

    image_paths = [os.path.join(crops_dir, b) for b in sorted(preannotations)]
    print(f"\nUploading {len(image_paths)} images to Label Studio review...")
    annotator.upload(images=image_paths, instance_name="review", preannotations=preannotations)

    review_project = cfg.annotation.label_studio.instances.review.project_name
    print(f"Done. Uploaded {len(image_paths)} cetacean crop images to: {review_project}")


if __name__ == "__main__":
    main()
