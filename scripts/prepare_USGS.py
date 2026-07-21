"""Prepare USGS detection data for training.

Three stages:
1. Process pre-workflow annotations from the UBFAI cumulative CSV into crops
2. Collect workflow annotations from detection/crops/ pipeline output
3. Combine both sources and create train/test splits

Usage:
  # Normal run: refresh detection crops (where annotation is newer), then copy into UBFAI
  uv run python prepare_USGS.py

  # Skip detection crop generation (use existing detection/crops only)
  uv run python prepare_USGS.py --no-generate-detection-crops

  # Skip overwriting UBFAI from detection/crops (use existing UBFAI when dest exists)
  uv run python prepare_USGS.py --no-update-labels

We default to generating detection crops (only where annotation CSV is newer)
and to updating labels (overwriting UBFAI from detection/crops). Use
--no-generate-detection-crops or --no-update-labels to skip either step.

Stage 0 parallelizes split_raster across images when multiple CPUs are available
(set SLURM_CPUS_PER_TASK in your sbatch script, or PREPARE_USGS_CROP_WORKERS).

Why new Label Studio annotations might not appear in classification:
- Stage 0 (generate detection crops): runs by default; only regenerates a crop
  CSV when that image's annotation CSV is newer (or the crop is missing).
- Stage 2 (collect_workflow_annotations): we copy detection/crops/*.csv into
  UBFAI crops. By default we overwrite so labels stay current; use
  --no-update-labels to keep existing UBFAI when the dest file already exists.
- USGS_classification.py reads only from UBFAI crops and applies filters
  (e.g. >25 images per class, two-word labels, split-by-image); see that script.
"""

import sys
from pathlib import Path

# Add project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import PIL.Image
import torch
from deepforest.utilities import read_file

# --- Paths ---
ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"
IMAGERY_BASE = "/blue/ewhite/b.weinstein/BOEM/imagery"
SCREENED_IMAGES_BASE = "/blue/ewhite/b.weinstein/BOEM/screened_images"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"
# Read-only Globus source (annotations + raw imagery)
AWS_ANNOTATIONS_DIR = "/blue/ewhite/b.weinstein/BOEM/UBFAI Data Collection/annotation_aws"
AWS_IMAGERY_DIR = "/blue/ewhite/b.weinstein/BOEM/UBFAI Data Collection/imagery"
# Derived training artifacts (NOT inside the read-only Globus collection)
TRAINING_BASE = "/blue/ewhite/b.weinstein/BOEM/training"
UBFAI_IMAGES = AWS_IMAGERY_DIR  # crops are split off the AWS imagery in place
UBFAI_CROPS = os.path.join(TRAINING_BASE, "crops")
PATCH_SIZE = 1000
PATCH_OVERLAP = 0


def _detection_crop_parallel_workers() -> int:
    """Workers for per-image split_raster in Stage 0.

    Prefer SLURM allocation, then explicit override, then host CPU count.
    """
    for key in ("PREPARE_USGS_CROP_WORKERS", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(key)
        if raw:
            return max(1, int(raw))
    return max(1, os.cpu_count() or 1)


def _detection_crop_worker(task: tuple) -> str:
    """Run split_raster for one image (pickled args; must stay top-level for spawn)."""
    root_dir, save_dir, image_path, records, patch_size, patch_overlap = task
    import pandas as pd
    from src import data_processing

    annotation_df = pd.DataFrame.from_records(records)
    data_processing.process_image(
        image_path=image_path,
        annotation_df=annotation_df,
        root_dir=root_dir,
        save_dir=save_dir,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        allow_empty=True,
    )
    return image_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare USGS detection data for training"
    )
    parser.add_argument(
        "--no-generate-detection-crops",
        action="store_true",
        help="Skip generating detection/crops from annotations. Default is to "
        "run it (only refreshes crop CSVs when annotation CSV is newer).",
    )
    parser.add_argument(
        "--regenerate-crops",
        action="store_true",
        help="Regenerate UBFAI crops via Dask even if they already exist on disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-zeroshot-flights",
        type=int,
        default=2,
        help="Number of flights to hold out for zero-shot evaluation (default: 2).",
    )
    parser.add_argument(
        "--zero-shot-flights",
        type=str,
        default="JPG_20260202_094800,JPG_2023_Dec14",
        help="Comma-separated flight names to use as zero-shot holdout (overrides random selection).",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=None,
        help="Max unique images in test.csv (default: no limit).",
    )
    parser.add_argument(
        "--max-zeroshot-images",
        type=int,
        default=None,
        help="Max empty images in zero_shot.csv (default: no limit). All object images are always kept.",
    )
    parser.add_argument(
        "--no-update-labels",
        action="store_true",
        help="Do not overwrite UBFAI crop CSVs from detection/crops when dest exists. "
        "Default is to update (overwrite) so labels stay current.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_annotation_columns(ann: pd.DataFrame) -> pd.DataFrame:
    """Ensure xmin, ymin, xmax, ymax, label, image_path; drop geometry."""
    ann = ann.copy()
    if "geometry" in ann.columns:
        ann = ann.drop(columns=["geometry"])
    if "xmin" not in ann.columns and "left" in ann.columns:
        ann["xmin"] = ann["left"]
        ann["ymin"] = ann["top"]
        ann["xmax"] = ann["left"] + ann["width"]
        ann["ymax"] = ann["top"] + ann["height"]
    for col in ("xmin", "ymin", "xmax", "ymax", "label", "image_path"):
        if col not in ann.columns:
            raise ValueError(f"Annotation CSV missing column: {col}")
    ann["image_path"] = ann["image_path"].astype(str).apply(
        lambda p: os.path.basename(p)
    )
    return ann


def _load_aws_annotations() -> pd.DataFrame:
    """Load human-curated annotations from the AWS annotation directory.

    Only keeps rows from Sagemaker `private.us-east-1.*` jobs (human review).
    Tallgrass and Normandeau rows are machine predictions and must never enter
    the training ground truth. Also drops rows whose bname_parent has no
    matching .JPG in AWS_IMAGERY_DIR.
    """
    csv_files = glob.glob(os.path.join(AWS_ANNOTATIONS_DIR, "*.csv"))
    if not csv_files:
        print("No AWS annotation manifests found; skipping.")
        return pd.DataFrame()
    parts = [pd.read_csv(f, low_memory=False) for f in csv_files]
    aws = pd.concat(parts, ignore_index=True)

    before = len(aws)
    aws = aws[aws["source"].fillna("").str.startswith("private.")]
    print(
        f"AWS source filter: kept {len(aws):,} human rows, "
        f"dropped {before - len(aws):,} machine rows."
    )

    imagery_stems = {
        os.path.splitext(fn)[0] for fn in os.listdir(AWS_IMAGERY_DIR)
    }
    aws = aws[aws["bname_parent"].isin(imagery_stems)]

    # Remove NA labels
    aws = aws[~aws.label.isna()]

    # Preserve the raw annotator label so downstream debugging can tell apart
    # an empty marker that came from a "Bird" auto-empty vs. a FalsePositive
    # nuisance conversion vs. a real positive that got blanket-relabeled to
    # "Object". Carried through every concat/dedup below.
    aws["original_label"] = aws["label"].astype(str)

    print(
        f"AWS annotations after image filter: {len(aws):,} rows, "
        f"{aws['bname_parent'].nunique():,} unique images."
    )
    return aws


def _ubfai_crop_worker(task: tuple) -> str:
    """Run split_raster for one UBFAI parent image (pickled args; top-level for spawn).

    Mirrors `_detection_crop_worker` but uses `allow_empty=False` so we don't
    explode the crop count with empty tiles from large UBFAI scenes — those
    are not hard negatives, just empty image regions.
    """
    image_path, records = task
    import pandas as pd
    from src import data_processing

    annotation_df = pd.DataFrame.from_records(records)
    data_processing.process_image(
        image_path=image_path,
        annotation_df=annotation_df,
        root_dir=UBFAI_IMAGES,
        save_dir=UBFAI_CROPS,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        allow_empty=False,
    )
    return image_path


def _regenerate_ubfai_crops(df: pd.DataFrame):
    """Regenerate per-image UBFAI crops with ProcessPoolExecutor.

    Skips any image whose crop CSV already exists (idempotent for resumes).
    Runs inside a single sbatch allocation — no nested slurm jobs.
    """
    os.makedirs(UBFAI_CROPS, exist_ok=True)
    images = df["image_path"].unique()
    tasks = []
    for image_path in images:
        out_csv = os.path.join(UBFAI_CROPS, image_path.replace(".JPG", ".csv"))
        if os.path.exists(out_csv):
            continue
        ann = df[df["image_path"] == image_path]
        tasks.append((image_path, ann.to_dict("records")))
    if not tasks:
        print("All UBFAI crops already exist; nothing to regenerate.")
        return
    n_workers = min(_detection_crop_parallel_workers(), len(tasks))
    print(
        f"UBFAI regenerate: {len(tasks):,} images to crop, {n_workers} workers"
    )
    n_done = 0
    n_failed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_ubfai_crop_worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
                n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"  failed: {e}")
            if n_done % 500 == 0:
                print(f"  progress: {n_done:,}/{len(tasks):,}")
    print(f"UBFAI regenerate done: {n_done:,} succeeded, {n_failed:,} failed.")


# ---------------------------------------------------------------------------
# Stage 0 (optional): Generate detection crops from annotation directories
# ---------------------------------------------------------------------------


def generate_detection_crops():
    """Generate detection/crops/<flight>/ from existing annotation directories.

    Discovers flights from train/validation/review annotation subdirs,
    then preprocesses their images into crops via data_processing.

    Only regenerates crop CSVs when the source annotation CSV for that image
    is newer than the existing crop CSV (or the crop CSV is missing). Crops
    whose annotations have not changed are left untouched.
    """
    from src import data_processing

    flight_dirs = set()
    for sub in ("train", "validation", "review"):
        parent = os.path.join(ANNOTATIONS_BASE, sub)
        if os.path.isdir(parent):
            for name in os.listdir(parent):
                if os.path.isdir(os.path.join(parent, name)):
                    flight_dirs.add(name)

    flights = sorted(flight_dirs)
    print(
        f"Generating detection crops for {len(flights)} flights "
        "from existing annotations (only where annotation CSV is newer)."
    )

    for flight_name in flights:
        # Try the primary imagery dir first; fall back to screened_images for
        # older flights whose imagery was archived there but never copied into
        # IMAGERY_BASE. Without this fallback ~40 flights' worth of annotations
        # are silently dropped at stage 0.
        primary = os.path.join(IMAGERY_BASE, flight_name)
        fallback = os.path.join(SCREENED_IMAGES_BASE, flight_name)
        if os.path.isdir(primary):
            root_dir = primary
        elif os.path.isdir(fallback):
            root_dir = fallback
            print(f"  {flight_name}: using screened_images fallback")
        else:
            print(
                f"  Skip {flight_name}: imagery dir not found in "
                f"{IMAGERY_BASE} or {SCREENED_IMAGES_BASE}"
            )
            continue
        save_dir = os.path.join(DETECTION_CROPS_BASE, flight_name)

        csvs = []
        for sub in ("train", "validation", "review"):
            csvs.extend(
                glob.glob(os.path.join(ANNOTATIONS_BASE, sub, flight_name, "*.csv"))
            )
        if not csvs:
            print(f"  Skip {flight_name}: no annotation CSVs")
            continue

        # Build combined and record each row's source CSV mtime
        parts = []
        for f in csvs:
            df = pd.read_csv(f)
            df["_source_mtime"] = os.path.getmtime(f)
            parts.append(df)
        combined = pd.concat(parts, ignore_index=True).drop_duplicates()
        combined = _normalize_annotation_columns(combined)

        # Keep only annotations whose images actually exist on disk
        combined["_path"] = combined["image_path"].apply(
            lambda p: os.path.join(root_dir, p)
        )
        combined = combined[combined["_path"].apply(os.path.exists)].drop(
            columns=["_path"]
        )
        if combined.empty:
            print(f"  Skip {flight_name}: no annotations with existing images")
            continue

        # Per image: max mtime of any annotation CSV that contains it
        image_ann_mtime = combined.groupby("image_path")["_source_mtime"].max()
        combined = combined.drop(columns=["_source_mtime"])

        # Only refresh images whose annotation is newer than existing crop CSV (or missing)
        os.makedirs(save_dir, exist_ok=True)
        images_to_refresh = []
        for image_path in combined["image_path"].unique():
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            crop_csv = os.path.join(save_dir, f"{image_stem}.csv")
            ann_mtime = image_ann_mtime.loc[image_path]
            if not os.path.exists(crop_csv) or os.path.getmtime(crop_csv) < ann_mtime:
                images_to_refresh.append(image_path)
                if os.path.exists(crop_csv):
                    os.remove(crop_csv)

        if not images_to_refresh:
            print(f"  {flight_name}: no images need refresh (all crop CSVs up to date)")
            continue

        combined_refresh = combined[combined["image_path"].isin(images_to_refresh)]
        n_workers = min(_detection_crop_parallel_workers(), len(images_to_refresh))
        if n_workers <= 1:
            data_processing.preprocess_images(
                combined_refresh,
                root_dir=root_dir,
                save_dir=save_dir,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                allow_empty=True,
            )
        else:
            tasks = []
            for image_path in images_to_refresh:
                ann = combined_refresh[combined_refresh["image_path"] == image_path]
                tasks.append(
                    (
                        root_dir,
                        save_dir,
                        image_path,
                        ann.to_dict("records"),
                        PATCH_SIZE,
                        PATCH_OVERLAP,
                    )
                )
            print(
                f"  {flight_name}: parallel split_raster "
                f"({len(tasks)} images, {n_workers} workers)"
            )
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_detection_crop_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    fut.result()
        print(
            f"  {flight_name}: refreshed {len(images_to_refresh)} images (of "
            f"{combined['image_path'].nunique()} total) -> {save_dir}"
        )

    print("Detection crop generation done.")


# ---------------------------------------------------------------------------
# Stage 1: Pre-workflow annotations (AWS human-curated)
# ---------------------------------------------------------------------------


def process_preworkflow_annotations(
    regenerate_crops: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crop and label-normalise the pre-workflow human annotations.

    Reads AWS human-curated annotations (Sagemaker private.* jobs only),
    validates image existence, optionally regenerates crops, normalises labels
    (non-biodiversity classes become hard-negative empty markers, actual
    detections become "Object"), and performs an initial 95/5 train/test split
    by parent scene.

    Returns:
        (train, test) DataFrames of crop annotations.
    """
    df = _load_aws_annotations()
    if df.empty:
        raise RuntimeError(
            f"No human annotations found in {AWS_ANNOTATIONS_DIR}"
        )
    print(df.label.value_counts(dropna=False).head(10))

    df["image_path"] = df["bname_parent"] + ".JPG"

    # Remove annotations whose source images are missing
    unique_images = df["image_path"].unique()
    exists_mask = [
        os.path.exists(os.path.join(UBFAI_IMAGES, x)) for x in unique_images
    ]
    print(f"Removing {len(unique_images) - sum(exists_mask)} images that do not exist")
    df = df[df["image_path"].isin(unique_images[exists_mask])]

    # Convert left/top/width/height -> xmin/ymin/xmax/ymax
    df["xmin"] = df["left"]
    df["ymin"] = df["top"]
    df["xmax"] = df["left"] + df["width"]
    df["ymax"] = df["top"] + df["height"]

    os.makedirs(UBFAI_CROPS, exist_ok=True)

    if regenerate_crops:
        _regenerate_ubfai_crops(df)

    # Read all per-image crop CSVs produced by earlier split_raster runs.
    # CRITICAL: skip the top-level output files (train.csv, test.csv,
    # zero_shot.csv, train_max_empty_*.csv, …). Those are PRIOR-RUN outputs
    # written to this same directory; if we slurp them back in, every prior
    # row gets concatenated alongside its species-labeled source row and
    # survives dedup-on-label, producing 2x duplication after the blanket
    # relabel-to-"Object" in stage 3.
    output_basenames = {"train.csv", "test.csv", "zero_shot.csv"}
    def _is_per_image_crop_csv(path: str) -> bool:
        base = os.path.basename(path)
        if base in output_basenames:
            return False
        if base.startswith(("train_", "test_", "zero_shot_")):
            return False
        return True

    crop_csvs = [
        f for f in glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
        if _is_per_image_crop_csv(f)
    ]
    crop_annotations = pd.concat([pd.read_csv(x) for x in crop_csvs])

    # Preserve the original annotator label before any downstream filtering.
    if "original_label" not in crop_annotations.columns:
        crop_annotations["original_label"] = crop_annotations["label"].astype(str)

    # Deduplicate by (image_path, bbox, label) — empty markers (all-zero bbox)
    # collapse to one row per image; real annotations dedupe identical boxes.
    # NOTE: keep the ORIGINAL labels here (don't filter, don't blanket-relabel
    # to "Object"). Stage 3 splits everything into hard-negative vs positive
    # buckets AFTER concat, using `is_blacklisted_label` on the original
    # labels. If we filtered or relabeled here, the bucket logic would lose
    # the information it needs.
    dedup_subset = [c for c in ("image_path", "xmin", "ymin", "xmax", "ymax", "label")
                    if c in crop_annotations.columns]
    crop_annotations = crop_annotations.drop_duplicates(subset=dedup_subset)

    # 95/5 train/test split by parent scene (group all crops of the same scene together
    # so no parent image straddles the split boundary).
    _CROP_STEM_RE = re.compile(r"^(.+)_\d+\.[^.]+$")

    def _crop_to_parent(path):
        m = _CROP_STEM_RE.match(os.path.basename(str(path)))
        return m.group(1) if m else os.path.splitext(os.path.basename(str(path)))[0]

    crop_annotations = crop_annotations.copy()
    crop_annotations["_parent"] = crop_annotations["image_path"].apply(_crop_to_parent)
    parent_scenes = list(crop_annotations["_parent"].unique())
    random.shuffle(parent_scenes)
    split_idx = int(len(parent_scenes) * 0.95)
    train_parents = set(parent_scenes[:split_idx])
    test_parents = set(parent_scenes[split_idx:])
    train = crop_annotations[crop_annotations["_parent"].isin(train_parents)].drop(columns=["_parent"])
    test = crop_annotations[crop_annotations["_parent"].isin(test_parents)].drop(columns=["_parent"])

    return train, test


# ---------------------------------------------------------------------------
# Stage 2: Workflow annotations (detection/crops/ from active learning)
# ---------------------------------------------------------------------------


def collect_workflow_annotations(update_labels: bool = True) -> pd.DataFrame:
    """Collect annotations produced by the active-learning workflow.

    Sweeps detection/crops/ for per-flight CSVs, copies them (and their
    associated crop images) into UBFAI_CROPS so everything lives in one place.
    When update_labels is True (default), always overwrite dest so labels stay current.

    Returns:
        Combined DataFrame of all workflow flight annotations.
    """
    flight_annotations = []
    png_pool = glob.glob(
        os.path.join(DETECTION_CROPS_BASE, "**", "*.png"), recursive=True
    )
    png_lookup = {os.path.basename(x): x for x in png_pool}

    for csv_file in glob.glob(
        os.path.join(DETECTION_CROPS_BASE, "**", "*.csv"), recursive=True
    ):
        # Skip any CSV that lives at the top level of DETECTION_CROPS_BASE —
        # those are strays from buggy older runs and don't carry a real flight.
        # The canonical layout is DETECTION_CROPS_BASE/<flight>/<image>.csv.
        if os.path.dirname(csv_file) == DETECTION_CROPS_BASE.rstrip("/"):
            print(f"Skipping stray top-level CSV (no flight dir): {csv_file}")
            continue
        flight_name = os.path.basename(os.path.dirname(csv_file))
        annotations = pd.read_csv(csv_file)
        annotations["flight"] = flight_name
        if "original_label" not in annotations.columns and "label" in annotations.columns:
            annotations["original_label"] = annotations["label"].astype(str)
        # Paths must be basename-only so they resolve under flat UBFAI_CROPS
        annotations["image_path"] = annotations["image_path"].astype(str).apply(
            os.path.basename
        )
        dest_csv = os.path.join(UBFAI_CROPS, os.path.basename(csv_file))

        src_mtime = os.path.getmtime(csv_file)
        dest_exists = os.path.exists(dest_csv)
        dest_older = dest_exists and os.path.getmtime(dest_csv) < src_mtime
        skip = dest_exists and not update_labels and not dest_older

        if skip:
            print(f"Skipping {csv_file}, already exists in {dest_csv}")
            annotations = pd.read_csv(dest_csv)
            annotations["flight"] = flight_name
            if "original_label" not in annotations.columns and "label" in annotations.columns:
                annotations["original_label"] = annotations["label"].astype(str)
            annotations["image_path"] = annotations["image_path"].astype(str).apply(
                os.path.basename
            )
        else:
            if dest_older:
                print(f"Overwriting {dest_csv} (source {csv_file} is newer)")
            annotations.to_csv(dest_csv, index=False)
            # Copy associated crop images that aren't already present
            for src in annotations["image_path"].unique():
                src_basename = os.path.basename(src)
                src_path = png_lookup.get(src_basename)
                if src_path is None:
                    print(f"Warning: {src_basename} not found in png_pool, skipping copy")
                    continue
                dst = os.path.join(UBFAI_CROPS, src_basename)
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)

        flight_annotations.append(annotations)

    return pd.concat(flight_annotations)


# ---------------------------------------------------------------------------
# Stage 3: Combine sources and create train/test splits
# ---------------------------------------------------------------------------


def create_train_test_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    flight_annotations: pd.DataFrame,
    n_zeroshot_flights: int = 2,
    zero_shot_flights: list[str] | None = None,
    max_test_images: int | None = None,
    max_zeroshot_images: int | None = None,
):
    """Combine pre-workflow and workflow annotations into final train/test splits.

    - Respects existing train/validation/review assignments from the annotation
      directories when splitting workflow data.
    - Holds out n_zeroshot_flights entirely for zero-shot evaluation (saved as
      zero_shot.csv); these flights are excluded from train and test.
    - Keeps empty-image rows in test and zero-shot; limits each to max_test_images
      and max_zeroshot_images unique images. Removes oversized (2000 px) images.
    - Saves train.csv, test.csv, and zero_shot.csv to UBFAI_CROPS.
    """
    # Load flight-level train/val assignments from annotation directories
    train_csvs = glob.glob(os.path.join(ANNOTATIONS_BASE, "train", "*", "*.csv"))
    reviewed_csvs = glob.glob(os.path.join(ANNOTATIONS_BASE, "review", "*", "*.csv"))
    val_csvs = glob.glob(os.path.join(ANNOTATIONS_BASE, "validation", "*", "*.csv"))

    flight_train = pd.concat([pd.read_csv(x) for x in train_csvs + reviewed_csvs])
    flight_val = pd.concat([pd.read_csv(x) for x in val_csvs])

    # NOTE: do NOT filter or transform flight_annotations here. The
    # consolidated hard-negative split runs once below (after combined_train /
    # combined_test / combined_zeroshot are built) so AWS-sourced and
    # flight-sourced nuisance labels go through the same logic in one place.

    # Normalise via deepforest read_file
    train = read_file(train.drop(columns="geometry"), root_dir=UBFAI_CROPS)
    test = read_file(test.drop(columns="geometry"), root_dir=UBFAI_CROPS)
    flight_train = read_file(flight_train, root_dir=UBFAI_CROPS)
    flight_val = read_file(flight_val, root_dir=UBFAI_CROPS)

    unique_flights = sorted(flight_annotations.loc[flight_annotations["xmin"] != 0,"flight"].unique())
    if zero_shot_flights is not None:
        missing = [f for f in zero_shot_flights if f not in unique_flights]
        if missing:
            raise ValueError(f"Specified zero-shot flights not found in data: {missing}")
        zero_shot_flights = set(zero_shot_flights)
    else:
        n_holdout = min(n_zeroshot_flights, len(unique_flights))
        zero_shot_flights = set(random.sample(unique_flights, n_holdout))
    print(f"Zero-shot held-out flights: {zero_shot_flights}")

    # Map workflow crop annotations to train/test by matching parent image names
    train_parent_images = [
        os.path.splitext(x)[0] for x in flight_train["image_path"]
    ]
    test_parent_images = [os.path.splitext(x)[0] for x in flight_val["image_path"]]

    flight_annotations["bname_parent"] = flight_annotations["image_path"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    train_flight = flight_annotations[
        flight_annotations["bname_parent"].isin(train_parent_images)
        & ~flight_annotations["flight"].isin(zero_shot_flights)
    ]
    test_flight = flight_annotations[
        flight_annotations["bname_parent"].isin(test_parent_images)
        & ~flight_annotations["flight"].isin(zero_shot_flights)
    ]
    zeroshot_flight = flight_annotations[
        flight_annotations["flight"].isin(zero_shot_flights)
    ]

    combined_train = pd.concat([train, train_flight])
    combined_test = pd.concat([test, test_flight])

    # ---- Reconcile train/test/zero-shot at parent level ----
    # Stage 1's random 95/5 split (over UBFAI_CROPS/*.csv) is independent of
    # Stage 3's workflow assignment (annotations/{train,validation,review}/)
    # and the zero-shot flight holdout. Without this pass, the SAME parent's
    # crops can appear in both combined_train and combined_test, and zero-shot
    # flights' crops leak ~95/5 into train/test because Stage 2 copies their
    # per-image CSVs into UBFAI_CROPS and Stage 1 splits everything there
    # without flight awareness.
    def _parent_of(p):
        base = os.path.basename(str(p))
        stem, ext = os.path.splitext(base)
        if ext.lower() == ".png":
            last = stem.rfind("_")
            if last > 0 and stem[last + 1:].isdigit():
                return stem[:last]
            return stem
        return stem

    zs_parents = (
        set(_parent_of(p) for p in zeroshot_flight["image_path"].unique())
        if not zeroshot_flight.empty
        else set()
    )
    train_flight_parents = set(
        _parent_of(p) for p in flight_train["image_path"].unique()
    )
    test_flight_parents = set(
        _parent_of(p) for p in flight_val["image_path"].unique()
    )

    combined_train = combined_train.copy()
    combined_test = combined_test.copy()
    combined_train["_parent"] = combined_train["image_path"].apply(_parent_of)
    combined_test["_parent"] = combined_test["image_path"].apply(_parent_of)

    before_t, before_v = len(combined_train), len(combined_test)
    combined_train = combined_train[~combined_train["_parent"].isin(zs_parents)]
    combined_test = combined_test[~combined_test["_parent"].isin(zs_parents)]
    combined_train = combined_train[~combined_train["_parent"].isin(test_flight_parents)]
    combined_test = combined_test[~combined_test["_parent"].isin(train_flight_parents)]
    overlap_parents = set(combined_train["_parent"]) & set(combined_test["_parent"])
    if overlap_parents:
        print(
            f"[reconcile] dropping {len(overlap_parents):,} parents from test "
            "(no workflow signal, give train precedence)"
        )
        combined_test = combined_test[~combined_test["_parent"].isin(overlap_parents)]

    combined_train = combined_train.drop(columns="_parent")
    combined_test = combined_test.drop(columns="_parent")
    print(
        f"[reconcile] train: {before_t:,} -> {len(combined_train):,} rows | "
        f"test: {before_v:,} -> {len(combined_test):,} rows | "
        f"zs_parents={len(zs_parents):,} "
        f"train_flight_parents={len(train_flight_parents):,} "
        f"test_flight_parents={len(test_flight_parents):,}"
    )

    # NOTE: do NOT dedup on (image_path, bbox, label) here. Stage 3 later
    # blanket-relabels every surviving row to "Object", so a species-labeled
    # row and a previously-"Object"-labeled twin at the same bbox would
    # survive a label-aware dedup and only become duplicates AFTER the
    # relabel. The consolidated post-relabel dedup below catches those.

    # ---- Hard-negative bucket split (consolidated) ----
    # All rows whose label is in NON_BIODIVERSITY_LABELS (Buoy, Algae, Boat,
    # Tree, Artificial, Unknown/Other, FalsePositive) or is a stray numeric
    # become zero-bbox empty markers (hard negatives). Everything else keeps
    # its bbox and gets relabeled "Object" below.
    from src.data_processing import is_blacklisted_label, nuisance_to_hard_negative
    for name, df in (("train", combined_train), ("test", combined_test)):
        if df.empty or "label" not in df.columns:
            continue
        mask = df["label"].apply(is_blacklisted_label)
        print(f"\n[{name}] Hard-negative bucket ({int(mask.sum()):,} rows) — label counts:")
        print(df.loc[mask, "label"].value_counts(dropna=False).head(20).to_string())
        print(f"[{name}] Positive Object bucket ({int((~mask).sum()):,} rows) — label counts:")
        print(df.loc[~mask, "label"].value_counts(dropna=False).head(25).to_string())
    combined_train = nuisance_to_hard_negative(combined_train, source="prepare_USGS/stage3/train")
    combined_test  = nuisance_to_hard_negative(combined_test,  source="prepare_USGS/stage3/test")

    # Mark empty images (zeroed coords); keep them in test and zero-shot
    combined_train["empty_image"] = (
        (combined_train["xmin"] == 0)
        & (combined_train["xmax"] == 0)
        & (combined_train["ymin"] == 0)
        & (combined_train["ymax"] == 0)
    )
    combined_test["empty_image"] = (
        (combined_test["xmin"] == 0)
        & (combined_test["xmax"] == 0)
        & (combined_test["ymin"] == 0)
        & (combined_test["ymax"] == 0)
    )

    # Blanket-relabel surviving positive rows to "Object" for the single-class
    # detector. nuisance_to_hard_negative already set hard-negative rows to
    # "Object" (with bbox=0); this catches the species-level labels.
    # Stash the original label first so it survives the relabel for debugging.
    for df in (combined_train, combined_test):
        if "original_label" not in df.columns:
            df["original_label"] = df["label"].astype(str)
    combined_train["label"] = "Object"
    combined_test["label"] = "Object"

    # Final dedup AFTER relabel — collapses (a) species + prior-"Object" twins
    # at the same bbox and (b) multiple nuisance-derived empty markers per
    # image. Keep the first occurrence so original_label stays informative.
    dedup_subset = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    for name in ("train", "test"):
        df = combined_train if name == "train" else combined_test
        before = len(df)
        df = df.drop_duplicates(subset=dedup_subset, keep="first")
        n_dropped = before - len(df)
        if n_dropped:
            print(
                f"[prepare_USGS:{name}] post-relabel dedup dropped {n_dropped:,} "
                f"duplicate rows ({before:,} -> {len(df):,})"
            )
        if name == "train":
            combined_train = df
        else:
            combined_test = df

    # Remove oversized (2000 px wide) images from the test set
    oversized = [
        x
        for x in combined_test.image_path.unique()
        if PIL.Image.open(os.path.join(UBFAI_CROPS, x)).size[0] == 2000
    ]
    print(f"Number of size 2000 images in test set: {len(oversized)}")
    combined_test = combined_test[~combined_test.image_path.isin(oversized)]

    # Optionally limit test to max_test_images unique images
    if max_test_images is not None:
        test_images = combined_test.image_path.unique().tolist()
        keep_test = set(random.sample(test_images, min(max_test_images, len(test_images))))
        combined_test = combined_test[combined_test.image_path.isin(keep_test)]

    # Drop geometry column and use read_file to recreate for all
    combined_train.drop(columns="geometry", inplace=True)
    combined_test.drop(columns="geometry", inplace=True)
    combined_train = read_file(combined_train, root_dir=UBFAI_CROPS)
    combined_test = read_file(combined_test, root_dir=UBFAI_CROPS)

    # Build zero-shot set (keep empty images; label, drop oversized, limit to max_zeroshot_images)
    if not zeroshot_flight.empty:
        combined_zeroshot = zeroshot_flight.copy()
        combined_zeroshot["empty_image"] = (
            (combined_zeroshot["xmin"] == 0)
            & (combined_zeroshot["xmax"] == 0)
            & (combined_zeroshot["ymin"] == 0)
            & (combined_zeroshot["ymax"] == 0)
        )
        combined_zeroshot["label"] = "Object"
        oversized_zs = [
            x
            for x in combined_zeroshot.image_path.unique()
            if os.path.exists(os.path.join(UBFAI_CROPS, x))
            and PIL.Image.open(os.path.join(UBFAI_CROPS, x)).size[0] == 2000
        ]
        combined_zeroshot = combined_zeroshot[~combined_zeroshot.image_path.isin(oversized_zs)]
        # Always keep all images that contain at least one object; only subsample empty images
        per_image_empty_zs = combined_zeroshot.groupby("image_path")["empty_image"].all()
        object_zs_images = set(per_image_empty_zs[~per_image_empty_zs].index)
        empty_zs_images = list(per_image_empty_zs[per_image_empty_zs].index)
        if max_zeroshot_images is None:
            keep_empty_zs = set(empty_zs_images)
        else:
            n_keep_empty = max(0, max_zeroshot_images - len(object_zs_images))
            keep_empty_zs = set(random.sample(empty_zs_images, min(n_keep_empty, len(empty_zs_images))))
        keep_zs = object_zs_images | keep_empty_zs
        combined_zeroshot = combined_zeroshot[combined_zeroshot.image_path.isin(keep_zs)]
        print(f"Zero-shot: {len(object_zs_images)} object images + {len(keep_empty_zs)} empty images = {len(keep_zs)} total")
        if "geometry" in combined_zeroshot.columns:
            combined_zeroshot.drop(columns="geometry", inplace=True)
        combined_zeroshot = read_file(combined_zeroshot, root_dir=UBFAI_CROPS)
        zeroshot_path = os.path.join(UBFAI_CROPS, "zero_shot.csv")
        combined_zeroshot.to_csv(zeroshot_path, index=False)
        print(f"Saved zero_shot ({len(combined_zeroshot)} rows) to {zeroshot_path}")
    else:
        print("No zero-shot flights; skipping zero_shot.csv")

    # Save all splits
    combined_train.to_csv(os.path.join(UBFAI_CROPS, "train.csv"), index=False)
    combined_test.to_csv(os.path.join(UBFAI_CROPS, "test.csv"), index=False)
    print(
        f"Saved train ({len(combined_train)} rows), "
        f"test ({len(combined_test)} rows) to {UBFAI_CROPS}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    # Generate detection crops from annotations (default: on; only refreshes when newer)
    if not args.no_generate_detection_crops:
        generate_detection_crops()

    # Stage 1: crop and label-normalise pre-workflow annotations
    train, test = process_preworkflow_annotations(
        regenerate_crops=args.regenerate_crops
    )

    # Stage 2: collect annotations from the active-learning workflow
    flight_annotations = collect_workflow_annotations(
        update_labels=not args.no_update_labels
    )

    # Stage 3: combine everything and produce final train/test splits
    explicit_zs_flights = (
        [f.strip() for f in args.zero_shot_flights.split(",")]
        if args.zero_shot_flights
        else None
    )
    create_train_test_split(
        train,
        test,
        flight_annotations,
        n_zeroshot_flights=args.n_zeroshot_flights,
        zero_shot_flights=explicit_zs_flights,
        max_test_images=args.max_test_images,
        max_zeroshot_images=args.max_zeroshot_images,
    )


if __name__ == "__main__":
    main()
