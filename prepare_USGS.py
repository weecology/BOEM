"""Prepare USGS detection data for training.

Three stages:
1. Process pre-workflow annotations from the UBFAI cumulative CSV into crops
2. Collect workflow annotations from detection/crops/ pipeline output
3. Combine both sources and create train/test splits
"""

import argparse
import glob
import os
import random
import shutil

import numpy as np
import pandas as pd
import PIL.Image
import torch
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file

# --- Paths ---
ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"
IMAGERY_BASE = "/blue/ewhite/b.weinstein/BOEM/imagery"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"
UBFAI_BASE = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data"
UBFAI_IMAGES = os.path.join(UBFAI_BASE, "images_parent")
UBFAI_CROPS = os.path.join(UBFAI_BASE, "crops")
CUMULATIVE_CSV = os.path.join(UBFAI_BASE, "20260112_annotation_cumulative.csv")
PATCH_SIZE = 1000
PATCH_OVERLAP = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare USGS detection data for training"
    )
    parser.add_argument(
        "--generate-detection-crops",
        action="store_true",
        help="Generate detection/crops/<flight>/ from existing annotations "
        "(train/validation/review) before the main preparation.",
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
        "--max-test-images",
        type=int,
        default=100,
        help="Max unique images in test.csv (default: 100).",
    )
    parser.add_argument(
        "--max-zeroshot-images",
        type=int,
        default=100,
        help="Max unique images in zero_shot.csv (default: 100).",
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


def _regenerate_ubfai_crops(df: pd.DataFrame):
    """Regenerate UBFAI crops using Dask for parallel split_raster calls."""
    from dask.distributed import as_completed
    from src.cluster import start

    client = start(cpus=5, mem_size="40GB")

    def process_image(image_annotations):
        x = image_annotations.image_path.unique()[0]
        filename = os.path.join(UBFAI_CROPS, x.replace(".JPG", ".csv"))
        if os.path.exists(filename):
            return pd.read_csv(filename)
        try:
            split_raster(
                annotations_file=image_annotations,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                path_to_raster=os.path.join(UBFAI_IMAGES, x),
                root_dir=UBFAI_IMAGES,
                base_dir=UBFAI_CROPS,
                allow_empty=False,
            )
            return filename
        except Exception as e:
            print(f"Error processing {x}: {e}")
            return None

    futures = [
        client.submit(process_image, df[df["image_path"] == x])
        for x in df.image_path.unique()
    ]
    for future in as_completed(futures):
        future.result()


# ---------------------------------------------------------------------------
# Stage 0 (optional): Generate detection crops from annotation directories
# ---------------------------------------------------------------------------


def generate_detection_crops():
    """Generate detection/crops/<flight>/ from existing annotation directories.

    Discovers flights from train/validation/review annotation subdirs,
    then preprocesses their images into crops via data_processing.
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
        "from existing annotations."
    )

    for flight_name in flights:
        root_dir = os.path.join(IMAGERY_BASE, flight_name)
        save_dir = os.path.join(DETECTION_CROPS_BASE, flight_name)
        if not os.path.isdir(root_dir):
            print(f"  Skip {flight_name}: imagery dir not found {root_dir}")
            continue

        csvs = []
        for sub in ("train", "validation", "review"):
            csvs.extend(
                glob.glob(os.path.join(ANNOTATIONS_BASE, sub, flight_name, "*.csv"))
            )
        if not csvs:
            print(f"  Skip {flight_name}: no annotation CSVs")
            continue

        combined = pd.concat(
            [pd.read_csv(f) for f in csvs], ignore_index=True
        ).drop_duplicates()
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

        os.makedirs(save_dir, exist_ok=True)
        data_processing.preprocess_images(
            combined,
            root_dir=root_dir,
            save_dir=save_dir,
            patch_size=PATCH_SIZE,
            patch_overlap=PATCH_OVERLAP,
            allow_empty=True,
        )
        print(
            f"  {flight_name}: {combined['image_path'].nunique()} images -> {save_dir}"
        )

    print("Detection crop generation done.")


# ---------------------------------------------------------------------------
# Stage 1: Pre-workflow annotations (UBFAI cumulative CSV)
# ---------------------------------------------------------------------------


def process_preworkflow_annotations(
    regenerate_crops: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crop and label-normalise the pre-workflow UBFAI cumulative annotations.

    Reads the cumulative annotation CSV, validates image existence, optionally
    regenerates crops, normalises labels (background classes become empty-image
    markers, actual detections become "Object"), and performs an initial 95/5
    train/test split by image.

    Returns:
        (train, test) DataFrames of crop annotations.
    """
    df = pd.read_csv(CUMULATIVE_CSV)
    print(df.label.value_counts())

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

    # Read all crop CSVs (produced by earlier split_raster runs)
    crop_csvs = glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
    crop_annotations = pd.concat([pd.read_csv(x) for x in crop_csvs])

    # Label normalisation: background classes (Algae/Boat/Buoy) become
    # empty-image markers (zeroed coords), everything else becomes
    # "FalsePositive" temporarily for deduplication, then all set to "Object".
    crop_annotations.loc[
        crop_annotations["label"].isin(["Algae", "Boat", "Buoy"]),
        ["xmin", "xmax", "ymin", "ymax", "label"],
    ] = [0, 0, 0, 0, "Object"]
    crop_annotations.loc[
        ~crop_annotations["label"].isin(["Algae", "Boat", "Buoy"]), "label"
    ] = "FalsePositive"

    # Deduplicate false-positive rows and drop any sharing an image with a
    # true positive (zeroed-coord background marker).
    falsepositives = crop_annotations[
        crop_annotations["label"] == "FalsePositive"
    ].drop_duplicates(subset=["xmin", "xmax", "ymin", "ymax"])
    true_positives = crop_annotations[crop_annotations["label"] != "FalsePositive"]
    falsepositives = falsepositives[
        ~falsepositives["image_path"].isin(true_positives["image_path"])
    ]
    crop_annotations = pd.concat([true_positives, falsepositives])
    crop_annotations["label"] = "Object"

    # 95/5 train/test split by image
    images = crop_annotations.image_path.unique()
    random.shuffle(images)
    split_idx = int(len(images) * 0.95)
    train = crop_annotations[crop_annotations["image_path"].isin(images[:split_idx])]
    test = crop_annotations[crop_annotations["image_path"].isin(images[split_idx:])]

    return train, test


# ---------------------------------------------------------------------------
# Stage 2: Workflow annotations (detection/crops/ from active learning)
# ---------------------------------------------------------------------------


def collect_workflow_annotations() -> pd.DataFrame:
    """Collect annotations produced by the active-learning workflow.

    Sweeps detection/crops/ for per-flight CSVs, copies them (and their
    associated crop images) into UBFAI_CROPS so everything lives in one place.

    Returns:
        Combined DataFrame of all workflow flight annotations.
    """
    flight_annotations = []
    png_pool = glob.glob(
        os.path.join(DETECTION_CROPS_BASE, "**", "*.png"), recursive=True
    )

    for csv_file in glob.glob(
        os.path.join(DETECTION_CROPS_BASE, "**", "*.csv"), recursive=True
    ):
        flight_name = os.path.basename(os.path.dirname(csv_file))
        annotations = pd.read_csv(csv_file)
        annotations["flight"] = flight_name
        # Paths must be basename-only so they resolve under flat UBFAI_CROPS
        annotations["image_path"] = annotations["image_path"].astype(str).apply(
            os.path.basename
        )
        dest_csv = os.path.join(UBFAI_CROPS, os.path.basename(csv_file))

        if os.path.exists(dest_csv):
            print(f"Skipping {csv_file}, already exists in {dest_csv}")
            annotations = pd.read_csv(dest_csv)
            annotations["flight"] = flight_name
            annotations["image_path"] = annotations["image_path"].astype(str).apply(
                os.path.basename
            )
        else:
            annotations.to_csv(dest_csv, index=False)
            # Copy associated crop images that aren't already present
            for src in annotations["image_path"].unique():
                src_path = [
                    x for x in png_pool if os.path.basename(x) == os.path.basename(src)
                ][0]
                dst = os.path.join(UBFAI_CROPS, os.path.basename(src))
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
    max_test_images: int = 100,
    max_zeroshot_images: int = 100,
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

    # Normalise via deepforest read_file
    train = read_file(train.drop(columns="geometry"), root_dir=UBFAI_CROPS)
    test = read_file(test.drop(columns="geometry"), root_dir=UBFAI_CROPS)
    flight_train = read_file(flight_train, root_dir=UBFAI_CROPS)
    flight_val = read_file(flight_val, root_dir=UBFAI_CROPS)

    unique_flights = sorted(flight_annotations.loc[flight_annotations["xmin"] != 0,"flight"].unique())
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

    combined_train["label"] = "Object"
    combined_test["label"] = "Object"

    # Remove oversized (2000 px wide) images from the test set
    oversized = [
        x
        for x in combined_test.image_path.unique()
        if PIL.Image.open(os.path.join(UBFAI_CROPS, x)).size[0] == 2000
    ]
    print(f"Number of size 2000 images in test set: {len(oversized)}")
    combined_test = combined_test[~combined_test.image_path.isin(oversized)]

    # Limit test to max_test_images unique images
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
        zs_images = combined_zeroshot.image_path.unique().tolist()
        keep_zs = set(random.sample(zs_images, min(max_zeroshot_images, len(zs_images))))
        combined_zeroshot = combined_zeroshot[combined_zeroshot.image_path.isin(keep_zs)]
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

    # Optional: generate detection crops from existing annotation directories
    if args.generate_detection_crops:
        generate_detection_crops()

    # Stage 1: crop and label-normalise pre-workflow annotations
    train, test = process_preworkflow_annotations(
        regenerate_crops=args.regenerate_crops
    )

    # Stage 2: collect annotations from the active-learning workflow
    flight_annotations = collect_workflow_annotations()

    # Stage 3: combine everything and produce final train/test splits
    create_train_test_split(
        train,
        test,
        flight_annotations,
        n_zeroshot_flights=args.n_zeroshot_flights,
        max_test_images=args.max_test_images,
        max_zeroshot_images=args.max_zeroshot_images,
    )


if __name__ == "__main__":
    main()
