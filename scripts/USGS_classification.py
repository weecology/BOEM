"""USGS species classification training on UBFAI crop annotations.

Classification crops are extracted from UBFAI crop images using each annotation's
bounding box plus a configurable buffer (expand_pixels) on all sides. Buffering is
done in src.classification.preprocess_images; prepare_USGS.py only produces
detection crops (patch-based, no bbox buffer).

To explore buffer size: use classification_model.expand. Outputs are written to
buffer-specific subdirs so runs do not overwrite (e.g. .../checkpoints/buffer_0/,
.../crops/train/buffer_200/<comet_id>/). Example:

  uv run python scripts/USGS_classification.py classification_model.expand=0    # tight
  uv run python scripts/USGS_classification.py classification_model.expand=30  # default
  uv run python scripts/USGS_classification.py classification_model.expand=200 # large context
"""
import os
import re
import sys
from pathlib import Path

# Add project root so we can import src (parent of scripts/ = repo root).
# Submit script must cd to repo root so resolve() is correct when run via sbatch.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import glob
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from deepforest.model import CropModel

# Crop image_path is like C1_L6_F560_T20241219_173703_737_23.png -> parent stem is C1_L6_F560_T20241219_173703_737
_CROP_SUFFIX_RE = re.compile(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$")

# New-format UBFAI image names embed the flight datetime as T{YYYYMMDD}_{HHMMSS}.
# This maps directly to a captures CSV key (e.g. 20241219_165719_captures.csv).
# Old-format names (e.g. 668140-0806171338187-CAM3.png) have no parseable datetime.
_FLIGHT_DATETIME_RE = re.compile(r"T(\d{8}_\d{6})")


def _flight_name_from_image_path(image_path: str) -> str | None:
    """Extract flight datetime key from new-format UBFAI image paths, or None."""
    m = _FLIGHT_DATETIME_RE.search(os.path.basename(str(image_path)))
    return m.group(1) if m else None

# Detection-task combined splits: exclude so we only load per-image CSVs (avoids duplicate rows and Object vs species label conflicts).
UBFAI_CROPS_EXCLUDE_CSV = frozenset({"train.csv", "test.csv", "zero_shot.csv"})
UBFAI_CROPS_EXCLUDE_PREFIX = "train_max_empty_"  # e.g. train_max_empty_0_10.csv


def _crop_path_to_parent_stem(crop_path: str) -> str | None:
    """From crop filename return parent stem; if no match, treat as parent (e.g. image basename)."""
    basename = os.path.basename(str(crop_path))
    m = _CROP_SUFFIX_RE.match(basename)
    if m is None:
        return basename
    return m.group(1)

def gentle_class_balance(df, factor=3.0, random_state=None):
    """
    Gently reduce majority-class dominance by capping each class at factor * median(class_counts).
    Not full balance: small classes are unchanged; only large classes are downsampled.
    """
    counts = df.groupby("label").size()
    median_count = int(counts.median())
    cap = max(median_count, int(median_count * factor))
    balanced = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        if len(class_df) <= cap:
            balanced.append(class_df)
        else:
            balanced.append(class_df.sample(n=cap, random_state=random_state))
    return pd.concat(balanced, ignore_index=True)


# Create train test split, split each class into 90% train and 10% test with a minimum of 5 images per class for test and a max of 100
def train_test_split(df, test_size=0.1, min_test_images=5, max_test_images=100):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        test_count = max(min_test_images, int(len(class_df) * test_size))
        test_count = min(test_count, max_test_images)
        
        test_class_df = class_df.sample(n=test_count)
        train_class_df = class_df.drop(test_class_df.index)
        
        train_df = pd.concat([train_df, train_class_df])
        test_df = pd.concat([test_df, test_class_df])
    
    return train_df, test_df


def train_test_split_by_image(
    df,
    test_size=0.1,
    min_test_images=5,
    max_test_images=100,
    random_state=42,
):
    """
    Split per class by parent image so no crop for the same label appears in both splits.

    For each class independently, shuffles that class's parent images and assigns
    enough to test to satisfy both the test_size fraction and the min_test_images
    crop count (whichever requires more parents). This guarantees minority classes
    get adequate test representation regardless of their share of the full dataset.

    A parent image may be in test for one class and train for another — that is
    fine because each crop has exactly one label (classification task, not detection).

    Returns:
        (train_df, test_df, report) where report has keys: n_parents, n_train_parents,
        n_test_parents, dropped_classes (list), n_dropped_classes, n_classes_kept.
    """
    if "image_path" not in df.columns:
        raise ValueError("df must have 'image_path' column")
    df = df.copy()
    df["parent_image"] = df["image_path"].map(_crop_path_to_parent_stem)
    n_parents = df["parent_image"].nunique()

    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    kept_classes: list[str] = []
    dropped_classes: list[str] = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        class_parents = np.array(sorted(class_df["parent_image"].unique()))
        shuffled = rng.permutation(class_parents)

        # Accumulate test parents until we meet both the fraction target and
        # the min_test_images crop threshold (whichever requires more parents).
        n_target = max(1, int(len(class_parents) * test_size))
        test_p: list[str] = []
        test_crop_count = 0
        for p in shuffled:
            if test_crop_count >= min_test_images and len(test_p) >= n_target:
                break
            test_p.append(p)
            test_crop_count += int((class_df["parent_image"] == p).sum())

        test_p_set = set(test_p)
        train_p_set = set(class_parents) - test_p_set

        test_class = class_df[class_df["parent_image"].isin(test_p_set)]
        train_class = class_df[class_df["parent_image"].isin(train_p_set)]

        if max_test_images and len(test_class) > max_test_images:
            test_class = test_class.sample(n=max_test_images, random_state=int(rng.integers(0, 2**31)))

        if len(test_class) >= min_test_images and len(train_class) >= 1:
            kept_classes.append(label)
            train_parts.append(train_class)
            test_parts.append(test_class)
        else:
            dropped_classes.append(label)
            print(
                f"  [split] drop '{label}': {len(test_class)} test crops "
                f"(need {min_test_images}), {len(train_class)} train crops"
            )

    train_df = (pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[:0]).drop(columns=["parent_image"])
    test_df = (pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[:0]).drop(columns=["parent_image"])

    report = {
        "n_parents": n_parents,
        "n_train_parents": train_df["image_path"].map(_crop_path_to_parent_stem).nunique() if not train_df.empty else 0,
        "n_test_parents": test_df["image_path"].map(_crop_path_to_parent_stem).nunique() if not test_df.empty else 0,
        "dropped_classes": dropped_classes,
        "n_dropped_classes": len(dropped_classes),
        "n_classes_kept": len(kept_classes),
    }
    return train_df, test_df, report


def normalize_label(l):
    if pd.isna(l):
        return l
    s = str(l).strip()
    s = s.replace("/", " ")
    s = " ".join(s.split()[:2])
    return s


@hydra.main(config_path=str(PROJECT_ROOT / "boem_conf"), config_name="boem_config")
def main(cfg: DictConfig):
    # Override the classification_model config with USGS.yaml, and forward CLI overrides (e.g. classification_model.expand=200)
    overrides = ["classification_model=USGS"] + list(HydraConfig.get().overrides.task)
    cfg = hydra.compose(config_name="boem_config", overrides=overrides)

    # From the detection script: use only per-image CSVs (exclude train/test/zero_shot) to avoid
    # duplicates and Object vs species label conflicts (prepare_USGS sets label=Object in combined splits).
    ubfai_crops_dir = "/blue/ewhite/b.weinstein/BOEM/training/crops"
    all_csvs = glob.glob(os.path.join(ubfai_crops_dir, "*.csv"))
    per_image_csvs = [
        x for x in all_csvs
        if os.path.basename(x) not in UBFAI_CROPS_EXCLUDE_CSV
        and not os.path.basename(x).startswith(UBFAI_CROPS_EXCLUDE_PREFIX)
    ]
    n_excluded = len(all_csvs) - len(per_image_csvs)
    if n_excluded:
        print(f"[annotation pool] Loading {len(per_image_csvs)} per-image CSVs (excluded {n_excluded} combined splits: train/test/zero_shot, train_max_empty_*)")
    if not per_image_csvs:
        raise ValueError(
            f"No per-image CSVs found in {ubfai_crops_dir}. "
            "Excluded combined splits: train.csv, test.csv, zero_shot.csv, train_max_empty_*.csv"
        )
    crop_annotations = [pd.read_csv(x) for x in per_image_csvs]
    crop_annotations = pd.concat(crop_annotations)

    # Drop exact duplicate annotation rows (same image, box, label)
    dup_subset = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    if all(col in crop_annotations.columns for col in dup_subset):
        dup_mask = crop_annotations.duplicated(subset=dup_subset, keep="first")
        n_dup = int(dup_mask.sum())
        if n_dup:
            print(f"[dedup] Dropping {n_dup} exact duplicate rows (image_path+xmin+ymin+xmax+ymax+label)")
            crop_annotations = crop_annotations.loc[~dup_mask]

        # Report boxes that have conflicting labels so they can be inspected
        box_cols = ["image_path", "xmin", "ymin", "xmax", "ymax"]
        label_counts = (
            crop_annotations.groupby(box_cols)["label"].nunique()
            if all(col in crop_annotations.columns for col in box_cols)
            else None
        )
        if label_counts is not None:
            n_conflict = int((label_counts > 1).sum())
            print(f"[dedup] Boxes with conflicting labels (same image+box, multiple labels): {n_conflict}")

    # Keep labels with more than 25 images
    crop_annotations = crop_annotations.groupby("label").filter(lambda x: len(x) > 25)

    # Only keep two word labels
    crop_annotations = crop_annotations[crop_annotations["label"].str.contains(" ", na=False)]
    crop_annotations = crop_annotations[~crop_annotations.label.isin([0,"0","FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
    
    crop_annotations["label"] = crop_annotations["label"].apply(normalize_label)

    # Two word labels
    crop_annotations["label"] = crop_annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
    crop_annotations = crop_annotations[crop_annotations["label"].apply(lambda x: len(x.split()) == 2)]

    # Remove any crop_annotations with empty boxes
    crop_annotations = crop_annotations[(crop_annotations['xmin'] != 0) & (crop_annotations['ymin'] != 0) & (crop_annotations['xmax'] != 0) & (crop_annotations['ymax'] != 0)]

    # Remove any negative values
    crop_annotations = crop_annotations[(crop_annotations['xmin'] >= 0) & (crop_annotations['ymin'] >= 0) & (crop_annotations['xmax'] >= 0) & (crop_annotations['ymax'] >= 0)]

    # Split per class by parent image before balancing so rare classes are not harmed.
    # Each class independently gets >= min_test_images crops in test; everything else
    # goes to train. Classes that cannot meet the threshold are dropped.
    if "image_path" not in crop_annotations.columns:
        raise ValueError("Crop annotations must have 'image_path' column for split-by-image")
    print(f"[split] pool: {len(crop_annotations)} crops, {crop_annotations['label'].nunique()} classes")
    train_df, validation_df, split_report = train_test_split_by_image(
        crop_annotations, test_size=0.1, min_test_images=5, max_test_images=100, random_state=42
    )
    print(
        f"[split by parent image] {split_report['n_parents']} parent images -> "
        f"{split_report['n_train_parents']} train / {split_report['n_test_parents']} test"
    )
    print(
        f"[split by parent image] kept {split_report['n_classes_kept']} classes, "
        f"dropped {split_report['n_dropped_classes']} classes (insufficient train/test after image split)"
    )

    # Gentle class balance: applied to train only after the split so test reflects the
    # natural distribution and rare classes are not starved of their test images.
    n_before = len(train_df)
    train_df = gentle_class_balance(train_df, factor=4.0, random_state=42)
    n_after = len(train_df)
    if n_before > n_after:
        print(f"[class balance] train downsampled {n_before} -> {n_after} (cap = 4x median per class)")

    comet_logger = CometLogger(project_name=cfg.comet.project, workspace=cfg.comet.workspace)
    comet_logger.experiment.log_parameter("class_balance_applied", True)
    comet_logger.experiment.log_parameter("class_balance_cap_factor", 4.0)
    comet_logger.experiment.log_parameter("train_pool_size_after_balance", n_after)
    comet_logger.experiment.log_parameter("split_by_parent_image", True)
    comet_logger.experiment.log_parameter("split_n_classes_dropped", split_report["n_dropped_classes"])
    comet_logger.experiment.log_parameter("split_n_classes_kept", split_report["n_classes_kept"])

    # Log train and val dataframes to comet
    train_csv_path = "/tmp/train_annotations.csv"
    val_csv_path = "/tmp/val_annotations.csv"
    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(val_csv_path, index=False)
    comet_logger.experiment.log_parameters(cfg.classification_model)
    comet_logger.experiment.log_parameter("num_classes", len(train_df['label'].unique()))
    comet_logger.experiment.log_table("train_annotations.csv", train_df)
    comet_logger.experiment.log_table("val_annotations.csv", validation_df)

    comet_logger.experiment.add_tag("classification")
    comet_id = comet_logger.experiment.id

    # Buffer (expand) in pixels around each bbox when extracting classification crops.
    # Different buffer sizes save to different dirs so runs do not overwrite each other.
    expand_pixels = int(cfg.classification_model.get("expand", 30))
    buffer_suffix = f"buffer_{expand_pixels}"
    comet_logger.experiment.log_parameter("classification_bbox_expand_pixels", expand_pixels)

    # Output dirs are buffer-specific: .../buffer_0/, .../buffer_30/, .../buffer_200/, etc.
    train_crop_image_dir = os.path.join(
        cfg.classification_model.train_crop_image_dir, buffer_suffix, comet_id
    )
    os.makedirs(train_crop_image_dir, exist_ok=True)

    val_crop_image_dir = os.path.join(
        cfg.classification_model.val_crop_image_dir, buffer_suffix, comet_id
    )
    os.makedirs(val_crop_image_dir, exist_ok=True)

    checkpoint_dir = os.path.join(cfg.classification_model.checkpoint_dir, buffer_suffix)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save train/val split for direct comparability with hierarchical model
    split_dir = os.path.join(checkpoint_dir, comet_id)
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "usgs_train_split.csv"), index=False)
    validation_df.to_csv(os.path.join(split_dir, "usgs_val_split.csv"), index=False)
    print(f"[split] saved train/val CSVs to {split_dir} for hierarchical comparability")

    balance_classes = bool(cfg.classification_model.get("balance_classes", False))
    comet_logger.experiment.log_parameter("balance_classes", balance_classes)

    use_metadata = bool(cfg.classification_model.get("use_metadata", False))
    metadata_dir = cfg.classification_model.get("metadata_dir", None) or cfg.report.get("metadata_dir", None)
    metadata_dim = int(cfg.classification_model.get("metadata_dim", 32))
    metadata_dropout = float(cfg.classification_model.get("metadata_dropout", 0.5))
    comet_logger.experiment.log_parameter("use_metadata", use_metadata)
    if use_metadata:
        comet_logger.experiment.log_parameter("metadata_dir", metadata_dir)
        comet_logger.experiment.log_parameter("metadata_dim", metadata_dim)
        comet_logger.experiment.log_parameter("metadata_dropout", metadata_dropout)
        # UBFAI crops span many flights in one directory. Extract per-row flight_name
        # from the T{YYYYMMDD}_{HHMMSS} pattern in new-format image paths so
        # build_crop_metadata_rows can look up the right captures CSV for each image.
        # Old-format images (no T-datetime pattern) get None and are skipped silently.
        for df in (train_df, validation_df):
            df["flight_name"] = df["image_path"].map(_flight_name_from_image_path)
        n_with = int(train_df["flight_name"].notna().sum())
        n_without = int(train_df["flight_name"].isna().sum())
        print(f"[metadata] flight_name extracted: {n_with} train crops matched, {n_without} unmatched (old-format, will be skipped)")

    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        comet_logger=comet_logger,
        checkpoint=cfg.classification_model.checkpoint,
        checkpoint_dir=checkpoint_dir,
        image_dir=cfg.classification_model.image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        expand_pixels=expand_pixels,
        fast_dev_run=cfg.classification_model.fast_dev_run,
        max_epochs=cfg.classification_model.max_epochs,
        lr=cfg.classification_model.lr,
        batch_size=cfg.classification_model.batch_size,
        workers=cfg.classification_model.workers,
        balance_classes=balance_classes,
        use_metadata=use_metadata,
        metadata_dir=metadata_dir,
        metadata_dim=metadata_dim,
        metadata_dropout=metadata_dropout,
    )
    

    # Log the label dictionary
    comet_logger.experiment.log_parameter("label_dict", trained_model.label_dict)

    # Log the numeric to label dictionary
    comet_logger.experiment.log_parameter("numeric_to_label_dict", trained_model.numeric_to_label_dict)

    trained_model.trainer.save_checkpoint(os.path.join(checkpoint_dir, f"{comet_id}.ckpt"))

    validation_results = trained_model.trainer.validate(trained_model)

    print(validation_results)

    image_dataset, true_label, predicted_label = trained_model.val_dataset_confusion(return_images=True)
    comet_logger.experiment.log_confusion_matrix(
            y_true=true_label,
            y_predicted=predicted_label,
            images=image_dataset,
            max_categories=len(trained_model.label_dict.keys()),
            labels=list(trained_model.label_dict.keys()),
        )

if __name__ == "__main__":
    main()
