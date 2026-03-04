import re
import numpy as np
import pandas as pd
import glob
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train
import hydra
from omegaconf import DictConfig
import os
from deepforest.model import CropModel

# Crop image_path is like C1_L6_F560_T20241219_173703_737_23.png -> parent stem is C1_L6_F560_T20241219_173703_737
_CROP_SUFFIX_RE = re.compile(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$")


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
    Split by parent image so no crop from the same image appears in both train and test.
    Adds a temporary column 'parent_image' (derived from image_path crop filename).
    Drops classes that have no test images or insufficient test/train after the split.

    Returns:
        (train_df, test_df, report) where report has keys: n_parents, n_train_parents,
        n_test_parents, dropped_classes (list), n_dropped_classes, n_classes_kept.
    """
    if "image_path" not in df.columns:
        raise ValueError("df must have 'image_path' column")
    df = df.copy()
    df["parent_image"] = df["image_path"].map(_crop_path_to_parent_stem)
    unique_parents = df["parent_image"].unique()
    n_parents = len(unique_parents)

    rng = np.random.default_rng(random_state)
    n_test_parents = max(1, int(n_parents * test_size))
    test_parents = set(rng.choice(unique_parents, size=n_test_parents, replace=False))
    train_parents = set(unique_parents) - test_parents

    train_df = df[df["parent_image"].isin(train_parents)].drop(columns=["parent_image"])
    test_df = df[df["parent_image"].isin(test_parents)].drop(columns=["parent_image"])

    kept_classes = []
    dropped_classes = []
    for label in df["label"].unique():
        train_count = (train_df["label"] == label).sum()
        test_count = (test_df["label"] == label).sum()
        if test_count >= min_test_images and train_count >= 1:
            kept_classes.append(label)
        else:
            dropped_classes.append(label)
    train_df = train_df[train_df["label"].isin(kept_classes)]
    test_df = test_df[test_df["label"].isin(kept_classes)]

    report = {
        "n_parents": n_parents,
        "n_train_parents": len(train_parents),
        "n_test_parents": len(test_parents),
        "dropped_classes": dropped_classes,
        "n_dropped_classes": len(dropped_classes),
        "n_classes_kept": len(kept_classes),
    }
    return train_df, test_df, report


@hydra.main(config_path="boem_conf", config_name="boem_config")
def main(cfg: DictConfig):
    # Override the classification_model config with USGS.yaml
    cfg = hydra.compose(config_name="boem_config", overrides=["classification_model=USGS"])
        
    # From the detection script
    crop_annotations = glob.glob("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/*.csv")
    crop_annotations = [pd.read_csv(x) for x in crop_annotations]
    crop_annotations = pd.concat(crop_annotations)
    
    # Keep labels with more than 25 images
    crop_annotations = crop_annotations.groupby("label").filter(lambda x: len(x) > 25)

    # Only keep two word labels
    crop_annotations = crop_annotations[crop_annotations["label"].str.contains(" ", na=False)]
    crop_annotations = crop_annotations[~crop_annotations.label.isin([0,"0","FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
    
    def normalize_label(l):
        if pd.isna(l):
            return l
        s = str(l).strip()
        s = s.replace("/", " ")
        s = " ".join(s.split()[:2])   # keep first two tokens if that is your convention
        return s
    
    crop_annotations["label"] = crop_annotations["label"].apply(normalize_label)

    # Two word labels
    crop_annotations["label"] = crop_annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
    crop_annotations = crop_annotations[crop_annotations["label"].apply(lambda x: len(x.split()) == 2)]

    # Remove any crop_annotations with empty boxes
    crop_annotations = crop_annotations[(crop_annotations['xmin'] != 0) & (crop_annotations['ymin'] != 0) & (crop_annotations['xmax'] != 0) & (crop_annotations['ymax'] != 0)]

    # Remove any negative values
    crop_annotations = crop_annotations[(crop_annotations['xmin'] >= 0) & (crop_annotations['ymin'] >= 0) & (crop_annotations['xmax'] >= 0) & (crop_annotations['ymax'] >= 0)]

    # Gentle class balance: cap each class at 3x median size to reduce majority-class dominance (e.g. larus argentinus)
    n_before = len(crop_annotations)
    crop_annotations = gentle_class_balance(crop_annotations, factor=3.0, random_state=42)
    n_after = len(crop_annotations)
    if n_before > n_after:
        print(f"[class balance] downsampled training pool {n_before} -> {n_after} (cap = 3x median per class)")

    if "image_path" not in crop_annotations.columns:
        raise ValueError("Crop annotations must have 'image_path' column for split-by-image")
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
    if split_report["dropped_classes"]:
        print("  Dropped classes:", split_report["dropped_classes"])

    comet_logger = CometLogger(project_name=cfg.comet.project, workspace=cfg.comet.workspace)
    comet_logger.experiment.log_parameter("class_balance_applied", True)
    comet_logger.experiment.log_parameter("class_balance_cap_factor", 3.0)
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

    # Add a experiment stamp to not use the same image_dir for different runs
    train_crop_image_dir = os.path.join(cfg.classification_model.train_crop_image_dir, comet_id)
    os.makedirs(train_crop_image_dir, exist_ok=True)

    val_crop_image_dir = os.path.join(cfg.classification_model.val_crop_image_dir, comet_id)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        comet_logger=comet_logger,
        checkpoint=cfg.classification_model.checkpoint,
        checkpoint_dir=cfg.classification_model.checkpoint_dir,
        image_dir=cfg.classification_model.image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        fast_dev_run=cfg.classification_model.fast_dev_run,
        max_epochs=cfg.classification_model.max_epochs,
        lr=cfg.classification_model.lr,
        batch_size=cfg.classification_model.batch_size,
        workers=cfg.classification_model.workers,
    )
    

    # Log the label dictionary
    comet_logger.experiment.log_parameter("label_dict", trained_model.label_dict)

    # Log the numeric to label dictionary
    comet_logger.experiment.log_parameter("numeric_to_label_dict", trained_model.numeric_to_label_dict)

    checkpoint_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints/"
    trained_model.trainer.save_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"))

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

    # Reload the model from checkpoint and validate
    trained_model = CropModel.load_from_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"))
    trained_model.create_trainer()
    validation_results = trained_model.trainer.validate(trained_model)
    print(validation_results)


if __name__ == "__main__":
    main()
