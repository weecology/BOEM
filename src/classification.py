import os
import glob

import numpy as np
import rasterio
from PIL import Image
from deepforest.model import CropModel
import torch
import datetime
import pandas as pd

from src.spatiotemporal_metadata import build_crop_metadata_rows

def get_latest_checkpoint(checkpoint_dir, num_classes):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = CropModel.load_from_checkpoint(checkpoint)
            return m
        else:
            return None
    else:
        return None

def train(model, comet_logger=None, fast_dev_run=False, max_epochs=10, batch_size=4, workers=0, lr=0.0001):
    """Train a model on labeled images."""
    model.batch_size = batch_size
    model.num_workers = workers
    model.lr = lr

    devices = torch.cuda.device_count()
    accelerator = "gpu" if devices > 0 else "cpu"
    if devices == 0:
        devices = 1
    print(f"[train] Using accelerator={accelerator}, devices={devices}.")

    model.create_trainer(
        logger=comet_logger,
        fast_dev_run=fast_dev_run,
        max_epochs=max_epochs,
        num_nodes=1,
        devices=devices,
        accelerator=accelerator,
        enable_checkpointing=False,
    )
    model.trainer.fit(model)

    return model

def filter_annotations(crop_annotations):
    # Only keep two word labels
    crop_annotations = crop_annotations[crop_annotations["label"].str.contains(" ")]
    crop_annotations = crop_annotations[~crop_annotations.label.isin([0,"0","FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
    
    # Two word labels
    crop_annotations["label"] = crop_annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
    crop_annotations = crop_annotations[crop_annotations["label"].apply(lambda x: len(x.split()) == 2)]

    # Remove any crop_annotations with empty boxes
    crop_annotations = crop_annotations[(crop_annotations['xmin'] != 0) & (crop_annotations['ymin'] != 0) & (crop_annotations['xmax'] != 0) & (crop_annotations['ymax'] != 0)]

    # Remove any negative values
    crop_annotations = crop_annotations[(crop_annotations['xmin'] >= 0) & (crop_annotations['ymin'] >= 0) & (crop_annotations['xmax'] >= 0) & (crop_annotations['ymax'] >= 0)]
    
    return crop_annotations

def write_crops(model, boxes, root_dir, images, labels, savedir):
    """Write crops to disk using PIL to preserve RGB channel order.

    CropModel.write_crops uses cv2.imwrite which interprets the RGB array
    from rasterio as BGR, swapping the R and B channels in saved PNGs.
    This function avoids that by saving with PIL.
    """
    for label in set(labels):
        os.makedirs(os.path.join(savedir, label), exist_ok=True)

    for index, box in enumerate(boxes):
        label = labels[index]
        image = images[index]
        with rasterio.open(os.path.join(root_dir, image)) as src:
            square_box = model.expand_bbox_to_square(box, src.width, src.height)
            xmin, ymin, xmax, ymax = [int(v) for v in square_box]
            img = src.read(window=((ymin, ymax), (xmin, xmax)))
            img = np.rollaxis(img, 0, 3)  # (C, H, W) -> (H, W, C), RGB
            image_basename = os.path.splitext(os.path.basename(image))[0]
            img_path = os.path.join(savedir, label, f"{image_basename}_{index}.png")
            Image.fromarray(img).save(img_path)


def preprocess_images(model, annotations, root_dir, save_dir, expand_pixels: int = 30):
    """Extract crop regions from images using bbox + expand_pixels on each side.

    Each bounding box is expanded by expand_pixels on all sides before making
    the crop square (via model.expand_bbox_to_square). This buffer controls how
    much context surrounds the object in the classification crop.
    """
    boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    boxes = [
        [box[0] - expand_pixels, box[1] - expand_pixels, box[2] + expand_pixels, box[3] + expand_pixels]
        for box in boxes
    ]

    # Clamp to non-negative coordinates (expand_bbox_to_square will handle image bounds)
    boxes = [[max(0, b[0]), max(0, b[1]), max(0, b[2]), max(0, b[3])] for b in boxes]

    images = annotations["image_path"].values
    labels = annotations["label"].values

    write_crops(model=model, boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir)

    return annotations

def preprocess_and_train(
    train_df,
    validation_df,
    checkpoint,
    checkpoint_dir,
    image_dir,
    train_crop_image_dir,
    val_crop_image_dir,
    expand_pixels: int = 30,
    lr=0.0001,
    batch_size=4,
    fast_dev_run=False,
    max_epochs=10,
    workers=0,
    comet_logger=None,
    balance_classes=False,
    use_metadata=False,
    metadata_dir=None,
    flight_name=None,
    metadata_dim=32,
    metadata_dropout=0.5,
    **kwargs,
):
    """Preprocess data and train a crop model.

    expand_pixels: pixels to add on each side of every bounding box when
    extracting classification crops (default 30). Use 0 for tight crops,
    larger values (e.g. 100–200) for more context.
    """
    # Filter annotations
    train_df = filter_annotations(train_df)
    validation_df = filter_annotations(validation_df)

    if checkpoint is not None:
        loaded_model = CropModel.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        num_classes = len(pd.concat([train_df, validation_df]).label.unique())
        if use_metadata:
            try:
                loaded_model = CropModel(config_args={
                    "use_metadata": True,
                    "metadata_dim": metadata_dim,
                    "metadata_dropout": metadata_dropout,
                })
            except TypeError as exc:
                raise TypeError(
                    "classification_model.use_metadata requires DeepForest PR #1334 "
                    "or a DeepForest release with CropModel(config_args=...)."
                ) from exc
        else:
            loaded_model = CropModel()

    loaded_model.config["cropmodel"]["balance_classes"] = balance_classes
    if use_metadata and not loaded_model.config["cropmodel"].get("use_metadata", False):
        raise ValueError(
            "classification_model.use_metadata=True requires training from scratch "
            "or loading a metadata-enabled CropModel checkpoint."
        )
    loaded_model.create_trainer()

    # Preprocess train and validation data
    preprocessed_train = preprocess_images(
        model=loaded_model,
        annotations=train_df,
        root_dir=image_dir,
        save_dir=train_crop_image_dir,
        expand_pixels=expand_pixels,
    )

    preprocessed_validation = preprocess_images(
        model=loaded_model,
        annotations=validation_df,
        root_dir=image_dir,
        save_dir=val_crop_image_dir,
        expand_pixels=expand_pixels,
    )

    metadata_csv = None
    if use_metadata:
        if metadata_dir is None:
            raise ValueError(
                "classification_model.use_metadata=True requires report.metadata_dir "
                "or classification_model.metadata_dir"
            )
        if flight_name is None:
            flight_name = os.path.basename(os.path.normpath(image_dir))
        metadata_rows = pd.concat(
            [
                build_crop_metadata_rows(train_df, metadata_dir, flight_name),
                build_crop_metadata_rows(validation_df, metadata_dir, flight_name),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["filename"])
        if metadata_rows.empty:
            raise ValueError(f"No crop metadata rows were created for flight {flight_name}")
        metadata_csv = os.path.join(checkpoint_dir, "classification_crop_metadata.csv")
        os.makedirs(checkpoint_dir, exist_ok=True)
        metadata_rows.to_csv(metadata_csv, index=False)
        print(f"[preprocess_and_train] wrote crop metadata sidecar {metadata_csv}")

    load_kwargs = {
        "train_dir": str(train_crop_image_dir),
        "val_dir": str(val_crop_image_dir),
    }
    if metadata_csv is not None:
        load_kwargs["metadata_csv"] = metadata_csv
    try:
        loaded_model.load_from_disk(**load_kwargs)
    except TypeError as exc:
        if metadata_csv is not None:
            raise TypeError(
                "classification_model.use_metadata requires DeepForest PR #1334 "
                "or a DeepForest release with CropModel.load_from_disk(..., metadata_csv=...)."
            ) from exc
        raise

    # Assert that the label_dict, the train_ds classes and val_ds classes are the same
    assert loaded_model.label_dict == loaded_model.train_ds.class_to_idx
    assert loaded_model.label_dict == loaded_model.val_ds.class_to_idx
    
    trained_model = train(
        batch_size=batch_size,
        lr=lr,
        model=loaded_model,
        fast_dev_run=fast_dev_run,
        max_epochs=max_epochs,
        comet_logger=comet_logger,
        workers=workers,
    )

    print(f"[preprocess_and_train] saving model to checkpoint {checkpoint_dir}")
    if comet_logger is not None:
        model_name = comet_logger.experiment.id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"classification_model_{timestamp}"
    classification_checkpoint_path = save_model(trained_model, checkpoint_dir, model_name)
    if comet_logger is not None:
        comet_logger.experiment.log_asset(
            file_data=classification_checkpoint_path, file_name="classification_model.ckpt"
        )

    return trained_model

def save_model(model, directory, basename):
    checkpoint_path = os.path.join(directory, f"{basename}.ckpt")
    if not os.path.exists(checkpoint_path):
        model.trainer.save_checkpoint(checkpoint_path)

    return checkpoint_path
