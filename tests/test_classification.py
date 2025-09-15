import os
import pandas as pd
import pytest
from src.classification import preprocess_images, train, preprocess_and_train
from deepforest.model import CropModel

# Create sample annotations DataFrame
@pytest.fixture
def sample_annotations():
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["genus species1", "genus species2"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60]
    }
    return pd.DataFrame(data)

# Create sample model
@pytest.fixture
def sample_model():
    return CropModel(num_classes=2)

# Test preprocess_images function
def test_preprocess_images(sample_model, sample_annotations, tmp_path):
    root_dir = "tests/data/"
    save_dir = tmp_path / "crops"
    os.makedirs(save_dir, exist_ok=True)
    preprocess_images(sample_model, sample_annotations, root_dir, save_dir)
    assert os.path.exists(save_dir)

# Minimal end-to-end via preprocess_and_train only
def test_preprocess_and_train(sample_annotations, tmp_path):
    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint = None
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    assert model is not None

# Minimal checkpoint reuse
def test_preprocess_and_train_with_checkpoint(sample_annotations, tmp_path):
    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=None,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved = os.path.join(checkpoint_dir, "tmp.ckpt")
    model.trainer.save_checkpoint(saved)

    model2 = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=saved,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    assert model2 is not None
