import os
import pandas as pd
import pytest
from src.detection import preprocess_and_train, load
from deepforest.model import CropModel
from omegaconf import DictConfig

# Create sample annotations DataFrame
@pytest.fixture
def sample_annotations():
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["Object", "Object"],
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

# Test detection preprocess_and_train function
def test_detection_preprocess_and_train(sample_annotations, tmp_path):
    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint = None
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)
    
    comet_logger = None
    
    trainer_config = DictConfig({
        "train": {
            "fast_dev_run": True,
        }
    })
    
    trained_model = preprocess_and_train(
        train_annotations=train_df,
        validation_annotations=validation_df,
        train_image_dir=train_image_dir,
        crop_image_dir=train_crop_image_dir,
        patch_size=256,
        patch_overlap=0.2,
        limit_empty_frac=0.5,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        trainer_config=trainer_config,
        comet_logger=comet_logger
    )
    
    # Check if the model is trained
    assert trained_model is not None
