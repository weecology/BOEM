import os
import pandas as pd
import pytest
from src.classification import preprocess_images, train, load, preprocess_and_train
from deepforest.model import CropModel

# Create sample annotations DataFrame
@pytest.fixture
def sample_annotations():
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["label1", "label2"],
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
    
    # Check if cropped images are saved
    assert len(os.listdir(save_dir)) > 0

# Test train function
def test_train(sample_model, sample_annotations, tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Preprocess images
    preprocess_images(sample_model, sample_annotations, "tests/data/", train_dir)
    preprocess_images(sample_model, sample_annotations, "tests/data/", val_dir)
    
    # Train the model
    trained_model = train(
        model=sample_model,
        train_dir=train_dir,
        val_dir=val_dir,
        comet_logger=None,
        fast_dev_run=True,
        max_epochs=1,
        batch_size=2,
        workers=0,
        lr=0.0001
    )
    
    # Check if the model is trained
    assert trained_model is not None

# Test preprocess_and_train_classification function
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
    
    comet_logger = None
    
    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        train_image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=comet_logger,
        checkpoint_num_classes=None
    )
    
    # Check if the model is trained
    assert trained_model is not None

# Test preprocess_and_train_classification function with additional class
def test_preprocess_and_train_classification_with_additional_class(sample_annotations, tmp_path):
    
    # Train a 2 class model
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
    
    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        train_image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=comet_logger,
        checkpoint_num_classes=None
    )

    trained_model.trainer.save_checkpoint(f"{checkpoint_dir}/model.ckpt")
    checkpoint = f"{checkpoint_dir}/model.ckpt"

    # Add an additional class to the annotations and retrain the model
    additional_data = {
        "image_path": ["birds.jpg"],
        "label": ["label3"],
        "xmin": [30],
        "ymin": [30],
        "xmax": [70],
        "ymax": [70]
    }
    additional_annotations = pd.DataFrame(additional_data)
    train_df = pd.concat([sample_annotations, additional_annotations])
    validation_df = sample_annotations
    checkpoint = checkpoint
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)
    
    comet_logger = None
    
    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        train_image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=comet_logger,
        checkpoint_num_classes=2
    )
    
    # Check if the model is trained and has all classes
    assert trained_model is not None
    assert len(trained_model.label_dict) == 3
    assert "label3" in trained_model.label_dict
