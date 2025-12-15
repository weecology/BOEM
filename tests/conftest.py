# Standard library imports
import os
import shutil
from pathlib import Path

# Third party imports
import pandas as pd
import pytest
from hydra import initialize, compose
from pytorch_lightning.loggers import CometLogger

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip loading .env

@pytest.fixture(scope="session")
def config(tmpdir_factory):
    # Use correct config path and name matching main.py
    with initialize(version_base=None, config_path="../boem_conf"):
        cfg = compose(config_name="boem_config", overrides=["classification_model=USGS"])

    # Label studio instances
    cfg.label_studio.instances.validation.csv_dir = tmpdir_factory.mktemp("validation_csvs").strpath
    cfg.label_studio.instances.train.csv_dir = tmpdir_factory.mktemp("train_csvs").strpath
    cfg.label_studio.instances.review.csv_dir = tmpdir_factory.mktemp("review_csvs").strpath

    # Detection model
    cfg.detection_model.crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.detection_model.trainer.train.fast_dev_run = True
    cfg.detection_model.checkpoint_dir = tmpdir_factory.mktemp("checkpoints").strpath
    cfg.detection_model.checkpoint = "bird"

    # Classification model
    cfg.classification_model.train_crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.classification_model.val_crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.classification_model.checkpoint_dir = tmpdir_factory.mktemp("checkpoints").strpath
    cfg.classification_model.fast_dev_run = True

    # Put images from tests/data into the image directory
    for f in os.listdir("tests/data/"):
        if f != '.DS_Store':
            shutil.copy("tests/data/" + f, cfg.label_studio.images_to_annotate_dir)

    # Create sample bounding box annotations
    train_data = {
        'image_path': ['empty.jpg', 'birds.jpg', "birds.jpg"],
        'xmin': [20, 200, 150],
        'ymin': [10, 300, 250],
        'xmax': [40, 250, 200],
        'ymax': [20, 350, 300],
        'label': ['Object', 'Object', 'Object'],
        'cropmodel_label': ['Bird', 'Bird', 'Mammal'],
        'annotator': ['test_user', 'test_user', 'test_user']
    }

    val_data = {
        'image_path': ['empty.jpg','birds_val.jpg', 'birds_val.jpg'],
        'xmin': [0,200, 150],
        'ymin': [0,300, 250],
        'xmax': [0,250, 200],
        'ymax': [0,350, 300],
        'label': ['Object','Object', 'Object'],
        'cropmodel_label': ['Bird', 'Bird', 'Mammal'],
        'annotator': ['test_user','test_user', 'test_user'],
    }

    metadata = {
        'unique_image':['birds_val', 'birds_val'],
        'camera_GUID': ['1234567890', '1234567890'],
        'flight_name': ['flight1', 'flight2'],
        'date': ['2024-01-01', '2024-01-01'],
        'lat': ['44.81369', '44.81369'],
        'long': ['-68.81369', '-68.81369']
    }

    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Save training data to CSV
    train_csv_path = os.path.join(cfg.label_studio.instances.train.csv_dir, 'training_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Save validation data to CSV
    val_csv_path = os.path.join(cfg.label_studio.instances.validation.csv_dir, 'validation.csv')
    val_df.to_csv(val_csv_path, index=False)

    # Save review data to CSV
    review_csv_path = os.path.join(cfg.label_studio.instances.review.csv_dir, 'review.csv')
    val_df.to_csv(review_csv_path, index=False)

    cfg.pipeline_evaluation.detection_true_positive_threshold = 0.4
   
    # Active learning
    cfg.active_learning.image_dir = cfg.label_studio.images_to_annotate_dir
    cfg.active_learning.n_images = 1
    cfg.active_testing.n_images = 1
    cfg.active_learning.min_score = 0.01

    # Labelstudio
    cfg.label_studio.instances.train.project_name = "test_BOEM"
    cfg.label_studio.instances.validation.project_name = "test_BOEM"
    
    cfg.check_annotations = False

    return cfg

@pytest.fixture(scope="session")
def comet_logger():
    return CometLogger(
        project_name="BOEM_debug"
    )
