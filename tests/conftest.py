# Standard library imports
import os
import shutil

# Third party imports
import pandas as pd
import pytest
from hydra import initialize, compose
from pytorch_lightning.loggers import CometLogger

@pytest.fixture(scope="session")
def config(tmpdir_factory):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["classification_model=USGS"])

    # Detection model
    cfg.detection_model.train_csv_folder = tmpdir_factory.mktemp("train_csvs").strpath
    cfg.label_studio.instances.validation.csv_dir = tmpdir_factory.mktemp("test_csvs").strpath
    cfg.detection_model.train_image_dir = tmpdir_factory.mktemp("images").strpath
    cfg.detection_model.crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.pipeline_evaluation.image_dir = cfg.detection_model.train_image_dir

    # Classification model
    cfg.classification_model.train_csv_folder = tmpdir_factory.mktemp("csvs").strpath
    cfg.classification_model.train_image_dir = tmpdir_factory.mktemp("images").strpath
    cfg.classification_model.crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.classification_model.checkpoint_dir = tmpdir_factory.mktemp("checkpoints").strpath

    # Put images from tests/data into the image directory
    for f in os.listdir("tests/data/"):
        if f != '.DS_Store':
            shutil.copy("tests/data/" + f, cfg.detection_model.train_image_dir)
            shutil.copy("tests/data/" + f, cfg.classification_model.train_image_dir)

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
    train_csv_path = os.path.join(cfg.detection_model.train_csv_folder, 'training_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Save validation data to CSV
    val_csv_path = os.path.join(cfg.label_studio.instances.validation.csv_dir, 'validation.csv')
    val_df.to_csv(val_csv_path, index=False)

    # Save classification training data to CSV
    train_csv_path = os.path.join(cfg.classification_model.train_csv_folder, 'training_data.csv')
    classification_train_df = train_df.copy(deep=True)
    classification_train_df['label'] = classification_train_df['cropmodel_label']
    classification_train_df.to_csv(train_csv_path, index=False)

    # Save classification validation data to CSV
    val_csv_path = os.path.join(cfg.classification_model.train_csv_folder, 'validation.csv')
    classification_val_df = val_df.copy(deep=True)
    classification_val_df['label'] = classification_val_df['cropmodel_label']
    classification_val_df.to_csv(val_csv_path, index=False)

    cfg.classification_model.trainer.fast_dev_run = True
    cfg.detection_model.checkpoint = "bird"
    cfg.detection_model.checkpoint_dir = tmpdir_factory.mktemp("checkpoints").strpath

    # Create detection annotations
    cfg.pipeline_evaluation.detect_ground_truth_dir = tmpdir_factory.mktemp("detection_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.detect_ground_truth_dir, 'detection_annotations.csv')
    val_df.to_csv(csv_path, index=False)

    cfg.label_studio.csv_dir_validation = cfg.pipeline_evaluation.detect_ground_truth_dir
    csv_path = os.path.join(cfg.pipeline_evaluation.detect_ground_truth_dir, 'validation.csv')
    val_df.to_csv(csv_path, index=False)

    # Create classification annotations
    cfg.pipeline_evaluation.classify_ground_truth_dir = tmpdir_factory.mktemp("classification_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.classify_ground_truth_dir, 'classification_annotations.csv')
    val_df.to_csv(csv_path, index=False)

    # Active learning
    cfg.active_learning.image_dir = cfg.detection_model.train_image_dir
    cfg.active_testing.image_dir = cfg.detection_model.train_image_dir
    cfg.active_learning.n_images = 1
    cfg.active_testing.n_images = 1
    cfg.active_learning.min_score = 0.01
    cfg.active_learning.gpus = 1

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
