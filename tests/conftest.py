# Standard library imports
import os
import shutil

# Third party imports
import pandas as pd
import pytest
from hydra import initialize, compose

@pytest.fixture(scope="session")
def config(tmpdir_factory):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")

    # Detection model
    cfg.detection_model.train_csv_folder = tmpdir_factory.mktemp(
        "csvs").strpath
    cfg.detection_model.train_image_dir = tmpdir_factory.mktemp(
        "images").strpath
    cfg.detection_model.crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    cfg.pipeline_evaluation.image_dir = cfg.detection_model.train_image_dir

    # Classification model
    cfg.classification_model.train_csv_folder = tmpdir_factory.mktemp(
        "csvs").strpath
    cfg.classification_model.train_image_dir = tmpdir_factory.mktemp(
        "images").strpath
    cfg.classification_model.crop_image_dir = tmpdir_factory.mktemp(
        "crops").strpath
    cfg.classification_model.checkpoint_dir = tmpdir_factory.mktemp(
        "checkpoints").strpath


    # Put images from tests/data into the image directory
    for f in os.listdir("tests/data/"):
        if f != '.DS_Store':
            shutil.copy("tests/data/" + f, cfg.detection_model.train_image_dir)
            shutil.copy("tests/data/" + f, cfg.classification_model.train_image_dir)
    # Create sample bounding box annotations
    train_data = {
        'image_path': ['empty.jpg', 'birds.jpg', "birds.jpg"],
        'xmin': [0, 200, 150],
        'ymin': [0, 300, 250],
        'xmax': [20, 300, 250],
        'ymax': [20, 400, 350],
        'label': ['Bird', 'Bird1', 'Bird2'],
        'annotator': ['test_user', 'test_user', 'test_user']
    }

    val_data = {
        'image_path': ['birds_val.jpg', 'birds_val.jpg'],
        'xmin': [150, 150],
        'ymin': [250, 250],
        'xmax': [250, 250],
        'ymax': [350, 350],
        'label': ['Bird1', 'Bird2'],
        'annotator': ['test_user', 'test_user'],
        "score": [0.9, 0.8]
    }

    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Save training data to CSV
    train_csv_path = os.path.join(cfg.detection_model.train_csv_folder,
                                  'training_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Save validation data to CSV
    val_csv_path = os.path.join(cfg.detection_model.train_csv_folder,
                                'validation.csv')
    val_df.to_csv(val_csv_path, index=False)

    # Save training data to CSV
    train_csv_path = os.path.join(cfg.classification_model.train_csv_folder,
                                  'training_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Save validation data to CSV
    val_csv_path = os.path.join(cfg.classification_model.train_csv_folder,
                                'validation.csv')
    val_df.to_csv(val_csv_path, index=False)

    cfg.detection_model.validation_csv_path = val_csv_path
    cfg.detection_model.fast_dev_run = True
    cfg.classification_model.fast_dev_run = True
    cfg.detection_model.checkpoint = "bird"
    cfg.detection_model.checkpoint_dir = tmpdir_factory.mktemp(
        "checkpoints").strpath

    # Create detection annotations
    cfg.pipeline_evaluation.detect_ground_truth_dir = tmpdir_factory.mktemp(
        "detection_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.detect_ground_truth_dir,
                            'detection_annotations.csv')
    val_df.to_csv(csv_path, index=False)

    # Create classification annotations
    cfg.pipeline_evaluation.classify_confident_ground_truth_dir = tmpdir_factory.mktemp(
        "confident_classification_annotations").strpath
    csv_path = os.path.join(
        cfg.pipeline_evaluation.classify_confident_ground_truth_dir,
        'confident_classification_annotations.csv')
    val_df.to_csv(csv_path, index=False)

    cfg.pipeline_evaluation.classify_uncertain_ground_truth_dir = tmpdir_factory.mktemp(
        "uncertain_classification_annotations").strpath
    csv_path = os.path.join(
        cfg.pipeline_evaluation.classify_uncertain_ground_truth_dir,
        'uncertain_classification_annotations.csv')
    val_df.to_csv(csv_path, index=False)

    # Active learning
    cfg.active_learning.image_dir = cfg.detection_model.train_image_dir
    cfg.active_testing.image_dir = cfg.detection_model.train_image_dir
    cfg.active_learning.n_images = 1
    cfg.active_testing.n_images = 1
    
    # Reporting
    cfg.reporting.image_dir = cfg.detection_model.train_image_dir

    return cfg
