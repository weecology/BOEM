# Standard library imports
import os
import shutil
from typing import Generator

# Third party imports
import pandas as pd
import pytest
from hydra import initialize, compose

# Local imports
from src import label_studio

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

def get_api_key():
    """Get Label Studio API key from config file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.label_studio.config')
    if not os.path.exists(config_path):
        return None
        
    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('api_key'):
                return line.split('=')[1].strip()
    return None

@pytest.fixture(scope="session")
def config(tmpdir_factory):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")
    
    cfg.train.train_csv_folder = tmpdir_factory.mktemp("csvs").strpath
    cfg.train.train_image_dir = tmpdir_factory.mktemp("images").strpath
    cfg.train.crop_image_dir = tmpdir_factory.mktemp("crops").strpath
    
    # Put images from tests/data into the image directory
    for f in os.listdir("tests/data/"):
        if f != '.DS_Store':
            shutil.copy("tests/data/" + f, cfg.train.train_image_dir)

    # Create sample bounding box annotations
    data = {
        'image_path': ['empty.jpg', 'birds.jpg', 'birds_val.jpg'],
        'xmin': [0, 200, 150],
        'ymin': [0, 300, 250], 
        'xmax': [0, 300, 250],
        'ymax': [0, 400, 350],
        'label': ['Bird', 'Bird', 'Bird'],
        'annotator': ['test_user', 'test_user', 'test_user']
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV in the configured training directory
    csv_path = os.path.join(cfg.train.train_csv_folder, 'training_data.csv')
    df.to_csv(csv_path, index=False)

    cfg.train.fast_dev_run = True
    cfg.checkpoint = "bird"
    cfg.train.checkpoint_dir = tmpdir_factory.mktemp("checkpoints").strpath

    # Create detection annotations
    cfg.pipeline_evaluation.detect_ground_truth_dir = tmpdir_factory.mktemp("detection_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.detect_ground_truth_dir, 'detection_annotations.csv')
    df.to_csv(csv_path, index=False)

    # Create classification annotations
    cfg.pipeline_evaluation.classify_confident_ground_truth_dir = tmpdir_factory.mktemp("confident_classification_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.classify_confident_ground_truth_dir, 'confident_classification_annotations.csv')
    df.to_csv(csv_path, index=False)

    cfg.pipeline_evaluation.classify_uncertain_ground_truth_dir = tmpdir_factory.mktemp("uncertain_classification_annotations").strpath
    csv_path = os.path.join(cfg.pipeline_evaluation.classify_uncertain_ground_truth_dir, 'uncertain_classification_annotations.csv')
    df.to_csv(csv_path, index=False)
    
    return cfg
    
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.fixture(scope="session")
def label_studio_client(config):
    """Initialize Label Studio client with API key from .comet.config"""
    api_key = get_api_key()
    if api_key is None:
        print("Warning: No Label Studio API key found in .comet.config")
        return None

    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    label_config = """<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
    <RectangleLabels name="label" toName="image">
        <Label value="Bird" background="#9efdff"/>
    </RectangleLabels>
    </View>
    """
    
    try:
        ls = label_studio.connect_to_label_studio(
            url=config.label_studio.url, 
            project_name="test_BOEM", 
            label_config=label_config
        )

        sftp_client = label_studio.create_sftp_client(user=config.server.user, host=config.server.host, key_filename=config.server.key_filename)
        images = ["tests/data/" + f for f in os.listdir("tests/data/")]
        # Filter for only jpg files
        images = [img for img in images if img.lower().endswith('.jpg')]

        label_studio.upload_to_label_studio(
            images=images,
            sftp_client=sftp_client,
            label_studio_project=ls,
            images_to_annotate_dir="tests/data",
            folder_name=config.label_studio.folder_name,
            preannotations=None
        )

        return ls
    except Exception as e:
        print(f"Warning: Failed to initialize Label Studio client: {str(e)}")
        return None

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.fixture(scope="session", autouse=True)
def cleanup_label_studio(label_studio_client, request) -> Generator:
    """
    Fixture that runs after all tests are completed to clean up Label Studio projects.
    This fixture has session scope and runs automatically.
    """
    # Setup: yield to allow tests to run
    yield