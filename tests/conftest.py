import pytest
from src import label_studio
from typing import Generator
import os
from hydra import initialize, compose
import pandas as pd

def get_api_key():
    """Get Label Studio API key from .comet.config file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.comet.config')
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
    
    cfg["train"]["train_csv_folder"] = tmpdir_factory.mktemp("data").strpath

    # Create sample bounding box annotations
    data = {
        'image_path': ['sample_image_1.jpg', 'sample_image_1.jpg', 'sample_image_2.jpg'],
        'xmin': [100, 200, 150],
        'ymin': [100, 300, 250], 
        'xmax': [200, 300, 250],
        'ymax': [200, 400, 350],
        'label': ['Bird', 'Bird', 'Bird'],
        'annotator': ['test_user', 'test_user', 'test_user']
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV in the configured training directory
    csv_path = os.path.join(cfg.train.train_csv_folder, 'training_data.csv')
    df.to_csv(csv_path, index=False)

    return cfg
    
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

        # Only try to upload images if we have a valid client
        if os.path.exists("tests/data"):
            images = ["tests/data/" + f for f in os.listdir("tests/data/")]
            label_studio.upload_to_label_studio(images, ls, "test_BOEM", "tests/data")

        return ls
    except Exception as e:
        print(f"Warning: Failed to initialize Label Studio client: {str(e)}")
        return None

@pytest.fixture(scope="session", autouse=True)
def cleanup_label_studio(label_studio_client, request) -> Generator:
    """
    Fixture that runs after all tests are completed to clean up Label Studio projects.
    This fixture has session scope and runs automatically.
    """
    # Setup: yield to allow tests to run
    yield

    # Teardown: Clean up Label Studio projects only if we have a valid client
    def cleanup() -> None:
        if label_studio_client is None:
            return
            
        try:
            # Get all test projects
            projects = label_studio_client.get_projects()

            # Delete test projects
            for project in projects:
                if project.title.startswith('test_'):
                    label_studio_client.delete_project(project.id)
                    print(f"Cleaned up test project: {project.title}")
        except Exception as e:
            print(f"Warning: Failed to cleanup Label Studio projects: {str(e)}")

    request.addfinalizer(cleanup)