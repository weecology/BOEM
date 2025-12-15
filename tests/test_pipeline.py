from src.pipeline import Pipeline
from src.label_studio import get_api_key
import pytest
import os
import torch
from tests.conftest import config

# Local imports
from src import label_studio

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
get_api_key()

@pytest.fixture()
def label_studio_client(config):
    """Initialize Label Studio client with API key from .comet.config"""
    if IN_GITHUB_ACTIONS:
        pytest.skip("Test doesn't work in Github Actions.")
    
    api_key = get_api_key()
    if api_key is None:
        pytest.skip("No Label Studio API key found in .comet.config")

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
        pytest.skip(f"Failed to initialize Label Studio client: {str(e)}")

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_pipeline_run(config, label_studio_client):
    """Integration test for full pipeline run."""
    pytest.skip("Integration test requires Hydra config wiring; skipping in unit test run")