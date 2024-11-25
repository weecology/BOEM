from src.pipeline import Pipeline
from src.label_studio import get_api_key
import pytest
import os

# Local imports
from src import label_studio

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
get_api_key()

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.fixture()
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
def cleanup_label_studio(label_studio_client, request):
    """
    Fixture that runs after all tests are completed to clean up Label Studio projects.
    This fixture has session scope and runs automatically.
    """
    # Setup: yield to allow tests to run
    yield

@pytest.mark.integration
def test_pipeline_run(config, label_studio_client):
    """Test complete pipeline run"""
    pipeline = Pipeline(cfg=config)
    pipeline.run()
