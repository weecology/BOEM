import os

import pandas as pd
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def sample_annotations():
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["Object", "Object"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60],
    }
    return pd.DataFrame(data)


def test_detection_preprocess_and_train(sample_annotations, tmp_path):
    from omegaconf import DictConfig

    from src.detection import preprocess_and_train

    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint = None
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(tmp_path / "crops/val", exist_ok=True)

    trainer_config = DictConfig(
        {
            "train": {
                "fast_dev_run": True,
            }
        }
    )

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
        comet_logger=None,
    )

    assert trained_model is not None
