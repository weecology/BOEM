import pytest
from src.active_learning import choose_train_images, choose_test_images

@pytest.fixture
def performance():
    return {"detection": {"mAP":{"map":0.9}}, "confident_classification": {"accuracy": 0.8}}

@pytest.fixture
def detection_model():
    from deepforest import main
    return main.deepforest()

@pytest.fixture
def classification_model():
    from deepforest.model import CropModel
    return CropModel()

def test_choose_train_images(performance, detection_model, config):
    train_images_to_annotate = choose_train_images(
        evaluation=performance,
        image_dir=config.active_learning.image_dir,
        model=detection_model,
        strategy=config.active_learning.strategy,
        n=config.active_learning.n_images,
        patch_size=config.active_learning.patch_size,
        patch_overlap=config.active_learning.patch_overlap,
        min_score=config.active_learning.min_score
        )
    assert len(train_images_to_annotate) > 0

def test_choose_test_images(detection_model, config):
    test_images_to_annotate = choose_test_images(
        image_dir=config.active_testing.image_dir,
        model=detection_model,
        strategy=config.active_testing.strategy,
        n=config.active_testing.n_images,
        patch_size=config.active_testing.patch_size,
        patch_overlap=config.active_testing.patch_overlap,
        min_score=config.active_testing.min_score)
    assert len(test_images_to_annotate) > 0
