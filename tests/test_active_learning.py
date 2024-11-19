import pytest
from src.active_learning import choose_train_images, choose_test_images

@pytest.fixture
def performance():
    return {"detection": 0.9, "classification": 0.8}

@pytest.fixture
def detection_model():
    from deepforest import main
    return main.deepforest()

@pytest.fixture
def classification_model():
    from deepforest.model import CropModel
    return CropModel()

def test_choose_train_images(performance, detection_model, config):
    train_images_to_annotate = choose_train_images(performance, detection_model, **config.active_learning)
    assert len(train_images_to_annotate) > 0

def test_choose_test_images(performance, detection_model, config):
    test_images_to_annotate = choose_test_images(performance, **config.active_testing)
    assert len(test_images_to_annotate) > 0