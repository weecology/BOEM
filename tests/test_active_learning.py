import pytest
from src.active_learning import choose_test_images, generate_pool_predictions, select_images

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

def test_generate_train_image_pool(performance, detection_model, config):
    train_image_pool = generate_pool_predictions(
        evaluation=performance,
        image_dir=config.active_learning.image_dir,
        model=detection_model,
        patch_size=config.active_learning.patch_size,
        patch_overlap=config.active_learning.patch_overlap,
        min_score=config.active_learning.min_score,
        target_labels=None
    )
    assert len(train_image_pool) > 0

def test_select_train_images(performance, config):
    train_image_pool = generate_pool_predictions(
        evaluation=performance,
        image_dir=config.active_learning.image_dir,
        model=None,  # Assuming model is not needed for selection
        patch_size=config.active_learning.patch_size,
        patch_overlap=config.active_learning.patch_overlap,
        min_score=config.active_learning.min_score,
        target_labels=None
    )
    train_images_to_annotate = select_images(
        image_pool=train_image_pool,
        strategy='random',
        n=config.active_learning.n_images
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
