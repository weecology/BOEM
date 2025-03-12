import pytest
from src.active_learning import generate_pool_predictions, select_images
from deepforest.model import CropModel
import os
import pandas as pd
import numpy as np
from deepforest import main
from deepforest.utilities import read_file

@pytest.fixture
def performance():
    return {"detection": {"mAP":{"map":0.9}}, "confident_classification": {"accuracy": 0.8}}

@pytest.fixture
def detection_model(comet_logger):
    """Create a mock deepforest model that produces bounding box predictions."""
    class MockDeepForest(main.deepforest):
        def __init__(self, label_dict, random=True):
            super().__init__(label_dict=label_dict, num_classes=len(label_dict))
            self.random = random
            self.comet_logger = comet_logger

        def predict_tile(self, raster_path, patch_size=450, patch_overlap=0, return_plot=False, crop_model=None):
            # Return realistic predictions based on image name
            if "empty" in raster_path:
                return None
                
            num_predictions = np.random.randint(1, 4)
            df = pd.DataFrame({
                    'xmin': np.random.randint(0, 800, num_predictions),
                    'ymin': np.random.randint(0, 600, num_predictions),
                    'xmax': np.random.randint(800, 1000, num_predictions),
                    'ymax': np.random.randint(600, 800, num_predictions),
                    'label': ['Object'] * num_predictions,
                    'cropmodel_label': [0] * num_predictions,
                    'score': np.random.uniform(0.1, 0.99, num_predictions),
                    'image_path': [os.path.basename(raster_path)] * num_predictions
                })
            df = read_file(df)
            
            return df

              
    return MockDeepForest(label_dict={"Object": 0})

@pytest.fixture
def random_crop_model():
    m = CropModel()
    m.label_dict = {"Bird": 0,"Mammal":1}
    return m


def test_generate_train_image_pool(detection_model):
    train_image_pool = generate_pool_predictions(
        image_dir="tests/data",
        model=detection_model,
        patch_size=450,
        patch_overlap=0,
        min_score=0,
    )
    assert len(train_image_pool) > 0

def test_select_train_images(detection_model):
    train_image_pool = generate_pool_predictions(
        image_dir="tests/data",
        patch_size=450,
        model=detection_model,
        patch_overlap=0,
        min_score=0.5,
    )
    train_images_to_annotate = select_images(
        preannotations=train_image_pool,
        strategy='random',
        n=1
    )
    assert len(train_images_to_annotate) > 0
