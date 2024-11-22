from src.pipeline_evaluation import PipelineEvaluation
from deepforest import main
from deepforest.model import CropModel
import pytest
import pandas as pd
import numpy as np
import os

@pytest.fixture
def mock_deepforest_model(config):
    """Create a mock deepforest model that produces bounding box predictions."""
    class MockDeepForest(main.deepforest):
        def __init__(self, label_dict, random=False):
            super().__init__(label_dict=label_dict, num_classes=len(label_dict))
            self.random = random
        def predict_tile(self, raster_path, patch_size=450, patch_overlap=0, return_plot=False, crop_model=None):
            # Return realistic predictions based on image name
            if "empty" in raster_path.lower():
                return pd.DataFrame({
                    'xmin': [],
                    'ymin': [],
                    'xmax': [],
                    'ymax': [],
                    'label': [],
                    'score': []
                })
                
            # If random, Generate 1-3 random predictions for non-empty images
            if self.random:
                num_predictions = np.random.randint(1, 4)
                return pd.DataFrame({
                        'xmin': np.random.randint(0, 800, num_predictions),
                        'ymin': np.random.randint(0, 600, num_predictions),
                        'xmax': np.random.randint(800, 1000, num_predictions),
                        'ymax': np.random.randint(600, 800, num_predictions),
                        'label': ['Bird1'] * num_predictions,
                        'score': np.random.uniform(0.5, 0.99, num_predictions),
                        'image_path': [os.path.basename(raster_path)] * num_predictions
                    })
            else:
                # Return the validation data for perfect performance
                val_csv_path = os.path.join(config.detection_model.train_csv_folder, 'validation.csv')
                validation_df = pd.read_csv(val_csv_path)
                # Add scores
                validation_df['score'] = 1.0

                return validation_df
              
    return MockDeepForest(label_dict={"Bird1": 1,"Bird2": 2})

@pytest.fixture
def random_model():
    m = main.deepforest(label_dict={"Bird1": 1,"Bird2": 2}, num_classes=2)
    return m

@pytest.fixture
def random_crop_model():
    m = CropModel()
    return m

def test_check_success(config, mock_deepforest_model, random_crop_model):
    """Test check success with mock model and perfect performance."""
    pipeline_evaluation = PipelineEvaluation(model=mock_deepforest_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    pipeline_evaluation.evaluate()
    assert pipeline_evaluation.check_success() is True

def test_evaluate_detection(config, mock_deepforest_model, random_crop_model):
    """Test evaluate detection with mock model."""
    # Cropmodel is mocked, it is not run
    pipeline_evaluation = PipelineEvaluation(model=mock_deepforest_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    detection_results = pipeline_evaluation.evaluate_detection()
    
    assert detection_results["mAP"]["map"] == 1.0

def test_confident_classification_accuracy(config, mock_deepforest_model, random_crop_model):
    """Test confident classification accuracy with mock model and perfect performance."""
    # Cropmodel is mocked, it is not run
    pipeline_evaluation = PipelineEvaluation(model=mock_deepforest_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    confident_classification_accuracy = pipeline_evaluation.evaluate_confident_classification()
   
    assert confident_classification_accuracy["confident_classification_accuracy"] == 1.0

def test_uncertain_classification_accuracy(config, mock_deepforest_model, random_crop_model):
    """Test uncertain classification accuracy with mock model and perfect performance."""
    # Cropmodel is mocked, it is not run
    pipeline_evaluation = PipelineEvaluation(model=mock_deepforest_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    uncertain_classification_accuracy = pipeline_evaluation.evaluate_uncertain_classification()
   
    assert uncertain_classification_accuracy["uncertain_classification_accuracy"] == 1.0

def test_evaluate(config, random_model, random_crop_model):
    """Test evaluate with mock model."""
    pipeline_evaluation = PipelineEvaluation(model=random_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    pipeline_evaluation.evaluate()

    # All the metrics should be undefined
    assert pipeline_evaluation.results["detection"]["mAP"]["map"] == -1
    assert pipeline_evaluation.results["confident_classification"]["confident_classification_accuracy"] == 0
    assert pipeline_evaluation.results["uncertain_classification"]["uncertain_classification_accuracy"] == 0