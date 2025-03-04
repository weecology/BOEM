import pytest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from src.visualization import PredictionVisualizer

@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable predictions."""
    class MockModel:
        def predict_image(self, image_path):
            return pd.DataFrame({
                'xmin': [100],
                'ymin': [100],
                'xmax': [200],
                'ymax': [200],
                'label': ['Bird'],
                'score': [0.9]
            })
    return MockModel()

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.ones((600, 800, 3), dtype=np.uint8) * 255

@pytest.fixture
def test_predictions():
    """Create test predictions."""
    return pd.DataFrame({
        'xmin': [100, 200],
        'ymin': [100, 200],
        'xmax': [200, 300],
        'ymax': [200, 300],
        'label': ['Bird', 'Bird'],
        'score': [0.9, 0.8],
        'image_path': ['image_1.jpg', 'image_2.jpg']
    })


def test_draw_predictions(mock_model, tmp_path, test_image, test_predictions):
    """Test drawing predictions on image."""
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    result = visualizer.draw_predictions(test_image, test_predictions)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == test_image.shape
    # Image should be different from original due to drawn boxes
    assert not np.array_equal(result, test_image)