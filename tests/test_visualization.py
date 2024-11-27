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


def test_visualizer_initialization(mock_model, tmp_path, test_predictions):
    """Test PredictionVisualizer initialization."""
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    assert visualizer.output_dir == tmp_path
    assert visualizer.fps == 30

def test_draw_predictions(mock_model, tmp_path, test_image, test_predictions):
    """Test drawing predictions on image."""
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    result = visualizer.draw_predictions(test_image, test_predictions)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == test_image.shape
    # Image should be different from original due to drawn boxes
    assert not np.array_equal(result, test_image)

def test_create_visualization(mock_model, tmp_path, test_predictions):
    """Test video creation from image sequence."""
    # Create test images
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    for i in range(5):
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(image_dir / f"image_{i:03d}.jpg"), img)

    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    output_path = visualizer.create_visualization(list(image_dir.glob("*.jpg")))
    
    assert Path(output_path).exists()
    assert output_path.endswith('.mp4')

def test_create_summary_image(mock_model, tmp_path, test_predictions):
    """Test creation of summary statistics image."""
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    
    predictions_list = [
        pd.DataFrame({
            'label': ['Bird', 'Bird'],
            'score': [0.9, 0.8]
        }),
        pd.DataFrame({
            'label': ['Bird'],
            'score': [0.95]
        })
    ]
    
    summary = visualizer.create_summary_image(predictions_list)
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (600, 800, 3)

def test_empty_image_dir(mock_model, tmp_path, test_predictions):
    """Test handling of empty image directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    with pytest.raises(ValueError, match="No images found"):
        visualizer.create_visualization(list(empty_dir.glob("*.jpg")))

def test_invalid_image(mock_model, tmp_path):
    """Test handling of invalid image file."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create invalid image file
    (image_dir / "invalid.jpg").write_text("not an image")
    
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    with pytest.raises(ValueError, match="Could not read first image"):
        visualizer.create_visualization(list(image_dir.glob("*.jpg")))

@pytest.mark.parametrize("confidence_threshold", [0.3, 0.7, 0.9])
def test_confidence_thresholds(mock_model, tmp_path, test_image, test_predictions, confidence_threshold):
    """Test different confidence thresholds."""
    visualizer = PredictionVisualizer(test_predictions, tmp_path)
    result = visualizer.draw_predictions(
        test_image,
        test_predictions,
        confidence_threshold=confidence_threshold   
    )
    assert isinstance(result, np.ndarray) 