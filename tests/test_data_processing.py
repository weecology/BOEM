import pytest
import pandas as pd
import numpy as np
import cv2
import os
from src.data_processing import density_cropping, adjust_coordinates, merge_crop_predictions

@pytest.fixture
def sample_predictions():
    """Create sample predictions with dense areas."""
    # Create two dense clusters and some noise
    cluster1 = pd.DataFrame({
        'xmin': np.random.uniform(100, 200, 5),
        'ymin': np.random.uniform(100, 200, 5),
        'xmax': np.random.uniform(200, 300, 5),
        'ymax': np.random.uniform(200, 300, 5),
        'label': ['Bird'] * 5,
        'score': np.random.uniform(0.8, 0.9, 5)
    })
    
    cluster2 = pd.DataFrame({
        'xmin': np.random.uniform(500, 600, 5),
        'ymin': np.random.uniform(500, 600, 5),
        'xmax': np.random.uniform(600, 700, 5),
        'ymax': np.random.uniform(600, 700, 5),
        'label': ['Bird'] * 5,
        'score': np.random.uniform(0.8, 0.9, 5)
    })
    
    noise = pd.DataFrame({
        'xmin': np.random.uniform(0, 1000, 3),
        'ymin': np.random.uniform(0, 1000, 3),
        'xmax': np.random.uniform(0, 1000, 3),
        'ymax': np.random.uniform(0, 1000, 3),
        'label': ['Bird'] * 3,
        'score': np.random.uniform(0.8, 0.9, 3)
    })
    
    return pd.concat([cluster1, cluster2, noise], ignore_index=True)

@pytest.fixture
def test_image(tmp_path):
    """Create a test image."""
    img_path = tmp_path / "test.jpg"
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

def test_density_cropping(sample_predictions, test_image):
    """Test density-based cropping."""
    result = density_cropping(
        predictions=sample_predictions,
        image_path=test_image,
        min_density=3,
        eps=100,
        min_samples=3
    )
    
    assert 'crops' in result
    assert 'clusters' in result
    assert len(result['crops']) > 0
    assert len(result['clusters']) == len(sample_predictions)
    
    # Check crop structure
    for crop in result['crops']:
        assert all(key in crop for key in ['xmin', 'ymin', 'xmax', 'ymax', 'path'])
        assert crop['num_detections'] >= 3
        assert os.path.exists(crop['path'])

def test_adjust_coordinates():
    """Test coordinate adjustment for crops."""
    predictions = pd.DataFrame({
        'xmin': [100, 200],
        'ymin': [100, 200],
        'xmax': [150, 250],
        'ymax': [150, 250]
    })
    
    crop_info = {
        'xmin': 50,
        'ymin': 50,
        'xmax': 300,
        'ymax': 300
    }
    
    adjusted = adjust_coordinates(predictions, crop_info)
    assert adjusted['xmin'].iloc[0] == 50  # 100 - 50
    assert adjusted['ymin'].iloc[0] == 50  # 100 - 50

def test_merge_crop_predictions():
    """Test merging predictions from multiple crops."""
    crops = [
        {'xmin': 100, 'ymin': 100, 'xmax': 300, 'ymax': 300},
        {'xmin': 500, 'ymin': 500, 'xmax': 700, 'ymax': 700}
    ]
    
    predictions = pd.DataFrame({
        'xmin': [50, 50, 50],
        'ymin': [50, 50, 50],
        'xmax': [100, 100, 100],
        'ymax': [100, 100, 100]
    })
    
    labels = [0, 0, 1]
    
    merged = merge_crop_predictions(crops, predictions, labels)
    assert len(merged) == len(predictions)
    assert merged['xmin'].iloc[0] == predictions['xmin'].iloc[0] + crops[0]['xmin']

def test_empty_predictions(test_image):
    """Test handling of empty predictions."""
    empty_preds = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax'])
    result = density_cropping(empty_preds, test_image)
    assert result['crops'] == []
    assert result['clusters'] == []

@pytest.mark.parametrize("min_density", [2, 3, 4])
def test_different_densities(sample_predictions, test_image, min_density):
    """Test different minimum density thresholds."""
    result = density_cropping(
        predictions=sample_predictions,
        image_path=test_image,
        min_density=min_density
    )
    
    for crop in result['crops']:
        assert crop['num_detections'] >= min_density 