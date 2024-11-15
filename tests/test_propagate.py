import pytest
import pandas as pd
import numpy as np
from src.propagate import LabelPropagator
from datetime import datetime
import os

@pytest.fixture
def sample_annotations():
    """Create sample annotations for testing."""
    return pd.DataFrame({
        'image_path': [
            'IMG_20230615_123456.jpg',
            'IMG_20230615_123457.jpg',
            'IMG_20230615_123458.jpg',
            'IMG_20230615_123506.jpg'  # 10 seconds later
        ],
        'xmin': [100, 150, 200, 500],
        'ymin': [100, 150, 200, 500],
        'xmax': [120, 170, 220, 520],
        'ymax': [120, 170, 220, 520],
        'label': ['Bird', 'Bird', 'Bird', 'Bird']
    })

@pytest.fixture
def complex_annotations():
    """Create more complex annotations for testing edge cases."""
    return pd.DataFrame({
        'image_path': [
            'IMG_20230615_123456.jpg',
            'IMG_20230615_123456.jpg',  # Multiple objects in same image
            'IMG_20230615_123457.jpg',
            'IMG_20230615_123458.jpg',
            'IMG_20230615_123506.jpg',
            'invalid_filename.jpg'  # Invalid filename format
        ],
        'xmin': [100, 200, 150, 200, 500, 300],
        'ymin': [100, 200, 150, 200, 500, 300],
        'xmax': [120, 220, 170, 220, 520, 320],
        'ymax': [120, 220, 170, 220, 520, 320],
        'label': ['Bird', 'Bird', 'Bird', 'Bird', 'Bird', 'Bird']
    })

@pytest.fixture
def propagator():
    """Create a LabelPropagator instance."""
    return LabelPropagator(time_threshold_seconds=5, distance_threshold_pixels=50)

def test_propagator_initialization(propagator):
    """Test propagator initialization with different parameters."""
    assert propagator.time_threshold == 5
    assert propagator.distance_threshold == 50
    
    # Test with different parameters
    prop2 = LabelPropagator(time_threshold_seconds=10, distance_threshold_pixels=100)
    assert prop2.time_threshold == 10
    assert prop2.distance_threshold == 100

def test_timestamp_parsing(propagator):
    """Test timestamp parsing from various filename formats."""
    # Test valid filename
    filename = 'IMG_20230615_123456.jpg'
    timestamp = propagator._parse_timestamp(filename)
    assert timestamp == datetime(2023, 6, 15, 12, 34, 56)
    
    # Test invalid filename
    invalid_filename = 'invalid_filename.jpg'
    assert propagator._parse_timestamp(invalid_filename) is None

def test_center_calculation(propagator):
    """Test bounding box center calculation."""
    # Test integer coordinates
    bbox = (100, 100, 120, 120)
    center = propagator._calculate_center(bbox)
    assert center == (110, 110)
    
    # Test float coordinates
    bbox = (100.5, 100.5, 120.5, 120.5)
    center = propagator._calculate_center(bbox)
    assert center == (110.5, 110.5)

def test_distance_calculation(propagator):
    """Test distance calculation between points."""
    # Test vertical distance
    point1 = (100, 100)
    point2 = (100, 150)
    distance = propagator._calculate_distance(point1, point2)
    assert distance == 50
    
    # Test diagonal distance
    point2 = (100 + 30, 100 + 40)  # 3-4-5 triangle
    distance = propagator._calculate_distance(point1, point2)
    assert distance == 50

def test_temporal_neighbors(propagator, sample_annotations):
    """Test finding temporal neighbors."""
    neighbors = propagator._find_temporal_neighbors(sample_annotations)
    
    # First three images should be neighbors
    assert 'IMG_20230615_123457.jpg' in neighbors['IMG_20230615_123456.jpg']
    assert 'IMG_20230615_123458.jpg' in neighbors['IMG_20230615_123456.jpg']
    
    # Last image should not be neighbor of first (10 seconds apart)
    assert 'IMG_20230615_123506.jpg' not in neighbors['IMG_20230615_123456.jpg']

def test_label_propagation(propagator, sample_annotations):
    """Test label propagation with various scenarios."""
    propagated_df = propagator.propagate_labels(sample_annotations)
    
    # Check basic properties
    assert 'propagated' in propagated_df.columns
    assert len(propagated_df) >= len(sample_annotations)
    assert (propagated_df['propagated'] == True).any()
    
    # Check that original annotations are preserved
    original_images = sample_annotations['image_path'].unique()
    for img in original_images:
        assert img in propagated_df['image_path'].values

def test_complex_propagation(propagator, complex_annotations):
    """Test label propagation with complex scenarios."""
    propagated_df = propagator.propagate_labels(complex_annotations)
    
    # Check handling of multiple objects in same image
    first_image_annotations = propagated_df[
        propagated_df['image_path'] == 'IMG_20230615_123456.jpg'
    ]
    assert len(first_image_annotations) >= 2
    
    # Check handling of invalid filenames
    invalid_annotations = propagated_df[
        propagated_df['image_path'] == 'invalid_filename.jpg'
    ]
    assert len(invalid_annotations) == 1  # Should preserve original annotation

def test_empty_dataframe(propagator):
    """Test handling of empty input DataFrame."""
    empty_df = pd.DataFrame(columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    result = propagator.propagate_labels(empty_df)
    assert len(result) == 0
    assert 'propagated' in result.columns