import pytest
import pandas as pd
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
def propagator():
    """Create a LabelPropagator instance."""
    return LabelPropagator(time_threshold_seconds=5, distance_threshold_pixels=50)

def test_propagator_initialization(propagator):
    """Test propagator initialization."""
    assert propagator.time_threshold == 5
    assert propagator.distance_threshold == 50

def test_timestamp_parsing(propagator):
    """Test timestamp parsing from filenames."""
    filename = 'IMG_20230615_123456.jpg'
    timestamp = propagator._parse_timestamp(filename)
    assert timestamp == datetime(2023, 6, 15, 12, 34, 56)

def test_center_calculation(propagator):
    """Test bounding box center calculation."""
    bbox = (100, 100, 120, 120)
    center = propagator._calculate_center(bbox)
    assert center == (110, 110)

def test_distance_calculation(propagator):
    """Test distance calculation between points."""
    point1 = (100, 100)
    point2 = (100, 150)
    distance = propagator._calculate_distance(point1, point2)
    assert distance == 50

def test_label_propagation(propagator, sample_annotations):
    """Test label propagation."""
    propagated_df = propagator.propagate_labels(sample_annotations)
    
    # Check that propagated column exists
    assert 'propagated' in propagated_df.columns
    
    # Check that original annotations are preserved
    assert len(propagated_df) >= len(sample_annotations)
    
    # Check that propagated annotations are marked
    assert (propagated_df['propagated'] == True).any()

def test_temporal_threshold(propagator, sample_annotations):
    """Test that labels only propagate within time threshold."""
    propagated_df = propagator.propagate_labels(sample_annotations)
    
    # The last image is 10 seconds later, should not receive propagated labels
    last_image_annotations = propagated_df[
        propagated_df['image_path'] == 'IMG_20230615_123506.jpg'
    ]
    assert len(last_image_annotations) == 1  # Only original annotation 