import pytest
import pandas as pd
import os
import numpy as np
from src.data_processing import undersample, preprocess_images, process_image

@pytest.fixture
def sample_annotations():
    """Create sample annotations for testing."""
    return pd.DataFrame({
        'image_path': ['img1.jpg', 'img1.jpg', 'img2.jpg', 'img3.jpg', 'img3.jpg'],
        'label': ['Bird', 'Bird', 'Bird', 'Bird', 'Rare'],
        'xmin': [100, 200, 300, 400, 500],
        'ymin': [100, 200, 300, 400, 500],
        'xmax': [150, 250, 350, 450, 550],
        'ymax': [150, 250, 350, 450, 550]
    })

@pytest.fixture
def empty_annotations():
    """Create annotations with some empty images."""
    return pd.DataFrame({
        'image_path': ['img1.jpg', 'img2.jpg'],
        'label': ['Bird', None],
        'xmin': [100, None],
        'ymin': [100, None],
        'xmax': [150, None],
        'ymax': [150, None]
    })

@pytest.fixture
def mock_split_raster(monkeypatch):
    """Mock the split_raster function."""
    def mock_fn(*args, **kwargs):
        return pd.DataFrame({
            'image_path': ['patch1.jpg'],
            'xmin': [10],
            'ymin': [10],
            'xmax': [20],
            'ymax': [20]
        })
    
    monkeypatch.setattr('src.data_processing.preprocess.split_raster', mock_fn)
    return mock_fn

def test_undersample_ratio():
    """Test undersampling with different ratios."""
    df = pd.DataFrame({
        'label': ['Bird'] * 8 + ['Rare'] * 2,
        'image_path': [f'img{i}.jpg' for i in range(10)]
    })
    
    # Test with ratio 0.5
    result = undersample(df, ratio=0.5)
    assert len(result) < len(df)
    
    # Test with ratio 0
    result = undersample(df, ratio=0)
    assert len(result) == 2  # Only rare class images
    
    # Test with ratio 1
    result = undersample(df, ratio=1)
    assert len(result) == len(df)

def test_undersample_invalid_ratio():
    """Test undersampling with invalid ratios."""
    df = pd.DataFrame({'label': ['Bird'], 'image_path': ['img1.jpg']})
    
    with pytest.raises(ValueError):
        undersample(df, ratio=-0.1)
    
    with pytest.raises(ValueError):
        undersample(df, ratio=1.1)

def test_preprocess_images(mock_split_raster, sample_annotations, tmp_path):
    """Test image preprocessing."""
    # Create temporary directories
    root_dir = tmp_path / "root"
    save_dir = tmp_path / "save"
    root_dir.mkdir()
    
    # Create dummy image files
    for img in sample_annotations['image_path'].unique():
        (root_dir / img).touch()
    
    # Test preprocessing
    result = preprocess_images(
        sample_annotations,
        str(root_dir),
        str(save_dir),
        patch_size=450
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'image_path' in result.columns

def test_preprocess_images_invalid_params(sample_annotations, tmp_path):
    """Test preprocessing with invalid parameters."""
    with pytest.raises(ValueError):
        preprocess_images(
            sample_annotations,
            str(tmp_path),
            str(tmp_path),
            patch_size=-1
        )
    
    with pytest.raises(ValueError):
        preprocess_images(
            sample_annotations,
            str(tmp_path),
            str(tmp_path),
            patch_size=450,
            patch_overlap=-1
        )

def test_process_image_empty(empty_annotations, tmp_path):
    """Test processing images with empty annotations."""
    root_dir = tmp_path / "root"
    save_dir = tmp_path / "save"
    root_dir.mkdir()
    save_dir.mkdir()
    
    # Create dummy image
    (root_dir / "img2.jpg").touch()
    
    result = process_image(
        "img2.jpg",
        None,
        str(root_dir),
        str(save_dir),
        patch_size=450,
        patch_overlap=0,
        allow_empty=True
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 'image_path' in result.columns
    assert result['xmin'].isna().all()

def test_process_image_existing_crops(mock_split_raster, tmp_path):
    """Test processing when crops already exist."""
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    
    # Create existing crop CSV
    crop_csv = save_dir / "img1.csv"
    pd.DataFrame({'image_path': ['patch1.jpg']}).to_csv(crop_csv)
    
    result = process_image(
        "img1.jpg",
        None,
        str(tmp_path),
        str(save_dir),
        patch_size=450,
        patch_overlap=0,
        allow_empty=True
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

def test_process_image_file_not_found(sample_annotations, tmp_path):
    """Test processing with missing image file."""
    with pytest.raises(FileNotFoundError):
        process_image(
            "nonexistent.jpg",
            sample_annotations,
            str(tmp_path),
            str(tmp_path),
            patch_size=450,
            patch_overlap=0,
            allow_empty=False
        )

@pytest.mark.parametrize("patch_size,patch_overlap", [
    (450, 0),
    (300, 50),
    (600, 100)
])
def test_process_image_different_sizes(mock_split_raster, sample_annotations, tmp_path, patch_size, patch_overlap):
    """Test processing with different patch sizes and overlaps."""
    root_dir = tmp_path / "root"
    save_dir = tmp_path / "save"
    root_dir.mkdir()
    save_dir.mkdir()
    
    # Create dummy image
    (root_dir / "img1.jpg").touch()
    
    result = process_image(
        "img1.jpg",
        sample_annotations,
        str(root_dir),
        str(save_dir),
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        allow_empty=False
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

@pytest.mark.integration
def test_full_preprocessing_pipeline(tmp_path):
    """Integration test for the full preprocessing pipeline."""
    # Create test data
    root_dir = tmp_path / "root"
    save_dir = tmp_path / "save"
    root_dir.mkdir()
    save_dir.mkdir()
    
    # Create test image and annotations
    (root_dir / "test.jpg").touch()
    annotations = pd.DataFrame({
        'image_path': ['test.jpg'],
        'label': ['Bird'],
        'xmin': [100],
        'ymin': [100],
        'xmax': [200],
        'ymax': [200]
    })
    
    try:
        result = preprocess_images(
            annotations,
            str(root_dir),
            str(save_dir),
            patch_size=450
        )
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")
