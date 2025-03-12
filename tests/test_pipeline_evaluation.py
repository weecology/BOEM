from src.pipeline_evaluation import PipelineEvaluation
from deepforest.utilities import read_file
import pytest
import pandas as pd

@pytest.fixture
def sample_predictions(tmp_path):
    # Create sample predictions
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60],
        "score": [0.9, 0.8],
        'label': ['label1', 'label2'],
        'cropmodel_label': [0, 1],
        'cropmodel_score': [0.9, 0.8]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, tmp_path)
    
    return gdf

@pytest.fixture
def sample_detection_annotations(tmp_path):
    # Create sample detection annotations
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["label1", "label2"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, tmp_path)
    
    return gdf

@pytest.fixture
def sample_classification_annotations(tmp_path):
    # Create sample classification annotations
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["label1", "label2"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60]
        
    }
    df = pd.DataFrame(data)
    
    return df

def test_check_success(config, sample_predictions, sample_detection_annotations, sample_classification_annotations, comet_logger):
    """Test check success with sample data and perfect performance."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        detection_annotations=sample_detection_annotations,
        classification_annotations=sample_classification_annotations,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    pipeline_evaluation.evaluate()
    assert pipeline_evaluation.check_success() is True

def test_evaluate_detection(config, sample_predictions, sample_detection_annotations, sample_classification_annotations, comet_logger):
    """Test evaluate detection with sample data."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        detection_annotations=sample_detection_annotations,
        classification_annotations=sample_classification_annotations,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    detection_results = pipeline_evaluation.evaluate_detection()
    
    # Detection results are mocked, one image is correct, the other is not.
    assert detection_results["recall"] == 1.0

def test_confident_classification_accuracy(config, sample_predictions, sample_detection_annotations, sample_classification_annotations, comet_logger):
    """Test confident classification accuracy with sample data and perfect performance."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        detection_annotations=sample_detection_annotations,
        classification_annotations=sample_classification_annotations,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    confident_classification_accuracy = pipeline_evaluation.evaluate_confident_classification()
   
    assert confident_classification_accuracy["multiclassaccuracy"] == 1.0

def test_uncertain_classification_accuracy(config, sample_predictions, sample_detection_annotations, sample_classification_annotations):
    """Test uncertain classification accuracy with sample data and perfect performance."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        detection_annotations=sample_detection_annotations,
        classification_annotations=sample_classification_annotations,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    uncertain_classification_accuracy = pipeline_evaluation.evaluate_uncertain_classification()
    list(uncertain_classification_accuracy.keys()) == ['multiclassaccuracy', 'avg_score_true_positive', 'avg_score_false_positive']
    assert uncertain_classification_accuracy["multiclassaccuracy"] == 0