import pandas as pd
import pytest
import torch
from deepforest.utilities import read_file

from src.pipeline_evaluation import PipelineEvaluation


def _scalar(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)

@pytest.fixture
def sample_predictions(tmp_path):
    # Create sample predictions
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "xmin": [10, 60],
        "ymin": [10, 60],
        "xmax": [50, 100],
        "ymax": [50, 100],
        "score": [0.9, 0.8],
        'label': ['label1', 'label1'],
        'cropmodel_label': ['genus species1', 'genus species2'],
        'cropmodel_score': [0.75, 0.45]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, tmp_path)
    
    return gdf

@pytest.fixture
def sample_annotations(tmp_path):
    # Create sample detection annotations
    data = {
        "image_path": ["birds.jpg", "birds.jpg", "birds.jpg"],
        "label": ["genus species2", "genus species2", "genus species1"],
        "xmin": [10, 60, 200],
        "ymin": [10, 60, 200],
        "xmax": [50, 100, 210],
        "ymax": [50, 100, 210]
    }
    df = pd.DataFrame(data)
    gdf = read_file(df, tmp_path)
    
    return gdf
@pytest.fixture
def classification_label_dict():
    return {"genus species1": 0, "genus species2": 1}

def test_check_success(sample_predictions, sample_annotations, classification_label_dict):
    """Recall is below detection_true_positive_threshold, so check_success is False."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        annotations=sample_annotations,
        classification_label_dict=classification_label_dict,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    pipeline_evaluation.evaluate()
    assert pipeline_evaluation.check_success() is False

def test_evaluate(sample_predictions, sample_annotations, classification_label_dict):
    """Test confident classification accuracy with sample data and perfect performance."""
    pipeline_evaluation = PipelineEvaluation(
        predictions=sample_predictions,
        annotations=sample_annotations,
        classification_label_dict=classification_label_dict,
        detection_true_positive_threshold=0.85,
        classification_threshold=0.5
    )
    results = pipeline_evaluation.evaluate()

    assert results["detection"]["recall"] == pytest.approx(2 / 3)
    assert results["detection"]["precision"] == pytest.approx(1.0)

    conf = results["classification"]["confident"]
    unc = results["classification"]["uncertain"]
    assert _scalar(conf["micro_accuracy"]) == pytest.approx(0.0)
    assert _scalar(conf["avg_false_classification_score"]) == pytest.approx(0.9)
    assert _scalar(unc["micro_accuracy"]) == pytest.approx(1.0)
    assert _scalar(unc["avg_true_classification_score"]) == pytest.approx(0.8)
