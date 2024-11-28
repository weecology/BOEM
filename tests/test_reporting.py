from src import reporting
from src.pipeline_evaluation import PipelineEvaluation
from deepforest import main
from deepforest.model import CropModel
import pytest
import os
import pandas as pd

@pytest.fixture
def random_model():
    m = main.deepforest(label_dict={"Bird1": 1,"Bird2": 2}, num_classes=2)
    return m

@pytest.fixture
def random_crop_model():
    m = CropModel()
    return m

@pytest.fixture
def pipeline_monitor(config, random_model, random_crop_model):
    """Create a pipeline monitor instance"""

    pipeline_evaluation = PipelineEvaluation(model=random_model, crop_model=random_crop_model, **config.pipeline_evaluation)
    pipeline_evaluation.results = {"detection": {"mAP":{"map":0.5}}, "confident_classification": {"confident_classification_accuracy":0.5}, "uncertain_classification": {"uncertain_classification_accuracy":0.5}}
    predictions = pd.read_csv(config.detection_model.validation_csv_path)
    pipeline_evaluation.predictions = [predictions]

    return pipeline_evaluation

@pytest.fixture
def reporter(config, pipeline_monitor, tmpdir_factory):
    report_dir = tmpdir_factory.mktemp("report_dir").strpath  
    return reporting.Reporting(report_dir, config.reporting.image_dir, pipeline_monitor)

def test_generate_video(reporter):
    output_path = reporter.generate_video()
    assert os.path.exists(output_path)

def test_write_predictions(reporter):
    output_path = reporter.write_predictions()
    assert os.path.exists(output_path)

def test_write_metrics(reporter):
    output_path = reporter.write_metrics()
    assert os.path.exists(output_path)

def test_generate_report(reporter):
    reporter.generate_report()
    assert os.path.exists(f"{reporter.report_dir}/predictions.csv")
    assert os.path.exists(f"{reporter.report_dir}/report.csv")
    assert os.path.exists(f"{reporter.report_dir}/predictions.mp4")
