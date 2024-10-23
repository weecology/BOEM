import pytest
from src.pipeline_evaluation import PipelineEvaluation
from src.monitoring import Monitoring

@pytest.fixture
def pipeline_evaluation():
    return PipelineEvaluation()

def test_evaluate_pipeline(pipeline_evaluation):
    # ... test code ...
