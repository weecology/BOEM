from src.pipeline_evaluation import PipelineEvaluation
from deepforest import main

def test_pipeline_evaluation(config):
    m = main.deepforest()
    pipeline_evaluation = PipelineEvaluation(model=m, **config.pipeline_evaluation)
    performance = pipeline_evaluation.evaluate()

def test_check_success(config):
    m = main.deepforest()
    pipeline_evaluation = PipelineEvaluation(model=m, **config.pipeline_evaluation)
    assert pipeline_evaluation.check_success() is False

def test_evaluate_detection(config):
    m = main.deepforest()
    pipeline_evaluation = PipelineEvaluation(model=m, **config.pipeline_evaluation)
    detection_results = pipeline_evaluation.evaluate_detection()
    assert detection_results["mAP"] is not None

def test_confident_classification_accuracy(config):
    m = main.deepforest()
    pipeline_evaluation = PipelineEvaluation(model=m, **config.pipeline_evaluation)
    confident_classification_accuracy = pipeline_evaluation.confident_classification_accuracy()
    assert confident_classification_accuracy is not None

    

