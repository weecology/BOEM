import os
from src.monitoring import Monitoring
from src.pipeline_evaluation import PipelineEvaluation

class ModelDeployment:
    def __init__(self):
        self.monitoring = Monitoring()
        self.pipeline_evaluation = PipelineEvaluation()

    # ... existing code ...
