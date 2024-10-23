from src.monitoring import Monitoring

class PipelineEvaluation:
    def __init__(self, prediction_directory):
        self.monitoring = Monitoring()
        self.prediction_directory = prediction_directory
        self.check_predictions()
        
    def check_predictions(self, prediction_directory):
        import os

        if not os.path.exists(prediction_directory):
            raise FileNotFoundError(f"Prediction directory '{prediction_directory}' does not exist.")
        
        prediction_files = [f for f in os.listdir(prediction_directory) if f.endswith('.pt') or f.endswith('.pth')]
        
        if not prediction_files:
            raise FileNotFoundError(f"No prediction files found in '{prediction_directory}'. Please run initiate.py to generate predictions.")
        
        return True

    def evaluate_pipeline(self, pipeline):
        # Implementation for pipeline evaluation
        pass

    # ... other methods ...
