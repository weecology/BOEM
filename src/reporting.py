import pandas as pd
import os
from datetime import datetime
from src.visualization import PredictionVisualizer
class Reporting:
    def __init__(self, report_dir, pipeline_monitor):
        """Initialize reporting class"""
        self.report_dir = report_dir
        self.pipeline_monitor = pipeline_monitor

    def write_predictions(self, predictions):
        """Write predictions to a csv file"""
        all_predictions = pd.concat(self.pipeline_monitor.predictions)
        all_predictions.to_csv(f"{self.report_dir}/predictions.csv", index=False)
    
    def get_coco_datasets(self):
        """Get coco datasets"""
        self.pipeline_monitor.mAP.get_coco_datasets()

    def generate_video(self):
        """Generate a video from the predictions"""
        visualizer = PredictionVisualizer()
        visualizer.create_video(
            predictions_list=self.pipeline_monitor.predictions,
            output_path=f"{self.report_dir}/predictions.mp4"
        )
        
    def write_metrics(self):
        """Write metrics to a csv file
        
        Args:
            pipeline_monitor: PipelineEvaluation instance containing model performance metrics
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get performance metrics
        performance = self.pipeline_monitor.results
        
        # Extract key metrics
        detection_map = performance['detection']['mAP']['map']
        confident_acc = performance['confident_classification']['accuracy']
        uncertain_acc = performance['uncertain_classification']['accuracy']

        # Get annotation counts
        total_annotations = len(performance['detection']['annotations'])
        confident_annotations = len(performance['confident_classification']['annotations'])
        uncertain_annotations = len(performance['uncertain_classification']['annotations'])
        
        # Calculate completion rate
        completion_rate = (confident_annotations + uncertain_annotations) / total_annotations if total_annotations > 0 else 0
        
        # Create report row
        report_data = {
            'timestamp': timestamp,
            'model_name': self.pipeline_monitor.model.__class__.__name__,
            'completion_rate': completion_rate,
            'total_annotations': total_annotations,
            'confident_annotations': confident_annotations,
            'uncertain_annotations': uncertain_annotations,
            'detection_map': detection_map,
            'confident_classification_accuracy': confident_acc,
            'uncertain_classification_accuracy': uncertain_acc
        }
        
        # Load existing or create new report file
        if os.path.exists(self.report_file):
            df = pd.read_csv(self.report_file)
            df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
        else:
            df = pd.DataFrame([report_data])
            
        # Save updated reports
        df.to_csv(self.report_file, index=False)
