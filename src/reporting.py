import pandas as pd
import os
from datetime import datetime

class Reporting:
    def __init__(self):
        """Initialize reporting class"""
        self.report_file = "pipeline_reports.csv"

    def generate_reports(self, pipeline_monitor):
        """Generate reports from pipeline monitoring
        
        Args:
            pipeline_monitor: PipelineEvaluation instance containing model performance metrics
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get performance metrics
        performance = pipeline_monitor.report()
        
        # Extract key metrics
        detection_map = performance['detection']['map'] if 'detection' in performance else None
        confident_acc = performance['confident_classification'] if 'confident_classification' in performance else None 
        uncertain_acc = performance['uncertain_classification'] if 'uncertain_classification' in performance else None

        # Get annotation counts
        total_annotations = len(pipeline_monitor.detection_annotations_df)
        confident_annotations = len(pipeline_monitor.confident_classification_annotations_df)
        uncertain_annotations = len(pipeline_monitor.uncertain_classification_annotations_df)
        
        # Calculate completion rate
        completion_rate = (confident_annotations + uncertain_annotations) / total_annotations if total_annotations > 0 else 0
        
        # Create report row
        report_data = {
            'timestamp': timestamp,
            'model_name': pipeline_monitor.model.__class__.__name__,
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
