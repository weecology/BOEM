import pandas as pd
import os
from datetime import datetime
from src.visualization import PredictionVisualizer

class Reporting:
    def __init__(self, report_dir, image_dir, pipeline_monitor=None):
        """Initialize reporting class
        
        Args:
            report_dir: Directory to save reports
            image_dir: Directory containing images to create video from
            pipeline_monitor: PipelineEvaluation instance containing model performance metrics
        """

        self.report_dir = report_dir
        self.report_file = f"{report_dir}/report.csv"
        self.image_dir = image_dir
        self.pipeline_monitor = pipeline_monitor

    def concat_predictions(self):
        """Concatenate predictions
        
        Args:
            predictions: List of dataframes containing predictions
        """
        self.all_predictions = pd.concat(self.pipeline_monitor.predictions, ignore_index=True)

    def generate_report(self):
        """Generate a report"""

        self.concat_predictions()
        self.write_predictions()
        self.write_metrics()
        self.generate_video()

    def write_predictions(self):
        """Write predictions to a csv file"""
        self.concat_predictions()
        self.all_predictions.to_csv(f"{self.report_dir}/predictions.csv", index=False)

        return f"{self.report_dir}/predictions.csv"
    def get_coco_datasets(self):
        """Get coco datasets"""
        self.pipeline_monitor.mAP.get_coco_datasets()

    def generate_video(self):
        """Generate a video from the predictions"""
        self.concat_predictions()
        visualizer = PredictionVisualizer(self.all_predictions, self.report_dir)
        output_path = f"{self.report_dir}/predictions.mp4"
        images = self.all_predictions['image_path'].unique()
        images = [os.path.join(self.image_dir, image) for image in images]

        output_path = visualizer.create_visualization(images=images)

        return output_path
        
    def write_metrics(self):
        """Write metrics to a csv file
        
        Args:
            pipeline_monitor: PipelineEvaluation instance containing model performance metrics
        """
        self.concat_predictions()
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get performance metrics
        performance = self.pipeline_monitor.results
        
        # Extract key metrics
        detection_map = performance['detection']['mAP']['map']
        confident_acc = performance['confident_classification']["confident_classification_accuracy"]
        uncertain_acc = performance['uncertain_classification']["uncertain_classification_accuracy"]

        # Get annotation counts and completion rate
        human_reviewed_images = len(self.all_predictions['image_path'].unique())
        total_images = len(os.listdir(self.image_dir))
        completion_rate = human_reviewed_images / total_images
        total_annotations = self.all_predictions.shape[0]

        try:
            confident_annotations = self.pipeline_monitor.confident_predictions.shape[0]
        except:
            confident_annotations = 0
        try:
            uncertain_annotations = self.pipeline_monitor.uncertain_predictions.shape[0]
        except:
            uncertain_annotations = 0

        # Create report row
        report_data = {
            'timestamp': timestamp,
            'model_name': self.pipeline_monitor.model.__class__.__name__,
            'total_annotations': total_annotations,
            'confident_annotations': confident_annotations,
            'uncertain_annotations': uncertain_annotations,
            'detection_map': detection_map,
            'human_reviewed_images': human_reviewed_images,
            'total_images': total_images,
            'completion_rate': completion_rate,
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

        return f"{self.report_dir}/report.csv"
