import pandas as pd
import os
from datetime import datetime
from src.visualization import PredictionVisualizer
from src.detection import predict
import geopandas as gpd
import glob

class Reporting:
    def __init__(self, report_dir, image_dir, metadata_csv, pipeline_monitor=None, model=None, classification_model=None, confident_predictions=None, uncertain_predictions=None, thin_factor=10,patch_overlap=0.2, patch_size=300, min_score=0.3):
        """Initialize reporting class
        
        Args:
            report_dir: Directory to save reports
            image_dir: Directory containing images to create video from
            pipeline_monitor: PipelineEvaluation instance containing model performance metrics
            model: Detection model
            classification_model: Classification model
            patch_overlap: Patch overlap for detection model
            patch_size: Patch size for detection model
            min_score: Minimum score for detection model
            thin_factor: Factor to thin images by for video creation
            metadata_csv: Path to metadata csv for image location
            confident_predictions: Dataframe containing confident predictions
            uncertain_predictions: Dataframe containing uncertain predictions
        """

        # Create timestamped report directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = os.path.join(report_dir, timestamp)
        self.report_file = f"{self.report_dir}/report.csv"
        self.image_dir = image_dir
        self.sample_prediction_dir = f"{self.report_dir}/samples"
        self.model = model
        self.classification_model = classification_model
        self.patch_overlap = patch_overlap
        self.patch_size = patch_size
        self.min_score = min_score
        self.thin_factor = thin_factor
        self.uncertain_predictions = uncertain_predictions
        self.confident_predictions = confident_predictions
        self.metadata = metadata_csv

        self.detection_experiment = model.trainer.logger.experiment
        self.classification_experiment = classification_model.trainer.logger.experiment
        
        # Check the dirs exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.sample_prediction_dir, exist_ok=True)

        self.pipeline_monitor = pipeline_monitor

    def concat_predictions(self):
        """Concatenate predictions
        
        Args:
            predictions: List of dataframes containing predictions
        """
        self.all_predictions = pd.concat(self.pipeline_monitor.predictions, ignore_index=True)

    def generate_report(self, create_video=False):
        """Generate a report"""

        if self.pipeline_monitor:
            self.concat_predictions()
            self.write_predictions()
        self.write_metrics()
        if create_video:
            self.generate_video()

    def write_predictions(self):
        """Write predictions to a csv file"""
        self.concat_predictions()
        self.all_predictions.to_csv(f"{self.report_dir}/predictions.csv", index=False)
        self.all_predictions['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.all_predictions["unique_image"] = self.all_predictions["image_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
        # Connect with metadata on location
        metadata_df = pd.read_csv(self.metadata)
        merged_predictions = self.all_predictions.merge(metadata_df[["unique_image", "flight_name","date","lat","long"]], on='unique_image')
        merged_predictions.to_csv(f"{self.report_dir}/predictions.csv", index=False)

        # Create shapefile
        gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat)).to_file(f"{self.report_dir}/predictions.shp")

        return f"{self.report_dir}/predictions.csv"

    def select_images_for_video(self):
        all_images = glob.glob(self.image_dir + "/*.jpg")
        # Thin by factor, select every nth image
        thinned_images = all_images[::self.thin_factor]

        return thinned_images
    
    def predict_video_images(self, images):
        """Predict on images selected for video"""

        predictions = self.video_predictions = predict(
            image_paths=images,
            m=self.model,
            crop_model=self.classification_model,
            patch_overlap=self.patch_overlap,
            patch_size=self.patch_size,
            )
        
        predictions = pd.concat(predictions, ignore_index=True)
        
        predictions = predictions[predictions.score > self.min_score]
        
        return predictions

    def get_coco_datasets(self):
        """Get coco datasets"""
        self.pipeline_monitor.mAP.get_coco_datasets()

    def generate_video(self):
        """Generate a video from the predictions"""
        images = self.select_images_for_video()
        video_predictions = self.predict_video_images(images)
        visualizer = PredictionVisualizer(video_predictions, self.report_dir)
        output_path = f"{self.report_dir}/predictions.mp4"
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
        confident_acc = performance['confident_classification']["multiclassaccuracy"]
        uncertain_acc = performance['uncertain_classification']["multiclassaccuracy"]

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
            'confident_predictions': confident_annotations,
            'uncertain_predictions': uncertain_annotations,
            'detection_map': detection_map,
            'human_reviewed_images': human_reviewed_images,
            'total_images': total_images,
            'completion_rate': completion_rate,
            'confident_classification_accuracy': confident_acc,
            'uncertain_classification_accuracy': uncertain_acc,
        }

        # If comet logger exists add model evaluation urls
        try:
            report_data['detection_model_url'] = self.detection_experiment.url
            for metric in self.detection_experiment.metrics:
                report_data[f"detection_{metric}"] = self.detection_experiment.metrics[metric]
        except AttributeError:
            report_data['detection_model_url'] = None

        try:
            report_data['classification_model_url'] = self.classification_experiment.url
            for metric in self.classification_experiment.metrics:
                report_data[f"classification_{metric}"] = self.classification_experiment.metrics[metric]
        except AttributeError:
            report_data['classification_model_url'] = None

        # Load existing or create new report file
        if os.path.exists(self.report_file):
            df = pd.read_csv(self.report_file)
            df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
        else:
            df = pd.DataFrame([report_data])
            
        # Save updated reports
        df.to_csv(self.report_file, index=False)

        return f"{self.report_dir}/report.csv"
