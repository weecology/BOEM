import pandas as pd
import os
from datetime import datetime
from src.visualization import PredictionVisualizer, convert_codec
from src.detection import predict
import geopandas as gpd
from pytorch_lightning.loggers import CometLogger
import glob
import zipfile
import os
import rasterio
import numpy as np
import cv2

class Reporting:
    def __init__(self, report_dir, image_dir, metadata_csv, pipeline_monitor=None, model=None, classification_model=None, confident_predictions=None, uncertain_predictions=None, thin_factor=10,patch_overlap=0.2, patch_size=300, min_score=0.3, batch_size=16):
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
            batch_size: Batch size for detection model

        """
        # Create timestamped report directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_dir = os.path.join(report_dir, timestamp)
        self.report_file = f"{self.report_dir}/report.csv"
        self.image_dir = image_dir
        self.model = model
        self.classification_model = classification_model
        self.patch_overlap = patch_overlap
        self.patch_size = patch_size
        self.min_score = min_score
        self.thin_factor = thin_factor
        self.uncertain_predictions = uncertain_predictions
        self.confident_predictions = confident_predictions
        self.metadata = metadata_csv
        self.batch_size = batch_size

        try:
            self.detection_experiment = model.trainer.logger
        except:
            self.detection_experiment = None
        try:
            self.classification_experiment = classification_model.trainer.logger
        except:
            self.classification_experiment = None
        
        # Check the dirs exist
        os.makedirs(self.report_dir, exist_ok=True)

        self.pipeline_monitor = pipeline_monitor

    def generate_report(self, create_video=False):
        """Generate a report and zip the contents
        
        Args:
            create_video (bool): Whether to create visualization video
        
        Returns:
            str: Path to the zipped report file
        """

        # Generate report contents
        if self.pipeline_monitor:
            validation_predictions = pd.concat(self.pipeline_monitor.predictions, ignore_index=True)
            self.write_predictions(validation_predictions, "validation_predictions")
        if self.confident_predictions is not None:
            self.write_predictions(self.confident_predictions, "confident_predictions")
        if self.uncertain_predictions is not None:
            self.write_predictions(self.uncertain_predictions, "uncertain_predictions")
        

        self.write_metrics()
        if create_video:
            video_path = self.generate_video()
            convert_codec(video_path, video_path.replace(".mp4", "_converted.mp4"))

        # Create zip file path
        zip_path = f"{self.report_dir}.zip"
        
        # Create zip file with just basenames
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.report_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname=arcname)
        
        return zip_path

    def write_predictions(self, annotations, basename):
        """Write predictions to a csv file"""
        annotations['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        annotations["unique_image"] = annotations["image_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
        # Connect with metadata on location
        metadata_df = pd.read_csv(self.metadata)
        merged_predictions = annotations.merge(metadata_df[["unique_image", "flight_name","date","lat","long"]], on='unique_image')
        merged_predictions.to_csv(f"{self.report_dir}/{basename}.csv", index=False)

        # Create shapefile
        gpd.GeoDataFrame(merged_predictions, geometry=gpd.points_from_xy(merged_predictions.long, merged_predictions.lat)).to_file(f"{self.report_dir}/{basename}.shp")

    
    def write_crops(self, root_dir, images, boxes, labels, savedir):
        """Write crops to disk.

        Args:
            root_dir (str): The root directory where the images are located.
            images (list): A list of image filenames.
            boxes (list): A list of bounding box coordinates in the format [xmin, ymin, xmax, ymax].
            labels (list): A list of labels corresponding to each bounding box.
            savedir (str): The directory where the cropped images will be saved.

        Returns:
            None
        """

        # Create a directory
        os.makedirs(os.path.join(savedir), exist_ok=True)

        # Use rasterio to read the image
        for index, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            label = labels[index]
            image = images[index]
            basename = os.path.splitext(os.path.basename(image))[0]
            with rasterio.open(os.path.join(root_dir, image)) as src:
                # Crop the image using the bounding box coordinates
                img = src.read(window=((ymin, ymax), (xmin, xmax)))
                
                # Save the cropped image as a PNG file using opencv
                img_path = os.path.join(savedir, f"{basename}_{label}_{index}.png")
                img = np.rollaxis(img, 0, 3)
                cv2.imwrite(img_path, img)
            
    def crop_images(self, annotations, root_dir, save_dir):
        # Remove any annotations with empty boxes
        annotations = annotations[(annotations['xmin'] != 0) & (annotations['ymin'] != 0) & (annotations['xmax'] != 0) & (annotations['ymax'] != 0)]
        
        # Remove any negative values
        annotations = annotations[(annotations['xmin'] >= 0) & (annotations['ymin'] >= 0) & (annotations['xmax'] >= 0) & (annotations['ymax'] >= 0)]
        boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        images = annotations["image_path"].values
        labels = annotations["label"].values
        self.write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir+"/crops")

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
            batch_size=self.batch_size,
            )
        
        predictions = pd.concat(predictions, ignore_index=True)
        
        predictions = predictions[predictions.score > self.min_score]
        
        # Save predictions
        self.write_predictions(predictions, "video_predictions")

        # Crop the images to the predictions
        self.crop_images(annotations=predictions, root_dir=self.image_dir, save_dir=self.report_dir)

        return predictions

    def get_coco_datasets(self):
        """Get coco datasets"""
        self.pipeline_monitor.mAP.get_coco_datasets()

    def generate_video(self):
        """Generate a video from the predictions"""
        images = self.select_images_for_video()
        video_predictions = self.predict_video_images(images)
        visualizer = PredictionVisualizer(video_predictions, self.report_dir)

        # Give the flightname as the video name
        flightname = self.image_dir.split("/")[-1]
        output_path = f"{self.report_dir}/{flightname}.mp4"
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
        if type(self.detection_experiment) == CometLogger:
            report_data['detection_model_url'] = self.detection_experiment.url
            for metric in self.detection_experiment.metrics:
                report_data[f"detection_{metric}"] = self.detection_experiment.metrics[metric]

            report_data['classification_model_url'] = self.classification_experiment.url
            for metric in self.classification_experiment.metrics:
                report_data[f"classification_{metric}"] = self.classification_experiment.metrics[metric]

        # Load existing or create new report file
        if os.path.exists(self.report_file):
            df = pd.read_csv(self.report_file)
            df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
        else:
            df = pd.DataFrame([report_data])
            
        # Save updated reports
        df.to_csv(self.report_file, index=False)

        return f"{self.report_dir}/report.csv"
