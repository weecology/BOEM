import os
import glob
import pandas as pd
from src.detection import predict
import cv2
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import subprocess
import zipfile
import tempfile

class PredictionVisualizer:
    def __init__(self, predictions: pd.DataFrame, output_dir: str, fps: int = 5, frame_size: Tuple[int, int] = (1920, 1080), thin_factor: int = 10, codec=None):
        """
        Initialize the prediction visualizer.
        
        Args:
            predictions: Prediction dataframe
            output_dir: Directory to save visualization outputs
            fps: Frames per second for output video
            frame_size: Output video frame size (width, height)
            thin_factor: Take every nth image from sorted list
            codec: Video codec to use for output video
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_size = frame_size
        self.thin_factor = thin_factor
        self.predictions = predictions
        self.codec = codec or cv2.VideoWriter_fourcc(*'avc1')  # Default to H.264
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique labels from predictions
        unique_labels = predictions['label'].unique()
        
        # Create color ramp based on number of unique labels
        color_ramp = {}
        for i, label in enumerate(unique_labels):
            # Create evenly spaced hue values between 0-255
            hue = int(255 * (i / len(unique_labels)))
            # Convert HSV to BGR (OpenCV uses BGR)
            rgb = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_ramp[label] = tuple(map(int, rgb))
            
        self.colors = color_ramp

    def draw_predictions(self, image: np.ndarray, predictions: pd.DataFrame, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image as numpy array
            predictions: DataFrame with predictions
            confidence_threshold: Minimum confidence to show prediction
            
        Returns:
            Image with drawn predictions
        """
        img_with_boxes = image.copy()
        
        # Filter predictions by confidence
        confident_preds = predictions[predictions['score'] >= confidence_threshold]
        
        for _, pred in confident_preds.iterrows():
            # Get coordinates
            xmin, ymin, xmax, ymax = pred[['xmin', 'ymin', 'xmax', 'ymax']]
            
            # Get color for class
            color = self.colors.get(pred['label'], (255, 255, 255))
            
            # Draw box
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label with confidence
            label = f"{pred['label']}: {pred['score']:.2f}"
            cv2.putText(img_with_boxes, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img_with_boxes

    def create_visualization(self, images: list, output_name: str = "predictions.mp4", confidence_threshold: float = 0.5) -> str:
        """
        Create video visualization of predictions on image sequence.
        
        Args:
            images: List of image paths
            output_name: Name of output video file
            confidence_threshold: Minimum confidence to show prediction
            
        Returns:
            Path to output video file
        """
        # Sort images by filename
        images.sort()
        
        # Thin the image list
        images = images[::self.thin_factor]
        
        # Create video writer
        output_path = str(self.output_dir / output_name)
        first_image = cv2.imread(images[0])
        if first_image is None:
            raise ValueError(f"Could not read first image: {images[0]}")
            
        height, width = first_image.shape[:2]
        
        video_writer = cv2.VideoWriter(output_path, self.codec, self.fps, (width, height))
        
        try:
            # Process each image
            for img_path in tqdm(images, desc="Creating visualization"):
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Get predictions
                predictions = self.predictions[self.predictions.image_path == img_path]
                
                # Draw predictions
                annotated_image = self.draw_predictions(image, predictions, confidence_threshold)
                
                # Write frame
                video_writer.write(annotated_image)
                
        finally:
            video_writer.release()
            
        return output_path
    
def write_crops(root_dir, images, boxes, labels, experiment, expand=30):
    """Write crops to disk or zip and upload as asset."""

    basenames = []

    # Use a single temp dir for all crops and the zip
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "crops.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for index, box in enumerate(boxes):
                try:
                    xmin, ymin, xmax, ymax = box
                    xmin = max(0, xmin - expand)
                    ymin = max(0, ymin - expand)
                    xmax = max(0, xmax + expand)
                    ymax = max(0, ymax + expand)

                    label = labels[index]
                    image = images[index]
                    basename = os.path.splitext(os.path.basename(image))[0]
                    img_path = os.path.join(tmpdir, f"{basename}_{label}_{index}.png")

                    with rasterio.open(os.path.join(root_dir, image)) as src:
                        img = src.read(window=((ymin, ymax), (xmin, xmax)))
                        img = np.rollaxis(img, 0, 3)
                        cv2.imwrite(img_path, img)
                        zipf.write(img_path, os.path.basename(img_path))
                        basenames.append(os.path.basename(img_path))

                except Exception as e:
                    print(f"Error processing image {image}: {e}")
                    basenames.append(None)
        experiment.log_asset(file_data=zip_path, file_name="crops.zip")
        
        return basenames
    
def crop_images(annotations, root_dir, experiment=None, expand=30):
    """Crop images based on bounding boxes and save them to disk."""
    # Get the bounding boxes, images, and labels
    boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    images = annotations["image_path"].values
    labels = annotations["cropmodel_label"].values

    # Write crops to disk or log to experiment, location is the path
    crop_image_paths = write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, experiment=experiment, expand=expand)
    
    return crop_image_paths

def select_images_for_video(image_dir, thin_factor):
    all_images = glob.glob(image_dir + "/*.jpg")
    # Thin by factor, select every nth image
    thinned_images = all_images[::thin_factor]
    return thinned_images

def predict_video_images(images, model, classification_model, patch_overlap, patch_size, batch_size, min_score):
    predictions = predict(
        image_paths=images,
        m=model,
        crop_model=classification_model,
        patch_overlap=patch_overlap,
        patch_size=patch_size,
        batch_size=batch_size,
    )
    
    predictions = pd.concat(predictions, ignore_index=True)
    predictions = predictions[predictions.score > min_score]
    return predictions

def generate_video(image_dir, report_dir, model, classification_model, patch_overlap, patch_size, batch_size, min_score, thin_factor):
    images = select_images_for_video(image_dir, thin_factor)
    video_predictions = predict_video_images(images, model, classification_model, patch_overlap, patch_size, batch_size, min_score)
    visualizer = PredictionVisualizer(video_predictions, report_dir)
    
    # Give the flightname as the video name
    flightname = image_dir.split("/")[-1]
    output_path = f"{report_dir}/{flightname}.mp4"
    output_path = visualizer.create_visualization(images=images)
    
    return output_path

def convert_codec(input_path: str, output_path: str) -> None:
    """
    Convert video codec using ffmpeg to a format that Streamlit can play.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to the output video file
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
    except:
        print("Error converting video codec. Make sure ffmpeg is installed and in your PATH.")


