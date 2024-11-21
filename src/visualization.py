import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import os
from deepforest.model import CropModel
from tqdm import tqdm

class PredictionVisualizer:
    def __init__(
        self,
        model: CropModel,
        output_dir: str,
        fps: int = 30,
        frame_size: Tuple[int, int] = (1920, 1080),
        thin_factor: int = 10
    ):
        """
        Initialize the prediction visualizer.
        
        Args:
            model: Trained CropModel instance
            output_dir: Directory to save visualization outputs
            fps: Frames per second for output video
            frame_size: Output video frame size (width, height)
            thin_factor: Take every nth image from sorted list
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_size = frame_size
        self.thin_factor = thin_factor
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define colors for different classes
        self.colors = {
            'Bird': (0, 255, 0),    # Green
            'Empty': (128, 128, 128) # Gray
        }

    def draw_predictions(
        self,
        image: np.ndarray,
        predictions: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
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
            xmin, ymin = int(pred['xmin']), int(pred['ymin'])
            xmax, ymax = int(pred['xmax']), int(pred['ymax'])
            
            # Get color for class
            color = self.colors.get(pred['label'], (255, 255, 255))
            
            # Draw box
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label with confidence
            label = f"{pred['label']}: {pred['score']:.2f}"
            cv2.putText(
                img_with_boxes,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        return img_with_boxes

    def create_visualization(
        self,
        image_dir: str,
        output_name: str = "predictions.mp4",
        confidence_threshold: float = 0.5
    ) -> str:
        """
        Create video visualization of predictions on image sequence.
        
        Args:
            image_dir: Directory containing images
            output_name: Name of output video file
            confidence_threshold: Minimum confidence to show prediction
            
        Returns:
            Path to output video file
        """
        # Get sorted list of images
        image_paths = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Thin the image list
        image_paths = image_paths[::self.thin_factor]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        # Create video writer
        output_path = str(self.output_dir / output_name)
        first_image = cv2.imread(os.path.join(image_dir, image_paths[0]))
        if first_image is None:
            raise ValueError(f"Could not read first image: {image_paths[0]}")
            
        height, width = first_image.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        try:
            # Process each image
            for img_name in tqdm(image_paths[:1000], desc="Creating visualization"):
                img_path = os.path.join(image_dir, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Get predictions
                predictions = self.model.predict_image(img_path)
                
                # Draw predictions
                annotated_image = self.draw_predictions(
                    image,
                    predictions,
                    confidence_threshold
                )
                
                # Write frame
                video_writer.write(annotated_image)
                
        finally:
            video_writer.release()
            
        return output_path

    def create_summary_image(
        self,
        predictions_list: List[pd.DataFrame],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        """
        Create a summary image showing prediction statistics.
        
        Args:
            predictions_list: List of prediction DataFrames
            image_size: Size of output image
            
        Returns:
            Summary image as numpy array
        """
        # Create blank image
        summary = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
        
        # Compile statistics
        total_predictions = sum(len(preds) for preds in predictions_list)
        class_counts = {}
        confidence_scores = []
        
        for preds in predictions_list:
            for _, pred in preds.iterrows():
                class_counts[pred['label']] = class_counts.get(pred['label'], 0) + 1
                confidence_scores.append(pred['score'])
        
        # Draw statistics
        y_pos = 30
        cv2.putText(
            summary,
            f"Total Predictions: {total_predictions}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        y_pos += 40
        for label, count in class_counts.items():
            cv2.putText(
                summary,
                f"{label}: {count}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.colors.get(label, (0, 0, 0)),
                2
            )
            y_pos += 40
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            cv2.putText(
                summary,
                f"Average Confidence: {avg_confidence:.2f}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )
        
        return summary 