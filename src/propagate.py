import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import os

class LabelPropagator:
    """
    A class to propagate object detection labels across temporally and spatially close images.
    
    This class analyzes image timestamps and object locations to identify potential missed
    detections in sequential images. When an object is detected in one image but missing in
    a temporally close image with similar spatial coordinates, the label is propagated.
    
    Attributes:
        time_threshold (int): Maximum time difference in seconds between images to consider for propagation
        distance_threshold (float): Maximum Euclidean distance in pixels between objects to consider them the same
    
    Example:
        >>> propagator = LabelPropagator(time_threshold_seconds=5, distance_threshold_pixels=50)
        >>> propagated_df = propagator.propagate_labels(annotations_df)
    """

    def __init__(self, time_threshold_seconds: int = 5, distance_threshold_pixels: float = 50):
        """
        Initialize the label propagator.
        
        Args:
            time_threshold_seconds: Maximum time difference between images to consider for propagation
            distance_threshold_pixels: Maximum distance between objects to consider them the same
        """
        self.time_threshold = time_threshold_seconds
        self.distance_threshold = distance_threshold_pixels

    def _parse_timestamp(self, filename: str) -> Optional[datetime]:
        """
        Extract timestamp from image filename.
        
        Args:
            filename: Name of the image file with embedded timestamp
            
        Returns:
            datetime object if parsing successful, None otherwise
            
        Example:
            >>> propagator._parse_timestamp("IMG_20230615_123456.jpg")
            datetime(2023, 6, 15, 12, 34, 56)
        """
        try:
            date_str = filename.split('_')[1] + filename.split('_')[2].split('.')[0]
            return datetime.strptime(date_str, '%Y%m%d%H%M%S')
        except:
            return None

    def _calculate_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box.
        
        Args:
            bbox: Tuple of (xmin, ymin, xmax, ymax) coordinates
            
        Returns:
            Tuple of (x, y) coordinates of the center point
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: Tuple of (x, y) coordinates of first point
            point2: Tuple of (x, y) coordinates of second point
            
        Returns:
            Euclidean distance between the points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _find_temporal_neighbors(self, annotations_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Find temporally close images within the time threshold.
        
        Args:
            annotations_df: DataFrame containing image annotations
            
        Returns:
            Dictionary mapping each image to list of temporal neighbors
            
        Note:
            Images are considered neighbors if their timestamps are within
            the time_threshold of each other.
        """
        temporal_neighbors = {}
        
        timestamps = {row['image_path']: self._parse_timestamp(os.path.basename(row['image_path'])) 
                     for _, row in annotations_df.iterrows()}
        
        for img1 in timestamps:
            temporal_neighbors[img1] = []
            t1 = timestamps[img1]
            if t1 is None:
                continue
                
            for img2 in timestamps:
                if img1 == img2:
                    continue
                t2 = timestamps[img2]
                if t2 is None:
                    continue
                    
                time_diff = abs((t2 - t1).total_seconds())
                if time_diff <= self.time_threshold:
                    temporal_neighbors[img1].append(img2)
                    
        return temporal_neighbors

    def propagate_labels(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Propagate labels to temporally and spatially close objects.
        
        This method analyzes the input annotations and propagates labels to nearby
        images when an object is detected in one image but missing in temporally
        close images at similar spatial coordinates.
        
        Args:
            annotations_df: DataFrame with columns ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
            
        Returns:
            DataFrame with original and propagated labels, including a 'propagated' column
            
        Note:
            Propagated labels are marked with propagated=True in the output DataFrame
        """
        propagated_df = annotations_df.copy()
        temporal_neighbors = self._find_temporal_neighbors(annotations_df)
        new_annotations = []
        
        for img1 in temporal_neighbors:
            img1_annotations = annotations_df[annotations_df['image_path'] == img1]
            
            for img2 in temporal_neighbors[img1]:
                img2_annotations = annotations_df[annotations_df['image_path'] == img2]
                
                for _, obj1 in img1_annotations.iterrows():
                    bbox1 = (obj1['xmin'], obj1['ymin'], obj1['xmax'], obj1['ymax'])
                    center1 = self._calculate_center(bbox1)
                    
                    match_found = False
                    for _, obj2 in img2_annotations.iterrows():
                        bbox2 = (obj2['xmin'], obj2['ymin'], obj2['xmax'], obj2['ymax'])
                        center2 = self._calculate_center(bbox2)
                        
                        distance = self._calculate_distance(center1, center2)
                        if distance <= self.distance_threshold:
                            match_found = True
                            break
                    
                    if not match_found:
                        new_annotation = obj1.copy()
                        new_annotation['image_path'] = img2
                        new_annotation['propagated'] = True
                        new_annotations.append(new_annotation)
        
        if new_annotations:
            propagated_df = pd.concat([propagated_df, pd.DataFrame(new_annotations)], ignore_index=True)
            
        if 'propagated' not in propagated_df.columns:
            propagated_df['propagated'] = False
            
        return propagated_df 