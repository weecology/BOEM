import pandas as pd
import os
from logging import warn
from deepforest import preprocess
from deepforest.utilities import read_file
from typing import Optional, Union, List, Dict
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import cv2

def undersample(train_df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    Undersample top classes by selecting most diverse images.
    
    This function reduces class imbalance by removing images that only contain
    the two most common classes, while preserving images that contain additional
    species.
    
    Args:
        train_df: DataFrame containing training annotations with 'label' and 'image_path' columns
        ratio: Float between 0 and 1 indicating what fraction of top-class-only images to keep
    
    Returns:
        DataFrame with undersampled annotations
        
    Example:
        >>> train_df = pd.DataFrame({
        ...     'label': ['Bird', 'Bird', 'Rare', 'Bird'],
        ...     'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img3.jpg']
        ... })
        >>> undersampled_df = undersample(train_df, ratio=0.5)
    """
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be between 0 and 1")
    
    # Find images that only have top two most common classes
    top_two_classes = train_df.label.value_counts().index[:2]
    top_two_labels = train_df[train_df["label"].isin(top_two_classes)]
    
    # Remove images that have any other classes
    top_two_images = train_df[train_df.image_path.isin(top_two_labels.image_path.unique())]
    with_additional_species = top_two_images[~top_two_images["label"].isin(top_two_classes)].image_path.unique()
    images_to_remove = [x for x in top_two_images.image_path.unique() if x not in with_additional_species]
    images_to_remove = images_to_remove[:int(len(with_additional_species)*ratio)]
    train_df = train_df[~train_df["image_path"].isin(images_to_remove)]

    return train_df

def preprocess_images(
    annotations: pd.DataFrame,
    root_dir: str,
    save_dir: str,
    limit_empty_frac: float = 0.1,
    patch_size: int = 450,
    patch_overlap: int = 0,
    allow_empty: bool = False
) -> pd.DataFrame:
    """
    Cut images into GPU-friendly chunks and process annotations accordingly.
    
    This function splits large images into smaller patches and adjusts their
    annotations to match the new coordinates. It also handles empty patches
    and maintains a balanced dataset.
    
    Args:
        annotations: DataFrame containing image annotations
        root_dir: Root directory containing the original images
        save_dir: Directory to save processed image patches
        limit_empty_frac: Maximum fraction of empty patches to keep
        patch_size: Size of the output patches in pixels
        patch_overlap: Overlap between patches in pixels
        allow_empty: Whether to allow patches without annotations
    
    Returns:
        DataFrame containing annotations for the processed image patches
        
    Raises:
        FileNotFoundError: If root_dir or image files don't exist
        ValueError: If patch_size <= 0 or patch_overlap < 0
    """
    if patch_size <= 0 or patch_overlap < 0:
        raise ValueError("Invalid patch_size or patch_overlap")
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # Remove any annotations with xmin == xmax
    #annotations = annotations[annotations.xmin != annotations.xmax]
    
    os.makedirs(save_dir, exist_ok=True)
    
    crop_annotations = []

    for image_path in annotations.image_path.unique():
        annotation_df = annotations[annotations.image_path == image_path]

        crop_annotation = process_image(
            image_path=image_path,
            annotation_df=annotation_df,
            root_dir=root_dir,
            save_dir=save_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            allow_empty=allow_empty
        )
        crop_annotations.append(crop_annotation)

    crop_annotations = pd.concat(crop_annotations)
    return crop_annotations

def process_image(
    image_path: str,
    annotation_df: Optional[pd.DataFrame],
    root_dir: str,
    save_dir: str,
    patch_size: int,
    patch_overlap: int,
    allow_empty: bool
) -> pd.DataFrame:
    """
    Process a single image by splitting it into patches and adjusting annotations.
    
    Args:
        image_path: Path to the image file
        annotation_df: DataFrame containing annotations for this image, or None if empty
        root_dir: Root directory containing the original images
        save_dir: Directory to save processed image patches
        patch_size: Size of the output patches in pixels
        patch_overlap: Overlap between patches in pixels
        allow_empty: Whether to allow patches without annotations
    
    Returns:
        DataFrame containing annotations for the processed image patches
        
    Note:
        If the crops already exist in save_dir, they will be skipped and
        the existing annotations will be returned.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_csv = "{}.csv".format(os.path.join(save_dir, image_name))
    
    if os.path.exists(crop_csv):
        return pd.read_csv(crop_csv)
        
    full_path = os.path.join(root_dir, image_path)
    
    crop_annotation = preprocess.split_raster(
        path_to_raster=full_path,
        annotations_file=annotation_df,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        save_dir=save_dir,
        root_dir=root_dir,
        allow_empty=allow_empty
    )

    # Split FalsePositives out for hard negative mining, for those images with no true positives
    true_positive_images = crop_annotation.loc[crop_annotation.label != "FalsePositive", "image_path"].unique()
    
    # Remove any FalsePositives that are in true positive images
    crop_annotation = crop_annotation.loc[~((crop_annotation.label == "FalsePositive") & (crop_annotation.image_path.isin(true_positive_images))), :]
    crop_annotation.loc[crop_annotation.label == "FalsePositive", ["xmin", "ymin", "xmax", "ymax"]] = 0
    crop_annotation["label"] = crop_annotation["label"].astype(str)
    crop_annotation.loc[crop_annotation.label == "FalsePositive", "label"] = "Bird"
    crop_annotation.loc[crop_annotation.label == "0", "label"] = "Bird"

    # Update geometry for FalsePositives
    crop_annotation.drop(columns=["geometry"], inplace=True)
    crop_annotation = pd.DataFrame(crop_annotation)
    crop_annotation = read_file(crop_annotation)
    
    # Remove duplicates
    crop_annotation = crop_annotation.drop_duplicates(subset=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    # Remove FalsePositives that are in true positive images
    crop_annotation = crop_annotation.loc[~((crop_annotation.label == "FalsePositive") & (crop_annotation.image_path.isin(true_positive_images))), :]
    
    # Save over the original csv
    crop_annotation.to_csv(crop_csv, index=False)

    if annotation_df is None:
        empty_annotations = []
        for i in range(len(crop_annotation)):
            empty_annotation = pd.DataFrame({
                "image_path": os.path.basename(crop_annotation[i]),
                "xmin": [None],
                "xmax": [None],
                "ymin": [None],
                "ymax": [None],
            })
            empty_annotations.append(empty_annotation)
        empty_annotations = pd.concat(empty_annotations)
        empty_annotations.root_dir = root_dir
        return empty_annotations
    else:   
        return crop_annotation

def density_cropping(
    predictions: pd.DataFrame,
    image_path: str,
    min_density: int = 3,
    eps: float = 50,
    min_samples: int = 3,
    padding: int = 100
) -> Dict[str, List[Dict]]:
    """
    Create crops around dense areas of detections using clustering and convex hulls.
    
    Args:
        predictions: DataFrame with columns ['xmin', 'ymin', 'xmax', 'ymax']
        image_path: Path to the original image
        min_density: Minimum number of detections to consider an area dense
        eps: Maximum distance between points for DBSCAN clustering
        min_samples: Minimum samples per cluster for DBSCAN
        padding: Padding around hull in pixels
        
    Returns:
        Dictionary containing:
            'crops': List of dictionaries with crop coordinates and paths
            'clusters': List of cluster assignments for each detection
    """
    if len(predictions) < min_density:
        return {'crops': [], 'clusters': []}
    
    # Get centers of bounding boxes
    centers = np.array([
        [(row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2]
        for _, row in predictions.iterrows()
    ])
    
    # Cluster centers using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # Process each cluster
    crops = []
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points
    
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_height, img_width = img.shape[:2]
    
    for label in unique_labels:
        cluster_points = centers[labels == label]
        
        if len(cluster_points) >= min_density:
            # Create convex hull around cluster points
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            
            # Get bounding box of hull
            xmin = max(0, int(np.min(hull_points[:, 0]) - padding))
            ymin = max(0, int(np.min(hull_points[:, 1]) - padding))
            xmax = min(img_width, int(np.max(hull_points[:, 0]) + padding))
            ymax = min(img_height, int(np.max(hull_points[:, 1]) + padding))
            
            # Create crop
            crop = img[ymin:ymax, xmin:xmax]
            
            # Generate crop filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            crop_name = f"{base_name}_cluster_{label}.jpg"
            crop_path = os.path.join(os.path.dirname(image_path), "crops", crop_name)
            
            # Ensure crops directory exists
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            
            # Save crop
            cv2.imwrite(crop_path, crop)
            
            crops.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'path': crop_path,
                'num_detections': len(cluster_points)
            })
    
    return {
        'crops': crops,
        'clusters': labels.tolist()
    }

def adjust_coordinates(
    predictions: pd.DataFrame,
    crop_info: Dict[str, int]
) -> pd.DataFrame:
    """
    Adjust coordinates of predictions relative to crop boundaries.
    
    Args:
        predictions: DataFrame with bounding box coordinates
        crop_info: Dictionary with crop boundaries (xmin, ymin, xmax, ymax)
        
    Returns:
        DataFrame with adjusted coordinates
    """
    adjusted = predictions.copy()
    adjusted['xmin'] = adjusted['xmin'] - crop_info['xmin']
    adjusted['xmax'] = adjusted['xmax'] - crop_info['xmin']
    adjusted['ymin'] = adjusted['ymin'] - crop_info['ymin']
    adjusted['ymax'] = adjusted['ymax'] - crop_info['ymin']
    return adjusted

def merge_crop_predictions(
    crops: List[Dict],
    predictions: pd.DataFrame,
    labels: List[int]
) -> pd.DataFrame:
    """
    Merge predictions from multiple crops back into original image coordinates.
    
    Args:
        crops: List of crop information dictionaries
        predictions: Original predictions DataFrame
        labels: Cluster labels for each prediction
        
    Returns:
        DataFrame with merged predictions
    """
    merged = []
    
    for i, crop in enumerate(crops):
        # Get predictions for this cluster
        cluster_mask = np.array(labels) == i
        cluster_preds = predictions[cluster_mask].copy()
        
        # Adjust coordinates back to original image space
        cluster_preds['xmin'] += crop['xmin']
        cluster_preds['xmax'] += crop['xmin']
        cluster_preds['ymin'] += crop['ymin']
        cluster_preds['ymax'] += crop['ymin']
        
        merged.append(cluster_preds)
    
    # Add predictions that weren't in any cluster
    noise_mask = np.array(labels) == -1
    if noise_mask.any():
        merged.append(predictions[noise_mask])
    
    return pd.concat(merged, ignore_index=True)