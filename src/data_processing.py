import pandas as pd
import os
from logging import warn
from deepforest import preprocess
from typing import Optional, Union, List

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
    patch_overlap: int = 0
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
    
    os.makedirs(save_dir, exist_ok=True)
    
    crop_annotations = []

    for image_path in annotations.image_path.unique():
        annotation_df = annotations[annotations.image_path == image_path]
        
        if annotation_df.empty:
            allow_empty = True
        else:
            allow_empty = False

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
        warn("Crops for {} already exist in {}. Skipping.".format(crop_csv, save_dir))
        return pd.read_csv(crop_csv)
        
    full_path = os.path.join(root_dir, image_path)
    
    # Check if all xmin values are 0, indicating empty annotations
    if annotation_df is not None and all(annotation_df['xmin'] == 0):
        allow_empty = True
    else:
        allow_empty = False
    
    crop_annotation = preprocess.split_raster(
        path_to_raster=full_path,
        annotations_file=annotation_df,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        save_dir=save_dir,
        root_dir=root_dir,
        allow_empty=allow_empty
    )
    
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
