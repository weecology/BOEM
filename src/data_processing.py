import pandas as pd
import os
from logging import warn
from deepforest import preprocess

def preprocess_images(annotations, root_dir, save_dir, limit_empty_frac=0.1, patch_size=450, patch_overlap=0):
    """Cut images into GPU friendly chunks"""
    crop_annotations = []

    for image_path in annotations.image_path.unique():
        crop_annotation = process_image(image_path, annotation_df=annotations, root_dir=root_dir, save_dir=save_dir, 
                                        limit_empty_frac=limit_empty_frac, patch_size=patch_size, patch_overlap=patch_overlap)
        crop_annotations.append(crop_annotation)

    crop_annotations = pd.concat(crop_annotations)
    crop_annotation_empty = crop_annotations.loc[crop_annotations.xmin==0]
    crop_annotation_non_empty = crop_annotations.loc[crop_annotations.xmin!=0]
    crop_annotation_empty = crop_annotation_empty.sample(frac=limit_empty_frac)
    crop_annotations = pd.concat([crop_annotation_empty, crop_annotation_non_empty])
    
    return crop_annotations

def process_image(image_path, annotation_df, root_dir, save_dir, limit_empty_frac, patch_size, patch_overlap):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_csv = "{}.csv".format(os.path.join(save_dir, image_name))
    if os.path.exists(crop_csv):
        warn("Crops for {} already exist in {}. Skipping.".format(crop_csv, save_dir))
        return pd.read_csv(crop_csv)
    full_path = os.path.join(root_dir, image_path)
    
    crop_annotation = preprocess.split_raster(
        path_to_raster=full_path,
        annotations_file=annotation_df,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        save_dir=save_dir,
        root_dir=root_dir,
        allow_empty=True
    )

    return crop_annotation