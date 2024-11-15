import pandas as pd
import os
from logging import warn
from deepforest import preprocess

def undersample(train_df, ratio):
    """Undersample top classes by selecting most diverse images"""
    
    # Find images that only have top two most common classes
    top_two_classes = train_df.label.value_counts().index[:2]
    top_two_labels = train_df[train_df["label"].isin(top_two_classes)]
    
    # remove images that have any other classes
    top_two_images= train_df[train_df.image_path.isin(top_two_labels.image_path.unique())]
    with_additional_species = top_two_images[~top_two_images["label"].isin(top_two_classes)].image_path.unique()
    images_to_remove = [x for x in top_two_images.image_path.unique() if x not in with_additional_species][:int(len(with_additional_species)*ratio)]
    train_df = train_df[~train_df["image_path"].isin(images_to_remove)]

    return train_df

def preprocess_images(annotations, root_dir, save_dir, limit_empty_frac=0.1, patch_size=450, patch_overlap=0):
    """Cut images into GPU friendly chunks"""
    crop_annotations = []

    for image_path in annotations.image_path.unique():
        annotation_df = annotations[annotations.image_path == image_path]
        annotation_df = annotation_df[~annotation_df.xmin.isnull()]
        if annotation_df.empty:
            allow_empty = True
            annotation_df = None
        else:
            allow_empty = False
        crop_annotation = process_image(
            image_path, 
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

def process_image(image_path, annotation_df, root_dir, save_dir, patch_size, patch_overlap, allow_empty):
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
