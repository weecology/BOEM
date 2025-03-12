import glob
import os
import random
from src import detection
import pandas as pd
from deepforest.utilities import read_file
import geopandas as gpd

def human_review(predictions, min_score=0.2, confident_threshold=0.5):
    """
    Predict on images and divide into confident and uncertain predictions.
    Args:
        confident_threshold (float): The threshold for confident predictions.
        min_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.1.
        predictions (pd.DataFrame, optional): A DataFrame of existing predictions. Defaults to None.
        Returns:
        tuple: A tuple of confident and uncertain predictions.
        """
    
    predictions[predictions["cropmodel_score"] > min_score]

    # Split predictions into confident and uncertain
    uncertain_predictions = predictions[
        predictions["cropmodel_score"] <= confident_threshold]
    
    confident_predictions = predictions[
        ~predictions["image_path"].isin(
            uncertain_predictions["image_path"])]
    
    return confident_predictions, uncertain_predictions

def generate_pool_predictions(image_dir, patch_size=512, patch_overlap=0.1, min_score=0.1, model=None, model_path=None, dask_client=None, batch_size=16, pool_limit=1000, crop_model=None):
    """
    Generate predictions for the flight pool.
    
    Args:
        image_dir (str): The path to a directory of images.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None.
        model_path (str, optional): The path to the model checkpoint file. Defaults to None. Only used in combination with dask.
        dask_client (dask.distributed.Client, optional): A Dask client for parallel processing. Defaults to None.
        batch_size (int, optional): The batch size for prediction. Defaults to 16.
        comet_logger (CometLogger, optional): A CometLogger object. Defaults to None.
        crop_model (bool, optional): A deepforest.model.CropModel object. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
    
    Returns:
        pd.DataFrame: A DataFrame of predictions.
    """
    pool = glob.glob(os.path.join(image_dir, "*.jpg"))  # Get all images in the data directory
    
    # Remove .csv files from the pool
    pool = [image for image in pool if not image.endswith('.csv')]
    
    #subsample
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    # Remove crop dir
    try:
        pool.remove(os.path.join(image_dir, "crops"))
    except ValueError:
        pass

    preannotations = detection.predict(m=model, model_path=model_path, image_paths=pool, patch_size=patch_size, patch_overlap=patch_overlap, batch_size=batch_size, crop_model=crop_model)
    preannotations = pd.concat(preannotations)

    if preannotations.empty:
        return None
    
    try:
        preannotations = read_file(preannotations, image_dir)
    except TypeError:
        preannotations = gpd.GeoDataFrame(preannotations, geometry="geometry")

    return preannotations

def select_images(preannotations, strategy, n=10, target_labels=None, min_score=0.3):
    """
    Select images to annotate based on the strategy.
    
    Args:
        preannotations (pd.DataFrame): A DataFrame of predictions.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
            - "target-labels": Choose images with target labels.
        n (int, optional): The number of images to choose. Defaults to 10.
        target_labels (list, optional): A list of target labels to filter images by. Defaults to None.
        min_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.3.
    
    Returns:
        list: A list of image paths.
        pd.DataFrame: A DataFrame of preannotations for the chosen images.
    """
    preannotations = preannotations[preannotations["score"] >= min_score]
    
    if preannotations.empty:
        return [], None
    
    if strategy == "random":
        chosen_images = random.sample(preannotations["image_path"].unique().tolist(), n)
    elif strategy == "most-detections":
        # Sort images by total number of predictions
        chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
    elif strategy == "target-labels":
        if target_labels is None:
            raise ValueError("Target labels are required for the 'target-labels' strategy.")
        # Filter images by target labels
        chosen_images = preannotations[preannotations.label.isin(target_labels)].groupby("image_path")["score"].mean().sort_values(ascending=False).head(n).index.tolist()
    else:
        raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]

    return chosen_images, chosen_preannotations