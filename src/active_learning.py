import random
from src import detection
import pandas as pd
import geopandas as gpd
import numpy as np

def human_review(predictions, min_detection_score=0.6, min_classification_score=0.5, confident_threshold=0.5):
    """
    Predict on images and divide into confident and uncertain predictions.
    Args:
        confident_threshold (float): The threshold for confident predictions.
        min_classification_score (float, optional): The minimum class score for a prediction to be included. Defaults to 0.5.
        min_detection_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.5.
        predictions (pd.DataFrame, optional): A DataFrame of existing predictions. Defaults to None.
        Returns:
        tuple: A tuple of confident and uncertain predictions.
    """
    # Check if predictions is None or empty
    if predictions is None or predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    filtered_predictions = predictions[
        (predictions["score"] >= min_detection_score) &
        (predictions["cropmodel_score"] < min_classification_score)
    ]

    # Split predictions into confident and uncertain
    uncertain_predictions = filtered_predictions[
        filtered_predictions["cropmodel_score"] <= confident_threshold]
    
    confident_predictions = filtered_predictions[
        ~filtered_predictions["image_path"].isin(
            uncertain_predictions["image_path"])]
    
    return confident_predictions, uncertain_predictions

def generate_pool_predictions(pool, patch_size=512, patch_overlap=0.1, min_score=0.1, model=None, model_path=None, dask_client=None, batch_size=16, pool_limit=1000, crop_model=None):
    """
    Generate predictions for the flight pool.
    
    Args:
        pool (str): List of image paths to predict on.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None.
        model_path (str, optional): The path to the model checkpoint file. Defaults to None. Only used in combination with dask.
        dask_client (dask.distributed.Client, optional): A Dask client for parallel processing. Defaults to None.
        batch_size (int, optional): The batch size for prediction. Defaults to 16.
        crop_model (bool, optional): A deepforest.model.CropModel object. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
    
    Returns:
        pd.DataFrame: A DataFrame of predictions.
    """
    if pool is None:
        return None

    #subsample
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    preannotations = detection.predict(
        m=model,
        model_path=model_path,
        image_paths=pool,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model,
        dask_client=dask_client)
    
    if len(preannotations) == 0:
        return None
    else:
        preannotations = pd.concat(preannotations)

    if preannotations.empty:
        return None
    
    preannotations = gpd.GeoDataFrame(preannotations, geometry="geometry")

    preannotations = preannotations[preannotations["score"] >= min_score]

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
    
    if preannotations is None or preannotations.empty:
        return [], None
    
    if strategy == "random":
        unique_imgs = preannotations["image_path"].unique().tolist()
        k = min(n, len(unique_imgs))
        chosen_images = random.sample(unique_imgs, k)

    else:
        preannotations = preannotations[preannotations["score"] >= min_score]

        if strategy == "most-detections":
            # Sort images by total number of predictions
            chosen_images = (
                preannotations
                .groupby("image_path")
                .size()
                .sort_values(ascending=False)
                .head(n)
                .index
                .tolist()
            )

        elif strategy == "target-labels":
            if target_labels is None:
                raise ValueError("Target labels are required for the 'target-labels' strategy.")
            # Filter images by target labels
            subset = preannotations[preannotations["cropmodel_label"].isin(target_labels)]
            if subset.empty:
                return [], None
            chosen_images = (
                subset
                .groupby("image_path")["score"]
                .mean()
                .sort_values(ascending=False)
                .head(n)
                .index
                .tolist()
            )

        elif strategy == "rarest":
            # Sort images by least common label
            label_counts = preannotations.groupby("cropmodel_label").size().sort_values(ascending=True)
            temp = preannotations.copy()
            temp["label_count"] = temp["cropmodel_label"].map(label_counts)
            temp.sort_values("label_count", ascending=True, inplace=True)
            chosen_images = (
                temp
                .drop_duplicates(subset=["image_path"], keep="first")
                .head(n)["image_path"]
                .tolist()
            )

        elif strategy == "uncertainty":
            # Images with classification scores closest to 0.5 (most uncertain)
            temp = preannotations.copy()
            if "cropmodel_score" in temp.columns:
                temp["uncertainty_score"] = np.abs(temp["cropmodel_score"] - 0.5)
            else:
                temp["uncertainty_score"] = np.abs(temp["score"] - 0.5)
            img_scores = temp.groupby("image_path")["uncertainty_score"].mean()
            chosen_images = img_scores.nsmallest(n).index.tolist()

        elif strategy == "qbc":
            # Query-By-Committee: combine random + uncertainty picks
            # 1) random pick
            unique_imgs = preannotations["image_path"].unique().tolist()
            k = min(n, len(unique_imgs))
            random_imgs = random.sample(unique_imgs, k)

            # 2) uncertainty pick
            temp = preannotations.copy()
            if "cropmodel_score" in temp.columns:
                temp["uncertainty_score"] = np.abs(temp["cropmodel_score"] - 0.5)
            else:
                temp["uncertainty_score"] = np.abs(temp["score"] - 0.5)
            img_scores = temp.groupby("image_path")["uncertainty_score"].mean()
            uncertain_imgs = img_scores.nsmallest(n).index.tolist()

            combined = list(dict.fromkeys(random_imgs + uncertain_imgs))
            chosen_images = combined[:n]

        else:
            raise ValueError(f"Invalid strategy '{strategy}'")

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]

    # Chosen preannotations is a dict with image_path as the key
    return chosen_images, chosen_preannotations