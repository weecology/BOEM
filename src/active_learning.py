import random
from src import detection
from src import hierarchical

def human_review(predictions, min_detection_score=0.6, min_classification_score=0.5, confident_threshold=0.5):
    """
    Predict on images and divide into confident and uncertain predictions.
    Args:
        confident_threshold (float): The threshold for confident predictions.
        min_classification_score (float, optional): The minimum class score for a prediction to be included. Defaults to 0.1.
        min_detection_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.1.
        predictions (pd.DataFrame, optional): A DataFrame of existing predictions. Defaults to None.
        Returns:
        tuple: A tuple of confident and uncertain predictions.
        """
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

def generate_pool_predictions(pool, patch_size=512, patch_overlap=0.1, min_score=0.1, model=None, batch_size=16, pool_limit=1000, crop_model=None, hcast_model=None, image_dir=None, hcast_batch_size=None, hcast_workers=None):
    """
    Generate predictions for the flight pool.
    
    Args:
        pool (str): List of image paths to predict on.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None.
        batch_size (int, optional): The batch size for prediction. Defaults to 16.
        comet_logger (CometLogger, optional): A CometLogger object. Defaults to None.
        crop_model (bool, optional): A deepforest.model.CropModel object. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
        hcast_model (optional): H-CAST hierarchical model wrapper. Defaults to None.
        image_dir (str, optional): Root directory where images are located. Required if hcast_model is provided.
        hcast_batch_size (int, optional): Batch size for H-CAST classification. Defaults to 64.
        hcast_workers (int, optional): Number of workers for H-CAST DataLoader. Defaults to 4.
    
    Returns:
        pd.DataFrame: A DataFrame of predictions with both cropmodel and hcast columns (if hcast_model provided).
    """
    
    #subsample
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    preannotations = detection.predict(
        m=model,
        image_paths=pool,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model)
    
    if preannotations is None:
        return None

    preannotations = preannotations[preannotations["score"] >= min_score]

    # Apply hierarchical classification if hcast_model is provided
    if hcast_model is not None:
        if image_dir is None:
            raise ValueError("image_dir is required when hcast_model is provided")
        preannotations = hierarchical.classify_dataframe(
            predictions=preannotations,
            image_dir=image_dir,
            model=hcast_model,
            batch_size=hcast_batch_size,
            num_workers=hcast_workers,
        )

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
    
    if preannotations.empty:
        return [], None
    
    if strategy == "random":
        n = min(n, len(preannotations["image_path"].unique()))
        chosen_images = random.sample(preannotations["image_path"].unique().tolist(), n)

    else:
        preannotations = preannotations[preannotations["score"] >= min_score]

        if strategy == "most-detections":
            # Sort images by total number of predictions
            chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "target-labels":
            if target_labels is None:
                raise ValueError("Target labels are required for the 'target-labels' strategy.")
            # Filter images by target labels
            chosen_images = preannotations[preannotations.cropmodel_label.isin(target_labels)].groupby("image_path")["score"].mean().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "rarest":
            # Drop most common class
            preannotations = preannotations[~preannotations["cropmodel_label"].isin(preannotations["cropmodel_label"].value_counts().nlargest(1).index)]
            
            # Sort images by least common label
            label_counts = preannotations.groupby("cropmodel_label").size().sort_values(ascending=True)
            
            # Sort preannoations by least common label
            preannotations["label_count"] = preannotations["cropmodel_label"].map(label_counts)
            preannotations.sort_values("label_count", ascending=True, inplace=True)
            chosen_images = preannotations.drop_duplicates(subset=["image_path"], keep="first").head(n)["image_path"].tolist()
        else:
            raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]

    # Chosen preannotations is a dict with image_path as the key
    return chosen_images, chosen_preannotations