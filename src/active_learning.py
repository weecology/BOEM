import glob
import os
import random
from src import detection
import dask.array as da
import pandas as pd

def choose_train_images(evaluation, image_dir, strategy, n=10, patch_size=512, patch_overlap=0.1, min_score=0.1, model=None, model_path=None, classification_model=None, dask_client=None, target_labels=None, pool_limit=1000, batch_size=16, selected_test_images=[], comet_logger=None):
    """
    Choose images to annotate.
    Args:
        evaluation (dict): A dictionary of evaluation metrics.
        image_dir (str): The path to a directory of images.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
            - "target-labels": Choose images with target labels.
        n (int, optional): The number of images to choose. Defaults to 10.
        dask_client (dask.distributed.Client, optional): A Dask client for parallel processing. Defaults to None.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None. 
        classification_model (main.deepforest, optional): A trained deepforest model for classification. Defaults to None.
        model_path (str, optional): The path to the model checkpoint file. Defaults to None. Only used in combination with dask
        target_labels: (list, optional): A list of target labels to filter images by. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
        batch_size (int, optional): The batch size for prediction. Defaults to 16.
        selected_test_images (list, optional): A list of test images that have already been selected. Defaults to [].
        comet_logger (CometLogger, optional): A CometLogger object. Defaults to None.
    Returns:
        list: A list of image paths.
        pd.DataFrame: A DataFrame of preannotations.
    """
    pool = glob.glob(os.path.join(image_dir,"*.jpg")) # Get all images in the data directory
    
    # Remove .csv files from the pool
    pool = [image for image in pool if not image.endswith('.csv')]
    
    # Remove crop dir
    try:
        pool.remove(os.path.join(image_dir,"crops"))
    except ValueError:
        pass

    #subsample
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    # Don't allow any test images that have just been selected to be chosen
    pool = [x for x in pool if not x in selected_test_images]

    if strategy=="random":
        chosen_images = random.sample(pool, n)
        return chosen_images    
    elif strategy in ["most-detections","target-labels"]:
        # Predict all images
        if model_path is None and model is None:
            raise ValueError("A model is required for the 'most-detections' or 'target-labels' strategy. Either pass a model or a model_path.")
        if dask_client:
            # load model on each client
            def update_sys_path():
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dask_client.run(update_sys_path)

            # Load model on each client
            dask_pool = da.from_array(pool, chunks=len(pool)//len(dask_client.ncores()))
            blocks = dask_pool.to_delayed().ravel()
            block_futures = []
            for block in blocks:
                block_future = dask_client.submit(detection.predict,image_paths=block.compute(), patch_size=patch_size, patch_overlap=patch_overlap, model_path=model_path)
                block_futures.append(block_future)
            # Get results
            dask_results = []
            for block_result in block_futures:
                block_result = block_result.result()
                dask_results.append(pd.concat(block_result))
            preannotations = pd.concat(dask_results)
        else:
            preannotations = detection.predict(m=model, image_paths=pool, patch_size=patch_size, patch_overlap=patch_overlap, batch_size=batch_size)
            preannotations = pd.concat(preannotations)

        if comet_logger:
            comet_logger.log_table("active_training_pool", preannotations)

        # Print the number of preannotations before removing min score
        preannotations = preannotations[preannotations["score"] >= min_score]

        if strategy == "most-detections":
            # Sort images by total number of predictions
            chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "target-labels":
            if target_labels is None:
                raise ValueError("Target labels are required for the 'target-labels' strategy.")
            # Filter images by target labels
            chosen_images = preannotations[preannotations.label.isin(target_labels)].groupby("image_path")["score"].mean().sort_values(ascending=False).head(n).index.tolist()
        else:
            raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")
        # Get full path
        chosen_images = [os.path.join(image_dir, image) for image in chosen_images]
    else:
        raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]

    return chosen_images, chosen_preannotations

def choose_test_images(image_dir, strategy, n=10, patch_size=512, patch_overlap=0, min_score=0.5, model=None, model_path=None, dask_client=None, target_labels=None, pool_limit=1000, batch_size=1, comet_logger=None):
    """Choose images to annotate.
    Args:
        evaluation (dict): A dictionary of evaluation metrics.
        image_dir (str): The path to a directory of images.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
            - "target-labels": Choose images with target labels.
        n (int, optional): The number of images to choose. Defaults to 10.
        dask_client (dask.distributed.Client, optional): A Dask client for parallel processing. Defaults to None.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None. 
        model_path (str, optional): The path to the model checkpoint file. Defaults to None. Only used in combination with dask
        target_labels: (list, optional): A list of target labels to filter images by. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
        batch_size (int, optional): The batch size for prediction. Defaults to 1.
        comet_logger (CometLogger, optional): A CometLogger object. Defaults to None.
    Returns:
        list: A list of image paths.
        pd.DataFrame: A DataFrame of preannotations.
    """
    pool = glob.glob(os.path.join(image_dir,"*")) # Get all images in the data directory
    # Remove .csv files from the pool
    pool = [image for image in pool if not image.endswith('.csv')]
    
    # Remove crop dir
    try:
        pool.remove(os.path.join(image_dir,"crops"))
    except ValueError:
        pass

    #subsample
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    if strategy=="random":
        chosen_images = random.sample(pool, n)
        preannotations = None
        return chosen_images, None    
    elif strategy in ["most-detections","target-labels"]:
        # Predict all images
        if model_path is None:
            raise ValueError("A model is required for the 'most-detections' or 'target-labels' strategy.")
        if dask_client:
            # load model on each client
            def update_sys_path():
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dask_client.run(update_sys_path)

            # Load model on each client
            dask_pool = da.from_array(pool, chunks=len(pool)//len(dask_client.ncores()))
            blocks = dask_pool.to_delayed().ravel()
            block_futures = []
            for block in blocks:
                block_future = dask_client.submit(detection.predict,image_paths=block.compute(), patch_size=patch_size, patch_overlap=patch_overlap, model_path=model_path)
                block_futures.append(block_future)
            # Get results
            dask_results = []
            for block_result in block_futures:
                block_result = block_result.result()
                dask_results.append(pd.concat(block_result))
            preannotations = pd.concat(dask_results)
        else:
            preannotations = detection.predict(model=model, image_paths=pool, patch_size=patch_size, patch_overlap=patch_overlap, batch_size=batch_size)
            preannotations = pd.concat(preannotations)
        
        if comet_logger:
            comet_logger.log_table("active_testing_pool", preannotations)

        print("There are {} preannotations before removing min score".format(preannotations.shape[0]))
        preannotations = preannotations[preannotations["score"] >= min_score]

        if strategy == "most-detections":
            # Sort images by total number of predictions
            chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "target-labels":
            # Filter images by target labels
            chosen_images = preannotations[preannotations.label.isin(target_labels)].groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        else:
            raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")
        # Get full path
        chosen_images = [os.path.join(image_dir, image) for image in chosen_images]
    else:
        raise ValueError("Invalid strategy. Must be one of 'random', 'most-detections', or 'target-labels'.")

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]
    return chosen_images, chosen_preannotations

def human_review(detection_model, classification_model, image_paths, patch_size, patch_overlap, confident_threshold, min_score, batch_size):
    """
    Predict on images and divide into confident and uncertain predictions.
    Args:
        detection_model (deepforest.deepforest): A trained detection model.
        classification_model (deepforest.deepforest): A trained classification model.
        image_paths (list): A list of image paths.
        patch_size (int): The size of the image patches to predict on.
        patch_overlap (float): The amount of overlap between image patches.
        confident_threshold (float): The threshold for confident predictions.
        min_score (float): The minimum score for a prediction to be included.
        batch_size (int): The batch size for prediction.
        existing_predictions (pd.DataFrame, optional): A DataFrame of existing predictions. Defaults to None.
        Returns:
        tuple: A tuple of confident and uncertain predictions.
        """
    # Check for existing predictions
    if existing_predictions is not None:
        image_basenames = [os.path.basename(image_path) for image_path in image_paths]
        existing_predictions = existing_predictions[existing_predictions["image_path"].isin(image_basenames)]
        image_paths = [image_path for image_path in image_paths if os.path.basename(image_path) not in existing_predictions["image_path"].unique()]
    if len(image_paths) > 0:
        predictions = detection.predict(
            m=detection_model,
            crop_model=classification_model,
            image_paths=image_paths,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size
        )
        predictions = pd.concat(predictions)
        combined_predictions = pd.concat([predictions, existing_predictions])
    else:
        combined_predictions = existing_predictions

    combined_predictions[combined_predictions["score"] > min_score]

    # Split predictions into confident and uncertain
    uncertain_predictions = combined_predictions[
        combined_predictions["score"] <= confident_threshold]
    
    confident_predictions = combined_predictions[
        ~combined_predictions["image_path"].isin(
            uncertain_predictions["image_path"])]
    
    return confident_predictions, uncertain_predictions

def generate_training_pool_predictions(image_dir, patch_size=512, patch_overlap=0.1, min_score=0.1, model=None, model_path=None, dask_client=None, batch_size=16, comet_logger=None):
    """
    Generate predictions for the training pool.
    
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
    
    Returns:
        pd.DataFrame: A DataFrame of predictions.
    """
    pool = glob.glob(os.path.join(image_dir, "*.jpg"))  # Get all images in the data directory
    
    # Remove .csv files from the pool
    pool = [image for image in pool if not image.endswith('.csv')]
    
    # Remove crop dir
    try:
        pool.remove(os.path.join(image_dir, "crops"))
    except ValueError:
        pass

    if dask_client:
        # load model on each client
        def update_sys_path():
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dask_client.run(update_sys_path)

        # Load model on each client
        dask_pool = da.from_array(pool, chunks=len(pool) // len(dask_client.ncores()))
        blocks = dask_pool.to_delayed().ravel()
        block_futures = []
        for block in blocks:
            block_future = dask_client.submit(detection.predict, image_paths=block.compute(), patch_size=patch_size, patch_overlap=patch_overlap, model_path=model_path)
            block_futures.append(block_future)
        # Get results
        dask_results = []
        for block_result in block_futures:
            block_result = block_result.result()
            dask_results.append(pd.concat(block_result))
        preannotations = pd.concat(dask_results)
    else:
        preannotations = detection.predict(m=model, image_paths=pool, patch_size=patch_size, patch_overlap=patch_overlap, batch_size=batch_size)
        preannotations = pd.concat(preannotations)

    if comet_logger:
        comet_logger.log_table("active_training_pool", preannotations)

    # Print the number of preannotations before removing min score
    preannotations = preannotations[preannotations["score"] >= min_score]

    return preannotations

def select_train_images(preannotations, strategy, n=10, target_labels=None):
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
    
    Returns:
        list: A list of image paths.
        pd.DataFrame: A DataFrame of preannotations for the chosen images.
    """
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