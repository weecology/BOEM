# Standard library imports
import glob
import os
import random
import tempfile
import warnings
from logging import warn
import math

# Third party imports
import dask.array as da
import pandas as pd
from deepforest import main, visualize
from deepforest.utilities import read_file
from pytorch_lightning.loggers import CometLogger
import geopandas as gpd

# Local imports
from src import data_processing
from src.label_studio import gather_data
from omegaconf import OmegaConf

def evaluate(model, test_csv, image_root_dir):
    """Evaluate a model on labeled images.
    
    Args:
        model (main.deepforest): A trained deepforest model.
        test_csv (str): The path to a CSV file containing annotations.
        image_root_dir (str): The directory containing the test images.
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # create trainer
    model.create_trainer()
    model.config["validation"]["csv_file"] = test_csv
    model.config["validation"]["root_dir"] = image_root_dir
    results = model.trainer.validate(model)

    return results

def load(checkpoint, annotations = None):
    """Load a trained model from disk.
    
    Args:
        checkpoint (str): The path to a model checkpoint.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    # Check classes
    if checkpoint == "tree":
        snapshot = main.deepforest()
        snapshot.use_release()
    elif checkpoint == "bird":
        snapshot = main.deepforest(label_dict={"Bird":0})
        snapshot.use_bird_release()
    else:
        snapshot = main.deepforest.load_from_checkpoint(checkpoint)

    if not annotations is None:
        num_labels = len(annotations.label.unique())
        if num_labels != len(snapshot.label_dict):
            snapshot = extract_backbone(snapshot, annotations)

    return snapshot

def extract_backbone(snapshot, annotations):
    warnings.warn("The number of classes in the model does not match the number of classes in the annotations. The backbone will be extracted and retrained.")
    new_labels = annotations.label.unique()
    new_labels = new_labels[~pd.isnull(new_labels)]

    label_dict = {value: index for index, value in enumerate(new_labels)}
    m = main.deepforest(num_classes=len(new_labels), label_dict=label_dict, config_file="deepforest_config.yml")
    m.model.backbone.load_state_dict(snapshot.model.backbone.state_dict())
    m.model.head.regression_head.load_state_dict(snapshot.model.head.regression_head.state_dict())

    return m

def create_train_test(annotations, train_test_split = 0.1):
    """Create a train and test set from annotations.
    
    Args:
        annotations (pd.DataFrame): A DataFrame containing annotations.
        train_test_split (float): The fraction of the data to use for validation.
    Returns:
        pd.DataFrame: A DataFrame containing training annotations.
        pd.DataFrame: A DataFrame containing validation annotations.
    """
    # split train images into 90% train and 10% validation for each class as much as possible
    test_images = []
    validation_df = None

    if annotations.label.value_counts().shape[0] > 1:
        for label in annotations.label.value_counts().sort_values().index.values:
            # is the current count already higher than 10% of the total count?
            if validation_df is not None:
                if label in validation_df.label.value_counts().index:
                    if validation_df.label.value_counts()[label] > annotations.label.value_counts()[label] * train_test_split:
                        continue
            images = list(annotations[annotations["label"] == label]["image_path"].unique())
            label_test_images = random.sample(images, int(len(images)*train_test_split))
            test_images.extend(label_test_images)
            validation_df = annotations[annotations["image_path"].isin(test_images)]
    else:
        sample_size = math.ceil(len(annotations)*train_test_split)
        test_images = random.sample(list(annotations.image_path.unique()),sample_size)
        validation_df = annotations[annotations["image_path"].isin(test_images)]

    # Save validation csv
    train_df = annotations[~annotations["image_path"].isin(test_images)]

    return train_df, validation_df

def limit_empty_frames(crop_annotations, limit_empty_frac):
    crop_annotation_empty = crop_annotations.loc[crop_annotations.xmin==0]
    crop_annotation_non_empty = crop_annotations.loc[crop_annotations.xmin!=0]
    crop_annotation_empty = crop_annotation_empty.sample(frac=limit_empty_frac)
    crop_annotations = pd.concat([crop_annotation_empty, crop_annotation_non_empty])

    return crop_annotations

def train(model, train_annotations, test_annotations, train_image_dir, comet_project=None, comet_workspace=None, config_args=None):
    """Train a model on labeled images.
    Args:
        image_paths (list): A list of image paths.
        train_annotations (pd.DataFrame): A DataFrame containing annotations.
        test_annotations (pd.DataFrame): A DataFrame containing annotations.
        train_image_dir (str): The directory containing the training images.
        comet_project (str): The comet project name for logging. Defaults to None.
        comet_workspace (str): The comet workspace for logging. Defaults to None.
        config_args (dict): A dictionary of configuration arguments to update the model.config. Defaults to None.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    tmpdir = tempfile.gettempdir()

    train_annotations.to_csv(os.path.join(tmpdir,"train.csv"), index=False)

    # Set config
    model.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
    model.config["train"]["root_dir"] = train_image_dir

    # Loop through all keys in model.config and set them to the value of the key in model.config
    config_args = OmegaConf.to_container(config_args)
    if config_args:
        for key, value in config_args.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    model.config[key][subkey] = subvalue
            else:
                model.config[key] = value

    if comet_project:
        comet_logger = CometLogger(project_name=comet_project, workspace=comet_workspace)
        comet_logger.experiment.add_tags(["detection"])
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", train_annotations)
        comet_logger.experiment.log_table("test.csv", test_annotations)

        model.create_trainer(logger=comet_logger)
    else:
        model.create_trainer()

    # with comet_logger.experiment.context_manager("train_images"):
    #     non_empty_train_annotations = train_annotations[~(train_annotations.xmax==0)]
    #     try:
    #         non_empty_train_annotations= gpd.GeoDataFrame(non_empty_train_annotations, geometry=non_empty_train_annotations["geometry"])
    #         non_empty_train_annotations.root_dir = train_image_dir
    #         non_empty_train_annotations = read_file(non_empty_train_annotations)
    #     except: 
    #         non_empty_train_annotations = read_file(non_empty_train_annotations, root_dir=train_image_dir)

    #     if non_empty_train_annotations.empty:
    #         pass
    #     else:
    #         sample_train_annotations = non_empty_train_annotations[non_empty_train_annotations.image_path.isin(non_empty_train_annotations.image_path.head(5))]
    #         for filename in sample_train_annotations.image_path:
    #             sample_train_annotations_for_image = sample_train_annotations[sample_train_annotations.image_path == filename]
    #             sample_train_annotations_for_image.root_dir = train_image_dir
    #             visualize.plot_results(sample_train_annotations_for_image, savedir=tmpdir)
    #             comet_logger.experiment.log_image(os.path.join(tmpdir, filename))

    model.trainer.fit(model)

    with comet_logger.experiment.context_manager("post-training prediction"):
        for image_path in test_annotations.image_path.head(5):
            prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
            if prediction is None:
                continue
            visualize.plot_results(prediction, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))

    comet_logger.experiment.end()
    model.trainer.logger.experiment.end()
    
    return model

def preprocess_and_train(config, model_type="detection"):
    """Preprocess data and train model.
    
    Args:
        config: Configuration object containing training parameters
        model_type (str): The type of model to train. Defaults to "detection".
    Returns:
        trained_model: Trained model object
    """
    # Get and split annotations
    train_df = gather_data(config.detection_model.train_csv_folder)
    validation_df = gather_data(config.label_studio.csv_dir_validation)
    validation_df.loc[validation_df.label==0,"label"] = "Bird"

    # Preprocess train and validation data
    train_df = data_processing.preprocess_images(train_df,
                               root_dir=config.detection_model.train_image_dir,
                               save_dir=config.detection_model.crop_image_dir)
    
    non_empty = train_df[train_df.xmin!=0]
    train_df.loc[train_df.label==0,"label"] = "Bird"

    if not validation_df.empty:
        validation_df = data_processing.preprocess_images(validation_df,
                                    root_dir=config.detection_model.train_image_dir,
                                    save_dir=config.detection_model.crop_image_dir)
        non_empty = validation_df[validation_df.xmin!=0]
        validation_df.loc[validation_df.label==0,"label"] = "Bird"

    # Limit empty frames
    if config.detection_model.limit_empty_frac > 0:
        train_df = limit_empty_frames(train_df, config.detection_model.limit_empty_frac)
        if not validation_df.empty:
            validation_df = limit_empty_frames(validation_df, config.detection_model.limit_empty_frac)

    # Train model
    # Load existing model
    if config.detection_model.checkpoint:
        loaded_model = load(config.detection_model.checkpoint, annotations=train_df)
    elif os.path.exists(config.detection_model.checkpoint_dir):
        loaded_model = get_latest_checkpoint(config.detection_model.checkpoint_dir, train_df)
    else:
        raise ValueError("No checkpoint or checkpoint directory found.")

    trained_model = train(train_annotations=train_df,
                            test_annotations=validation_df,
                            train_image_dir=config.detection_model.crop_image_dir,
                            model=loaded_model,
                            comet_project=config.comet.project,
                            comet_workspace=config.comet.workspace,
                            config_args=config.detection_model.trainer)

    return trained_model

def get_latest_checkpoint(checkpoint_dir, annotations):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = load(checkpoint)
        else:
            warn("No checkpoints found in {}".format(checkpoint_dir))
            label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
            m = main.deepforest(label_dict=label_dict)
    else:
        os.makedirs(checkpoint_dir)
        label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
        m = main.deepforest(label_dict=label_dict)

    return m

def _predict_list_(image_paths, patch_size, patch_overlap, model_path, m=None, crop_model=None):
    if model_path:
        m = load(model_path)
    else:
        if m is None:
            raise ValueError("A model or model_path is required for prediction.")

    m.create_trainer(fast_dev_run=False)

    predictions = []
    for image_path in image_paths:
        prediction = m.predict_tile(raster_path=image_path, return_plot=False, patch_size=patch_size, patch_overlap=patch_overlap, crop_model=crop_model)
        if prediction is None:
            prediction = pd.DataFrame({"image_path": image_path, "xmin": [None], "ymin": [None], "xmax": [None], "ymax": [None], "label": [None], "score": [None]})
        predictions.append(prediction)

    return predictions

def predict(image_paths, patch_size, patch_overlap, m=None, model_path=None, dask_client=None, crop_model=None):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.          
        crop_model (main.deepforest): A trained deepforest model for classification.
        model_path (str): The path to a model checkpoint.
        dask_client (dask.distributed.Client): A dask client for parallel prediction.
    Returns:
        list: A list of image predictions.
    """
    if dask_client:
        # load model on each client
        def update_sys_path():
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dask_client.run(update_sys_path)

        # Load model on each client
        dask_pool = da.from_array(image_paths, chunks=len(image_paths)//len(dask_client.ncores()))
        blocks = dask_pool.to_delayed().ravel()
        block_futures = []
        for block in blocks:
            block_future = dask_client.submit(_predict_list_,
                                              image_paths=block.compute(),
                                              patch_size=patch_size,
                                              patch_overlap=patch_overlap,
                                              model_path=model_path,
                                              crop_model=crop_model)
            block_futures.append(block_future)
        # Get results
        predictions = []
        for block_result in block_futures:
            block_result = block_result.result()
            predictions.append(pd.concat(block_result))
    else:
        predictions = _predict_list_(image_paths=image_paths, patch_size=patch_size, patch_overlap=patch_overlap, model_path=model_path, m=m, crop_model=crop_model)

    return predictions
