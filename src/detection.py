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
        if num_labels > len(snapshot.label_dict):
            snapshot = extract_backbone(snapshot, annotations)
    
    snapshot.label_dict  = {'Object': 0}
    snapshot.numeric_to_label_dict = {0: 'Object'}
    
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

def train(model, train_annotations, test_annotations, train_image_dir, comet_logger=None, config_args=None):
    """Train a model on labeled images.
    Args:
        image_paths (list): A list of image paths.
        train_annotations (pd.DataFrame): A DataFrame containing annotations.
        test_annotations (pd.DataFrame): A DataFrame containing annotations.
        train_image_dir (str): The directory containing the training images.
        model (main.deepforest): A trained deepforest model.
        comet_logger (CometLogger): A CometLogger instance for logging.
        config_args (dict): A dictionary of configuration arguments to update the model.config. Defaults to None.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    tmpdir = tempfile.gettempdir()

   # Fix taxonomy
    train_annotations = fix_taxonomy(train_annotations)
    test_annotations = fix_taxonomy(test_annotations)

    train_annotations.to_csv(os.path.join(tmpdir,"train.csv"), index=False)
    test_annotations.to_csv(os.path.join(tmpdir,"test.csv"), index=False)

    # Set config
    model.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
    model.config["train"]["root_dir"] = train_image_dir

    model.config["validation"]["csv_file"] = os.path.join(tmpdir,"test.csv")
    model.config["validation"]["root_dir"] = train_image_dir

    # Loop through all keys in model.config and set them to the value of the key in model.config
    config_args = OmegaConf.to_container(config_args)
    if config_args:
        for key, value in config_args.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    model.config[key][subkey] = subvalue
            else:
                model.config[key] = value

    if comet_logger:
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", train_annotations)
        comet_logger.experiment.log_table("test.csv", test_annotations)
        model.create_trainer(logger=comet_logger)
    else:
        model.create_trainer()

    with comet_logger.experiment.context_manager("train_images"):
        non_empty_train_annotations = read_file(model.config["train"]["csv_file"], root_dir=train_image_dir)
        # Sanity check for debug
        n = 5 if non_empty_train_annotations.shape[0] > 5 else non_empty_train_annotations.shape[0]
        for filename in non_empty_train_annotations.image_path.sample():
            sample_train_annotations_for_image = non_empty_train_annotations[non_empty_train_annotations.image_path == filename]
            sample_train_annotations_for_image.root_dir = train_image_dir
            visualize.plot_annotations(sample_train_annotations_for_image, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, filename))
    
    with comet_logger.experiment.context_manager("test_images"):
        non_empty_validation_annotations = read_file(model.config["validation"]["csv_file"], root_dir=train_image_dir)
        n = 5 if non_empty_validation_annotations.shape[0] > 5 else non_empty_validation_annotations.shape[0]
        for filename in non_empty_validation_annotations.image_path.head(5):
            sample_validation_annotations_for_image = non_empty_validation_annotations[non_empty_validation_annotations.image_path == filename]
            sample_validation_annotations_for_image.root_dir = train_image_dir
            visualize.plot_annotations(sample_validation_annotations_for_image, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, filename))

    with comet_logger.experiment.context_manager("detection"):
        model.trainer.fit(model)

    with comet_logger.experiment.context_manager("post-training prediction"):
        for image_path in test_annotations.image_path.head(5):
            prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
            if prediction is None:
                continue
            visualize.plot_results(prediction, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))
    
    return model

def fix_taxonomy(df):
    df["label"] = "Object"
    #df["label"] = df.label.replace('Turtle', 'Reptile')
    #df["label"] = df.label.replace('Cetacean', 'Mammal')

    return df

def preprocess_and_train(config, comet_logger=None):
    """Preprocess data and train model.
    
    Args:
        config: Configuration object containing training parameters
        comet_logger: CometLogger instance for logging.
    Returns:
        trained_model: Trained model object
    """
    # Get and split annotations
    train_df = gather_data(config.detection_model.train_csv_folder)
    validation = gather_data(config.label_studio.csv_dir_validation)

    if config.detection_model.limit_empty_frac > 0:
        validation = limit_empty_frames(validation, config.detection_model.limit_empty_frac)
    
    validation.loc[validation.label==0,"label"] = "Object"

    # Remove the empty frames, using hard mining instead
    train_df = train_df[~(train_df.label.astype(str)== "0")]

    # Preprocess train and validation data
    train_df = data_processing.preprocess_images(train_df,
                               root_dir=config.detection_model.train_image_dir,
                               save_dir=config.detection_model.crop_image_dir,
                               patch_size=config.predict.patch_size,
                               patch_overlap=config.predict.patch_overlap)
    
    non_empty = train_df[train_df.xmin!=0]

    train_df.loc[train_df.label==0,"label"] = "Object"
    validation.loc[validation.label==0,"label"] = "Object"

    if not validation.empty:
        validation_df = data_processing.preprocess_images(validation,
                                    root_dir=config.detection_model.train_image_dir,
                                    save_dir=config.detection_model.crop_image_dir,
                                    patch_size=config.predict.patch_size,
                                    patch_overlap=config.predict.patch_overlap,
                                    allow_empty=True
                                    )
        validation_df.loc[validation_df.label==0,"label"] = "Object"
        non_empty = validation_df[(validation_df.xmin!=0)]
        empty = validation_df[validation_df.xmin==0]

        # TO DO confirm empty frames here
        validation_df = non_empty
        
    # Train model is just a single class
    validation_df["label"] = "Object"
    train_df["label"] = "Object"

    # Load existing model
    if config.detection_model.checkpoint:
        loaded_model = load(config.detection_model.checkpoint, annotations=train_df)
    elif os.path.exists(config.detection_model.checkpoint_dir):
        loaded_model = get_latest_checkpoint(config.detection_model.checkpoint_dir, train_df)
    else:
        raise ValueError("No checkpoint or checkpoint directory found.")

    # Assert no FalsePositive label in train
    assert "FalsePositive" not in train_df.label.unique(), "FalsePositive label found in training data."

    trained_model = train(train_annotations=train_df,
                            test_annotations=validation_df,
                            train_image_dir=config.detection_model.crop_image_dir,
                            model=loaded_model,
                            comet_logger=comet_logger,
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

def _predict_list_(image_paths, patch_size, patch_overlap, model_path, m=None, crop_model=None, batch_size=16):
    if model_path:
        m = load(model_path)
    else:
        if m is None:
            raise ValueError("A model or model_path is required for prediction.")

    m.create_trainer(fast_dev_run=False)
    m.config["batch_size"] = batch_size
    predictions = []
    for image_path in image_paths:
        prediction = m.predict_tile(raster_path=image_path, return_plot=False, patch_size=patch_size, patch_overlap=patch_overlap, crop_model=crop_model)
        if prediction is None:
            prediction = pd.DataFrame({"image_path": image_path, "xmin": [None], "ymin": [None], "xmax": [None], "ymax": [None], "label": [None], "score": [None]})
        predictions.append(prediction)

    return predictions

def predict(image_paths, patch_size, patch_overlap, m=None, model_path=None, dask_client=None, crop_model=None, batch_size=16):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.          
        crop_model (main.deepforest): A trained deepforest model for classification.
        model_path (str): The path to a model checkpoint.
        dask_client (dask.distributed.Client): A dask client for parallel prediction.
        batch_size (int): The batch size for prediction.
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
                                              crop_model=crop_model,
                                              batch_size=batch_size)
            block_futures.append(block_future)
        # Get results
        predictions = []
        for block_result in block_futures:
            block_result = block_result.result()
            predictions.append(pd.concat(block_result))
    else:
        predictions = _predict_list_(image_paths=image_paths, patch_size=patch_size, patch_overlap=patch_overlap, model_path=model_path, m=m, crop_model=crop_model, batch_size=batch_size)

    return predictions
