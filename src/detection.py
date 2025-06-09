# Standard library imports
import glob
import os
import random
import tempfile
import warnings
from logging import warn
import math

# Third party imports
import pandas as pd
from deepforest import main, visualize
from deepforest.utilities import read_file
import torch

# Local imports
from src import data_processing
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
    devices = torch.cuda.device_count()
    strategy = "ddp" if devices > 1 else "auto"
    model.create_trainer(num_nodes=1, devices=devices, strategy=strategy)
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

    train_annotations.to_csv(os.path.join(tmpdir,"train.csv"), index=False)

    # Set config
    model.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
    model.config["train"]["root_dir"] = train_image_dir

    if test_annotations is not None:
        test_annotations.to_csv(os.path.join(tmpdir,"test.csv"), index=False)
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

    devices = torch.cuda.device_count()
    strategy = "ddp" if devices > 1 else "auto"
    if comet_logger:
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", train_annotations)
        comet_logger.experiment.log_table("test.csv", test_annotations)
        
    model.create_trainer(logger=comet_logger, num_nodes=1, accelerator="gpu", strategy=strategy, devices=devices)

    non_empty_train_annotations = read_file(model.config["train"]["csv_file"], root_dir=train_image_dir)
    
    # Skip for fast_dev run
    if not model.trainer.fast_dev_run:
        if comet_logger:
            n = 5 if non_empty_train_annotations.shape[0] > 5 else non_empty_train_annotations.shape[0]
            for filename in non_empty_train_annotations.image_path.sample(n=n).unique():
                sample_train_annotations_for_image = non_empty_train_annotations[non_empty_train_annotations.image_path == filename]
                sample_train_annotations_for_image.root_dir = train_image_dir
                visualize.plot_annotations(sample_train_annotations_for_image, savedir=tmpdir)
                comet_logger.experiment.log_image(os.path.join(tmpdir, filename),metadata={"name":filename,"context":'detection_train'})

            if test_annotations is not None:
                non_empty_validation_annotations = read_file(model.config["validation"]["csv_file"], root_dir=train_image_dir)
                n = 20 if non_empty_validation_annotations.shape[0] > 20 else non_empty_validation_annotations.shape[0]
                for filename in non_empty_validation_annotations.image_path.sample(n=n).unique():
                    sample_validation_annotations_for_image = non_empty_validation_annotations[non_empty_validation_annotations.image_path == filename]
                    sample_validation_annotations_for_image.root_dir = train_image_dir
                    visualize.plot_annotations(sample_validation_annotations_for_image, savedir=tmpdir)
                    comet_logger.experiment.log_image(os.path.join(tmpdir, filename),metadata={"name":filename,"context":'detection_validation'})

    model.trainer.fit(model)

    if not model.trainer.fast_dev_run:
        if test_annotations is not None:
            for image_path in test_annotations.head().image_path.unique():
                prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
                if prediction is None:
                    continue
                visualize.plot_results(prediction, savedir=tmpdir)
                comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))

    return model

def preprocess_and_train(train_annotations, validation_annotations, train_image_dir, crop_image_dir, patch_size, patch_overlap, limit_empty_frac, checkpoint, checkpoint_dir, trainer_config, comet_logger=None):
    """Preprocess data and train model.
    
    Args:
        train_annotations: DataFrame containing training annotations
        validation_annotations: DataFrame containing validation annotations
        train_image_dir: Directory containing training images
        crop_image_dir: Directory to save cropped images
        patch_size: Size of the patches for preprocessing
        patch_overlap: Overlap between patches for preprocessing
        limit_empty_frac: Fraction to limit empty frames
        checkpoint: Path to model checkpoint
        checkpoint_dir: Directory containing model checkpoints
        trainer_config: Configuration dictionary for training parameters
        comet_logger: CometLogger instance for logging.
    Returns:
        trained_model: Trained model object
    """

    if train_annotations is not None:
        # Remove the empty frames, using hard mining instead
        
        train_annotations = train_annotations[~(train_annotations.label.astype(str)== "0")]
        if train_annotations.empty:
            train_annotations = None
        else:
            # Preprocess train and validation data
            train_df = data_processing.preprocess_images(train_annotations,
                                    root_dir=train_image_dir,
                                    save_dir=crop_image_dir,
                                    patch_size=patch_size,
                                    patch_overlap=patch_overlap)
            
            train_df.loc[train_df.label==0,"label"] = "Object"
            train_df["label"] = "Object"
            
            # Assert no FalsePositive label in train
            assert "FalsePositive" not in train_df.label.unique(), "FalsePositive label found in training data."

    if validation_annotations is not None:
        validation_annotations.loc[validation_annotations.label==0,"label"] = "Object"

        if limit_empty_frac > 0:
            validation_annotations = limit_empty_frames(validation_annotations, limit_empty_frac)
        
        if not validation_annotations.empty:
            validation_df = data_processing.preprocess_images(validation_annotations,
                                        root_dir=train_image_dir,
                                        save_dir=crop_image_dir,
                                        patch_size=patch_size,
                                        patch_overlap=patch_overlap,
                                        allow_empty=True
                                        )
            validation_df.loc[validation_df.label==0,"label"] = "Object"
                
            # Train model is just a single class
            validation_df["label"] = "Object"
        else:
            validation_df = None
    else:
        validation_df = None        

    # Load existing model
    if checkpoint:
        loaded_model = load(checkpoint)
    elif os.path.exists(checkpoint_dir):
        loaded_model = get_latest_checkpoint(checkpoint_dir)
        if loaded_model is None:
            label_dict = {value: index for index, value in enumerate(train_df.label.unique())}
            loaded_model = main.deepforest(label_dict=label_dict)
    else:
        label_dict = {value: index for index, value in enumerate(train_df.label.unique())}
        loaded_model = main.deepforest(label_dict=label_dict)

    if train_annotations is not None:
        if validation_df is None or validation_df.empty:
            warn("Validation data is empty. Training model with training data only.")
            validation_df = None
        # Train model
        trained_model = train(train_annotations=train_df,
                                test_annotations=validation_df,
                                train_image_dir=crop_image_dir,
                                model=loaded_model,
                                comet_logger=comet_logger,
                                config_args=trainer_config)

        detection_checkpoint_path = save_model(trained_model, checkpoint_dir, basename=comet_logger.experiment.id)
        comet_logger.experiment.log_asset(file_data=detection_checkpoint_path, file_name="detection_model.ckpt")
        comet_logger.experiment.log_parameter("detection_checkpoint_path", detection_checkpoint_path)
    else:
        trained_model = loaded_model
        comet_logger.experiment.log_parameter("detection_checkpoint_path", checkpoint)

    return trained_model

def get_latest_checkpoint(checkpoint_dir):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = main.load_from_checkpoint(checkpoint)
            return m
        else:
            return None
    else:
        return None



def predict(m, image_paths, patch_size, patch_overlap, crop_model=None, batch_size=6, workers=5):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.          
        crop_model (main.deepforest): A trained deepforest model for classification.
        model_path (str): The path to a model checkpoint.
        batch_size (int): The batch size for prediction.
    Returns:
        list: A list of image predictions.
    """

    m.config["batch_size"] = batch_size
    m.config["workers"] = workers
    predictions = m.predict_tile(path=image_paths,
                   patch_size=patch_size,
                   patch_overlap=patch_overlap,
                   dataloader_strategy="batch",
                   crop_model=crop_model)

    return predictions

def save_model(model, directory, basename):
    checkpoint_path = os.path.join(directory, f"{basename}.ckpt")
    if not os.path.exists(checkpoint_path):
        model.trainer.save_checkpoint(checkpoint_path)

    return checkpoint_path

