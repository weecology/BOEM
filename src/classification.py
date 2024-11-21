# Standard library imports
import os
import glob
import warnings
# Third party imports
import pandas as pd
from deepforest.model import CropModel

# Local imports
from src.label_studio import gather_data

import tempfile
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CometLogger
from deepforest import visualize


def create_train_test(annotations):
    return annotations.sample(frac=0.8, random_state=1), annotations.drop(
        annotations.sample(frac=0.8, random_state=1).index)

def get_latest_checkpoint(checkpoint_dir, annotations):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = CropModel.load_from_checkpoint(checkpoint)
        else:
            warnings.warn("No checkpoints found in {}".format(checkpoint_dir))
            label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
            m = CropModel()
    else:
        os.makedirs(checkpoint_dir)
        m = CropModel()

    return m

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
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", train_annotations)
        comet_logger.experiment.log_table("test.csv", test_annotations)

        model.create_trainer(logger=comet_logger)
    else:
        model.create_trainer()

    with comet_logger.experiment.context_manager("train_images"):
        non_empty_train_annotations = train_annotations[~(train_annotations.xmax==0)]
        if non_empty_train_annotations.empty:
            pass
        else:
            sample_train_annotations = non_empty_train_annotations[non_empty_train_annotations.image_path.isin(non_empty_train_annotations.image_path.head(5))]
            for filename in sample_train_annotations.image_path:
                sample_train_annotations_for_image = sample_train_annotations[sample_train_annotations.image_path == filename]
                sample_train_annotations_for_image.root_dir = train_image_dir
                visualize.plot_results(sample_train_annotations_for_image, savedir=tmpdir)
                comet_logger.experiment.log_image(os.path.join(tmpdir, filename))

    model.trainer.fit(model)

    with comet_logger.experiment.context_manager("post-training prediction"):
        for image_path in test_annotations.image_path.head(5):
            prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
            if prediction is None:
                continue
            visualize.plot_results(prediction, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))

    return model

def preprocess_images(model, annotations, root_dir, save_dir):
    for image_path in annotations["image_path"]:
        boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        image_path = os.path.join(root_dir, image_path)
        model.write_crops(boxes=boxes, labels=annotations.label.values, image_path=image_path, savedir=save_dir)

def preprocess_and_train_classification(config, validation_df=None):
    """Preprocess data and train a crop model.
    
    Args:
        config: Configuration object containing training parameters
        validation_df (pd.DataFrame): A DataFrame containing validation annotations.
    Returns:
        trained_model: Trained model object
    """
    # Get and split annotations
    annotations = gather_data(config.classification_model.train_csv_folder)

    if validation_df is None:
        train_df, validation_df = create_train_test(annotations)
    else:
        train_df = annotations[~annotations["image_path"].
                               isin(validation_df["image_path"])]

    # Train model

    # Load existing model
    if config.classification_model.checkpoint:
        loaded_model = CropModel(config.classification_model.checkpoint)

    elif os.path.exists(config.classification_model.checkpoint_dir):
        loaded_model = get_latest_checkpoint(
            config.classification_model.checkpoint_dir, train_df)
    else:
        raise ValueError("No checkpoint or checkpoint directory found.")


    # Preprocess train and validation data
    preprocess_images(model=loaded_model, annotations=train_df, root_dir=config.classification_model.train_image_dir, save_dir=config.classification_model.crop_image_dir)
    preprocess_images(model=loaded_model, annotations=validation_df, root_dir=config.classification_model.train_image_dir, save_dir=config.classification_model.crop_image_dir)

    trained_model = train(
        train_annotations=train_df,
        test_annotations=validation_df,
        train_image_dir=config.classification_model.crop_image_dir,
        model=loaded_model,
        comet_project=config.comet.project,
        comet_workspace=config.comet.workspace,
        config_args=config.deepforest)

    return trained_model
