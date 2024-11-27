# Standard library imports
import os
import glob
import warnings
# Third party imports
import pandas as pd
from deepforest.model import CropModel

# Local imports
from src.label_studio import gather_data
from pytorch_lightning.loggers import CometLogger


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
            m = CropModel(num_classes=len(annotations["label"].unique()))
    else:
        os.makedirs(checkpoint_dir)
        m = CropModel(num_classes=len(annotations["label"].unique()))

    return m

def train(model, train_dir, val_dir, comet_project=None, comet_workspace=None, fast_dev_run=False):
    """Train a model on labeled images.
    Args:
        model (CropModel): A CropModel object.
        train_dir (str): The directory containing the training images.
        val_dir (str): The directory containing the validation images.
        comet_project (str): The comet project name for logging. Defaults to None.
        comet_workspace (str): The comet workspace for logging. Defaults to None.

    Returns:
        main.deepforest: A trained deepforest model.
    """
    # Update
    if comet_project:
        comet_logger = CometLogger(project_name=comet_project, workspace=comet_workspace)
        model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run)
    else:
        model.create_trainer(fast_dev_run=fast_dev_run)

    # Get the data stored from the write_crops step above.
    model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    model.trainer.fit(model)

    return model

def preprocess_images(model, annotations, root_dir, save_dir):
    # Remove any annotations with empty boxes
    annotations = annotations[annotations['xmin'] != 0]
    boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    images = annotations["image_path"].values
    labels = annotations["label"].values
    model.write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir)

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
        loaded_model = CropModel(config.classification_model.checkpoint, num_classes=len(train_df["label"].unique()))

    elif os.path.exists(config.classification_model.checkpoint_dir):
        loaded_model = get_latest_checkpoint(
            config.classification_model.checkpoint_dir, train_df)
    else:
        raise ValueError("No checkpoint or checkpoint directory found.")


    # Preprocess train and validation data
    preprocess_images(
        model=loaded_model, 
        annotations=train_df, 
        root_dir=config.classification_model.train_image_dir, 
        save_dir=config.classification_model.crop_image_dir)    
    preprocess_images(
        model=loaded_model, 
        annotations=validation_df, 
        root_dir=config.classification_model.train_image_dir, 
        save_dir=config.classification_model.crop_image_dir)

    trained_model = train(
        train_dir=config.classification_model.crop_image_dir,
        val_dir=config.classification_model.crop_image_dir,
        model=loaded_model,
        comet_project=config.comet.project,
        comet_workspace=config.comet.workspace,
        fast_dev_run=config.classification_model.fast_dev_run)

    return trained_model
