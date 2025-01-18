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

def get_latest_checkpoint(checkpoint_dir, annotations, lr=0.0001, num_classes=None):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            try:
                m = CropModel.load_from_checkpoint(checkpoint)
            except Exception as e:
                warnings.warn("Could not load model from checkpoint, {}".format(e))
                if num_classes:
                    m = CropModel(num_classes=num_classes, lr=lr)
                else:
                    m = CropModel(num_classes=len(annotations["label"].unique()), lr=lr)
        else:
            warnings.warn("No checkpoints found in {}".format(checkpoint_dir))
            if num_classes:
                m = CropModel(num_classes=num_classes, lr=lr)
            else:
                m = CropModel(num_classes=len(annotations["label"].unique()), lr=lr)
    else:
        os.makedirs(checkpoint_dir)
        if num_classes:
            m = CropModel(num_classes=num_classes, lr=lr)
        else:
            m = CropModel(num_classes=len(annotations["label"].unique()), lr=lr)

    return m

def load(checkpoint=None, annotations=None, checkpoint_dir=None, lr=0.0001, num_classes=None):
    if checkpoint: 
        if num_classes:
            loaded_model = CropModel(checkpoint, num_classes=num_classes, lr=lr)
        else:
            loaded_model = CropModel(checkpoint, num_classes=len(annotations["label"].unique()), lr=lr)
    elif checkpoint_dir:
        loaded_model = get_latest_checkpoint(
            checkpoint_dir,
            num_classes=num_classes,
            annotations=annotations)
    else:
        raise ValueError("No checkpoint or checkpoint directory found.")
    
    return loaded_model

def train(model, train_dir, val_dir, comet_workspace=None, comet_project=None, fast_dev_run=False, max_epochs=10):
    """Train a model on labeled images.
    Args:
        model (CropModel): A CropModel object.
        train_dir (str): The directory containing the training images.
        val_dir (str): The directory containing the validation images.
        fast_dev_run (bool): Whether to run a fast development run.
        max_epochs (int): The maximum number of epochs to train for.

    Returns:
        main.deepforest: A trained deepforest model.
    """
    
    if comet_project:
        comet_logger = CometLogger(project_name=comet_project, workspace=comet_workspace)
        comet_logger.experiment.add_tags(["classification"])
    else:
        comet_logger = None

    model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs)

    # Get the data stored from the write_crops step above.
    model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    model.trainer.fit(model)

    model.trainer.logger.experiment.end()
    comet_logger.experiment.end()

    return model

def preprocess_images(model, annotations, root_dir, save_dir):
    # Remove any annotations with empty boxes
    annotations = annotations[annotations['xmin'] != 0]
    boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    images = annotations["image_path"].values
    labels = annotations["label"].values
    model.write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir)

def preprocess_and_train_classification(config, validation_df=None, num_classes=None):
    """Preprocess data and train a crop model.
    
    Args:
        config: Configuration object containing training parameters
        validation_df (pd.DataFrame): A DataFrame containing validation annotations.
    Returns:
        trained_model: Trained model object
    """
    # Get and split annotations
    annotations = gather_data(config.classification_model.train_csv_folder)

    # Remove the empty frames
    annotations = annotations[~(annotations.label.astype(str)== "0")]
    annotations = annotations[annotations.label != "FalsePositive"]

    if validation_df is None:
        train_df, validation_df = create_train_test(annotations)
    else:
        train_df = annotations[~annotations["image_path"].
                               isin(validation_df["image_path"])]

    # Load existing model
    loaded_model = load(
        checkpoint=config.classification_model.checkpoint,
        checkpoint_dir=config.classification_model.checkpoint_dir,
        annotations=annotations,
        lr=config.classification_model.trainer.lr,
        num_classes=num_classes
        )

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
        comet_workspace=config.comet.workspace,
        comet_project=config.comet.project,
        fast_dev_run=config.classification_model.trainer.fast_dev_run,
        max_epochs=config.classification_model.trainer.max_epochs,
        )

    return trained_model
