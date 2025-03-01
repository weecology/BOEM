import os
import glob
import warnings
from PIL import Image

from deepforest.model import CropModel

# Local imports
from src.label_studio import gather_data

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

def train(model, train_dir, val_dir, comet_logger=None, fast_dev_run=False, max_epochs=10, batch_size=4):
    """Train a model on labeled images.
    Args:
        model (CropModel): A CropModel object.
        train_dir (str): The directory containing the training images.
        val_dir (str): The directory containing the validation images.
        fast_dev_run (bool): Whether to run a fast development run.
        max_epochs (int): The maximum number of epochs to train for.
        comet_logger (CometLogger): A CometLogger object.
        batch_size (int): The batch size for training.

    Returns:
        main.deepforest: A trained deepforest model.
    """
    model.batch_size = batch_size
    model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs)

    # Get the data stored from the write_crops processing.
    model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    
    # Log the validation dataset images
    for image_path, label in model.val_ds.imgs:
        label_name = model.numeric_to_label_dict[label]
        image_name = os.path.basename(image_path)
        comet_logger.experiment.log_image(image_path, name=f"{label_name}_{image_name}")

    with comet_logger.experiment.context_manager("classification"):
        model.trainer.fit(model)

    # Compute confusion matrix and upload to cometml
    image_dataset = []
    y_true = []
    y_predicted = []
    for index, (image,label) in enumerate(model.val_ds):
        image_path, label = model.val_ds.imgs[index]
        original_image = Image.open(image_path)
        image_dataset += [original_image]
        y_true += [label]
        y_predicted += [model(image.unsqueeze(0)).argmax().item()]
    labels = model.val_ds.classes

    # Log the confusion matrix to Comet
    comet_logger.experiment.log_confusion_matrix(
        y_true=y_true,
        y_predicted=y_predicted,
        images=image_dataset,
        labels=labels,
    )

    return model

def preprocess_images(model, annotations, root_dir, save_dir):
    # Remove any annotations with empty boxes
    annotations = annotations[(annotations['xmin'] != 0) & (annotations['ymin'] != 0) & (annotations['xmax'] != 0) & (annotations['ymax'] != 0)]
    # Remove any negative values
    annotations = annotations[(annotations['xmin'] >= 0) & (annotations['ymin'] >= 0) & (annotations['xmax'] >= 0) & (annotations['ymax'] >= 0)]
    boxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    images = annotations["image_path"].values
    labels = annotations["label"].values
    model.write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir)

def preprocess_and_train_classification(config, train_df=None, validation_df=None, comet_logger=None):
    """Preprocess data and train a crop model.
    
    Args:
        config: Configuration object containing training parameters
        train_df (pd.DataFrame): A DataFrame containing training annotations.
        validation_df (pd.DataFrame): A DataFrame containing validation annotations.
        comet_logger: CometLogger object for logging experiments
    Returns:
        trained_model: Trained model object
    """
    # Get and split annotations
    if train_df is not None:
        annotations = gather_data(config.classification_model.train_csv_folder)

    num_classes = len(annotations["label"].unique())

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

    # Force the label dict, DeepForest will update this soon
    loaded_model.label_dict = {v:k for k,v in enumerate(annotations["label"].unique())}
    loaded_model.numeric_to_label_dict = {v:k for k,v in loaded_model.label_dict.items()}

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
        batch_size=config.classification_model.trainer.batch_size,
        train_dir=config.classification_model.crop_image_dir,
        val_dir=config.classification_model.crop_image_dir,
        model=loaded_model,
        fast_dev_run=config.classification_model.trainer.fast_dev_run,
        max_epochs=config.classification_model.trainer.max_epochs,
        comet_logger=comet_logger
        )

    return trained_model