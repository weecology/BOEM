import os
import glob
import warnings
from PIL import Image

from deepforest.model import CropModel
import torch
from torch.nn import functional as F

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

def load(checkpoint=None, annotations=None, checkpoint_dir=None, num_classes=None):
    if checkpoint: 
        loaded_model = CropModel.load_from_checkpoint(checkpoint)
        num_classes = loaded_model.num_classes
    elif checkpoint_dir:
            loaded_model = get_latest_checkpoint(
                checkpoint_dir,
                num_classes=num_classes,
                annotations=annotations)
            num_classes = loaded_model.num_classes
    else:
        loaded_model = CropModel(num_classes=num_classes)
    
    if annotations is not None:
        if annotations.label.unique().tolist() != num_classes:
            finetune_model = CropModel(num_classes=len(annotations.label.unique()))

            # Strip the last layer off the checkout model and replace with new layer
            num_ftrs = loaded_model.model.fc.in_features
            loaded_model.model.fc = torch.nn.Linear(num_ftrs, len(annotations["label"].unique()))
            finetune_model.model = loaded_model.model
            loaded_model = finetune_model

    return loaded_model

def train(model, train_dir, val_dir, comet_logger=None, fast_dev_run=False, max_epochs=10, batch_size=4, workers=0, lr=0.0001):
    """Train a model on labeled images.
    Args:
        model (CropModel): A CropModel object.
        lr: Learning rate for training.
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
    model.num_workers = workers
    model.lr = lr

    devices = torch.cuda.device_count()
    model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs, num_nodes=1, devices = devices)

    # Get the data stored from the write_crops processing.
    model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    
    model.label_dict = model.train_ds.class_to_idx
    model.numeric_to_label = {v: k for k, v in model.train_ds.class_to_idx.items()}

    # Log the validation dataset images, max 10 per class
    label_count = {}
    numeric_to_label = {v: k for k, v in model.val_ds.class_to_idx.items()}
    for image_path, label in model.val_ds.imgs:
        label_name =numeric_to_label[label]
        if label_name not in label_count:
            label_count[label_name] = 0
        if label_count[label_name] < 10:
            image_name = os.path.basename(image_path)
            if comet_logger:
                comet_logger.experiment.log_image(image_path, name=f"{label_name}_{image_name}")
            label_count[label_name] += 1
    
    model.trainer.fit(model)
    
    dl = model.predict_dataloader(model.val_ds)
    
    # Iterate over dl and get batched predictions
    y_true = []
    y_predicted = []
    image_dataset = []

    for batch in dl:
        images, labels = batch
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_predicted.extend(preds.cpu().numpy())
        
    labels = model.val_ds.classes
    images = [model.val_ds.imgs[i][0] for i in range(len(model.val_ds.imgs))]
    image_dataset = [Image.open(image) for image in images]

    # Log the confusion matrix to Comet
    if comet_logger:
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
    
    # Expand by 20 pixels on all sides
    boxes = [[box[0]-20, box[1]-20, box[2]+20, box[3]+20] for box in boxes]
    
    # Make sure no negative values
    boxes = [[max(0, box[0]), max(0, box[1]), max(0, box[2]), max(0, box[3])] for box in boxes]
    
    images = annotations["image_path"].values
    labels = annotations["label"].values
    
    model.write_crops(boxes=boxes, root_dir=root_dir, images=images, labels=labels, savedir=save_dir)

def preprocess_and_train(
    train_df,
    validation_df,
    checkpoint,
    checkpoint_dir,
    train_image_dir,
    train_crop_image_dir,
    val_crop_image_dir,
    lr=0.0001,
    batch_size=4,
    fast_dev_run=False,
    max_epochs=10,
    workers=0,
    comet_logger=None
):
    """Preprocess data and train a crop model.
    
    Args:
        checkpoint (str): Path to the checkpoint file.
        checkpoint_dir (str): Directory containing checkpoint files.
        image_dir (str): Directory containing images to be cropped.
        train_crop_image_dir (str): Directory to save cropped training images.
        val_crop_image_dir (str): Directory to save cropped validation images.
        lr (float): Learning rate for training.
        batch_size (int): Batch size for training.
        fast_dev_run (bool): Whether to run a fast development run.
        max_epochs (int): Maximum number of epochs for training.
        workers (int): Number of workers for data loading.
        train_df (pd.DataFrame): DataFrame containing training annotations.
        validation_df (pd.DataFrame): DataFrame containing validation annotations.
        comet_logger: CometLogger object for logging experiments.
        
    Returns:
        trained_model: Trained model object.
    """
    # Get and split annotations
    num_classes = len(train_df["label"].unique())

    if train_df.empty:
        # Load existing model
        loaded_model = load(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
        )
        return loaded_model
        
    # Load existing model
    loaded_model = load(
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        annotations=train_df,
        num_classes=num_classes,
    )

    if train_df.empty:
        print("No annotations found.")
        return loaded_model
    
    # Preprocess train and validation data
    preprocess_images(
        model=loaded_model, 
        annotations=train_df, 
        root_dir=train_image_dir, 
        save_dir=train_crop_image_dir
    )    
    
    preprocess_images(
        model=loaded_model, 
        annotations=validation_df, 
        root_dir=train_image_dir, 
        save_dir=val_crop_image_dir
    )

    trained_model = train(
        batch_size=batch_size,
        train_dir=train_crop_image_dir,
        val_dir=val_crop_image_dir,
        model=loaded_model,
        fast_dev_run=fast_dev_run,
        max_epochs=max_epochs,
        comet_logger=comet_logger,
        workers=workers,
    )

    return trained_model