import os
import glob
from PIL import Image

from deepforest.model import CropModel
import torch
from torch.nn import functional as F

def get_latest_checkpoint(checkpoint_dir, num_classes):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = CropModel.load_from_checkpoint(checkpoint, num_classes=num_classes)
            return m
        else:
            return None
    else:
        return None

def load(checkpoint=None, annotations=None, checkpoint_dir=None, num_classes=None, checkpoint_num_classes=None, checkpoint_train_dir=None):
    if checkpoint: 
        loaded_model = CropModel.load_from_checkpoint(
            checkpoint,
            num_classes=checkpoint_num_classes)
        num_classes = loaded_model.num_classes
        # This is a workaround for a bug in Deepforest, where the class_to_idx is not saved in the checkpoint
        loaded_model.load_from_disk(train_dir=checkpoint_train_dir, val_dir=checkpoint_train_dir)
        loaded_model.label_dict = loaded_model.train_ds.class_to_idx
        loaded_model.numeric_to_label_dict = {v: k for k, v in loaded_model.train_ds.class_to_idx.items()}

    elif checkpoint_dir:
            loaded_model = get_latest_checkpoint(
                num_classes = checkpoint_num_classes,
                checkpoint_dir=checkpoint_dir)

            if loaded_model is None:
                loaded_model = CropModel(num_classes=num_classes)
            num_classes = loaded_model.num_classes
    else:
        loaded_model = CropModel(num_classes=num_classes)
    
    loaded_model.create_trainer()

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
    model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs, num_nodes=1, devices = devices, enable_checkpointing=False)

    # Check if the model has been trained on a different set of classes
    if model.label_dict is not None:
        classes_in_checkpoint = list(model.label_dict.keys())
        model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
        classes_in_new_data = list(model.train_ds.class_to_idx.keys())

        # If there are new classes in the training data, update the model
        if set(classes_in_new_data).issubset(set(classes_in_checkpoint)):
            finetune_model = model
        else:
            combined_classes = set(classes_in_new_data).union(set(classes_in_checkpoint))
            finetune_model = CropModel(num_classes=len(combined_classes))

            # Strip the last layer off the checkpoint model and replace with new layer
            num_ftrs = model.model.fc.in_features
            model.model.fc = torch.nn.Linear(num_ftrs, len(combined_classes))
            finetune_model.model = model.model
            finetune_model.label_dict = model.train_ds.class_to_idx
            finetune_model.numeric_to_label_dict = {v: k for k, v in model.train_ds.class_to_idx.items()}
            finetune_model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs, num_nodes=1, devices = devices, enable_checkpointing=False)
            finetune_model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    else:
        finetune_model = model
        finetune_model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
    
    # If rank 0, set the if model.trainer.global_rank == 0:
    finetune_model.trainer.fit(finetune_model)
    
    dl = finetune_model.predict_dataloader(finetune_model.val_ds)
    
    # Iterate over dl and get batched predictions
    y_true = []
    y_predicted = []
    image_dataset = []

    for batch in dl:
        images, labels = batch
        outputs = finetune_model(images)
        _, preds = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_predicted.extend(preds.cpu().numpy())
        
    labels = finetune_model.val_ds.classes
    images = [finetune_model.val_ds.imgs[i][0] for i in range(len(finetune_model.val_ds.imgs))]
    image_dataset = [Image.open(image) for image in images]

    # Log the confusion matrix to Comet
    if comet_logger:
        comet_logger.experiment.log_confusion_matrix(
            y_true=y_true,
            y_predicted=y_predicted,
            images=image_dataset,
            labels=labels,
        )

    return finetune_model

def preprocess_images(model, annotations, root_dir, save_dir):
    annotations = annotations[~annotations.label.isin([0,"0","FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
    if annotations.empty:
        return None
    
    # Two word labels
    annotations["label"] = annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
    annotations = annotations[annotations["label"].apply(lambda x: len(x.split()) == 2)]

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
    image_dir,
    train_crop_image_dir,
    val_crop_image_dir, 
    lr=0.0001,
    batch_size=4,
    fast_dev_run=False,
    max_epochs=10,
    workers=0,
    comet_logger=None,
    checkpoint_num_classes=None,
    checkpoint_train_dir=None
):
    """Preprocess data and train a crop model.
    
    Args:
        checkpoint (str): Path to the checkpoint file.
        checkpoint_num_classes (int): Number of classes in the checkpoint model. To be removed in DeepForest 1.6
        checkpoint_dir (str): Directory containing checkpoint files.
        image_dir (str): Directory containing images to be cropped.
        train_crop_image_dir (str): Directory to save cropped training images.
        val_crop_image_dir (str): Directory to save cropped validation images.
        lr (float): Learning rate for training.
        batch_size (int): Batch size for training.
        fast_dev_run (bool): Whether to run a fast development run.
        max_epochs (int): Maximum number of epochs for training.
        workers (int): Number of workers for data loading.
        train_df (pd.DataFrame): DataFrame containing training annotations. If none, load from existing dir
        validation_df (pd.DataFrame): DataFrame containing validation annotations. If none, load from existing dir
        checkpoint_train_dir: Directory containing training images for the checkpoint model.
        comet_logger: CometLogger object for logging experiments.
        
    Returns:
        trained_model: Trained model object.
    """

    if checkpoint:
        if checkpoint_num_classes is None:
            raise ValueError("checkpoint_num_classes must be provided if checkpoint is passed.")
    
    if train_df is None:
        num_classes = checkpoint_num_classes
        # Load existing model
        loaded_model = load(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            checkpoint_train_dir=checkpoint_train_dir,
            checkpoint_num_classes=checkpoint_num_classes,
        )
    else:
        num_classes = len(train_df["label"].unique())

        # Load existing model
        loaded_model = load(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            annotations=train_df,
            num_classes=num_classes,
            checkpoint_num_classes=checkpoint_num_classes,
            checkpoint_train_dir=checkpoint_train_dir,
        )

    # Preprocess train and validation data
    if train_df is not None:
        preprocess_images(
            model=loaded_model, 
            annotations=train_df, 
            root_dir=image_dir, 
            save_dir=train_crop_image_dir
        )    
        non_empty_train = train_df[train_df.xmin != 0]
        if non_empty_train.empty:
            train_df = None

    if validation_df is not None:
         # Preprocess validation data
        preprocess_images(
            model=loaded_model, 
            annotations=validation_df, 
            root_dir=image_dir, 
            save_dir=val_crop_image_dir
        )
        non_empty_val = validation_df[validation_df.xmin != 0]
        if non_empty_val.empty:
            validation_df = None

    if train_df is not None and validation_df is not None:
        # Check for non-empty train and validation data
            trained_model = train(
                batch_size=batch_size,
                lr=lr,
                train_dir=train_crop_image_dir,
                val_dir=val_crop_image_dir,
                model=loaded_model,
                fast_dev_run=fast_dev_run,
                max_epochs=max_epochs,
                comet_logger=comet_logger,
                workers=workers,
            )
            classification_checkpoint_path = save_model(trained_model, checkpoint_dir, comet_logger.experiment.id)
            comet_logger.experiment.log_asset(file_data=classification_checkpoint_path, file_name="classification_model.ckpt")
    else:
        trained_model = loaded_model

    return trained_model

def save_model(model, directory, basename):
    checkpoint_path = os.path.join(directory, f"{basename}.ckpt")
    if not os.path.exists(checkpoint_path):
        model.trainer.save_checkpoint(checkpoint_path)

    return checkpoint_path
