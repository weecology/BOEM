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

def train(model, comet_logger=None, fast_dev_run=False, max_epochs=10, batch_size=4, workers=0, lr=0.0001):
    """Train a model on labeled images."""
    model.batch_size = batch_size
    model.num_workers = workers
    model.lr = lr

    devices = torch.cuda.device_count()
    accelerator = "gpu" if devices > 0 else "cpu"
    if devices == 0:
        devices = 1
    print(f"[train] Using accelerator={accelerator}, devices={devices}.")

    model.create_trainer(
        logger=comet_logger,
        fast_dev_run=fast_dev_run,
        max_epochs=max_epochs,
        num_nodes=1,
        devices=devices,
        accelerator=accelerator,
        enable_checkpointing=False,
    )
    model.trainer.fit(model)

    return model

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

    return annotations

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
):
    """Preprocess data and train a crop model."""
    if train_df is None:
        loaded_model = CropModel.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        # infer classes from training labels
        label_dict = {value: index for index, value in enumerate(sorted(train_df.label.unique()))}
        loaded_model = CropModel(num_classes=len(label_dict))
        loaded_model.label_dict = label_dict

    loaded_model.create_trainer()

    # Preprocess train and validation data
    if train_df is not None:
        preprocessed_train = preprocess_images(
            model=loaded_model, 
            annotations=train_df, 
            root_dir=image_dir, 
            save_dir=train_crop_image_dir
        )    
        non_empty_train = train_df[train_df.xmin != 0]
        if non_empty_train.empty:
            print("[preprocess_and_train] All train_df boxes are empty!")
            train_df = None

    if validation_df is not None:
        preprocessed_validation = preprocess_images(
            model=loaded_model, 
            annotations=validation_df, 
            root_dir=image_dir, 
            save_dir=val_crop_image_dir
        )
        if preprocessed_validation is not None and not preprocessed_validation.empty:
            preprocessed_validation = preprocessed_validation
        else:
            print("[preprocess_and_train] Validation data is empty after preprocessing.")
            preprocessed_validation = None
    
    loaded_model.load_from_disk(train_dir=str(train_crop_image_dir), val_dir=str(val_crop_image_dir))

    if preprocessed_train is not None and preprocessed_validation is not None:
        print("[preprocess_and_train] Training with preprocessed train and validation data.")
        trained_model = train(
            batch_size=batch_size,
            lr=lr,
            model=loaded_model,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            comet_logger=comet_logger,
            workers=workers,
        )
        if trained_model.trainer.global_rank == 0 and comet_logger is not None:
            print(f"[preprocess_and_train] saving model to checkpoint {checkpoint_dir}")
            classification_checkpoint_path = save_model(trained_model, checkpoint_dir, comet_logger.experiment.id)
            comet_logger.experiment.log_asset(file_data=classification_checkpoint_path, file_name="classification_model.ckpt")
    else:
        print("[preprocess_and_train] No training performed, returning loaded model.")
        trained_model = loaded_model

    return trained_model

def save_model(model, directory, basename):
    checkpoint_path = os.path.join(directory, f"{basename}.ckpt")
    if not os.path.exists(checkpoint_path):
        model.trainer.save_checkpoint(checkpoint_path)

    return checkpoint_path
