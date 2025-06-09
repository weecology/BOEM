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

def load(checkpoint=None, num_classes=None):
    if checkpoint: 
        print(f"[load] Loading model from checkpoint: {checkpoint}")
        loaded_model = CropModel.load_from_checkpoint(
            checkpoint_path=checkpoint)
        num_classes = loaded_model.num_classes
        print(f"[load] Loaded model num_classes: {num_classes}")
        print(f"[load] Model output features: {loaded_model.model.fc.out_features}")
        print(f"[load] label_dict length: {len(loaded_model.label_dict)}")
        # Assert that the crop_model.model.fc.out_features == num_classes
        assert loaded_model.model.fc.out_features == num_classes, f"Model output features {loaded_model.model.fc.out_features} do not match num_classes {num_classes}"
        assert loaded_model.model.fc.out_features == len(loaded_model.label_dict), f"Model output features {loaded_model.model.fc.out_features} do not match label_dict {len(loaded_model.label_dict)}"
    else:
        print(f"[load] Creating new CropModel with num_classes={num_classes}")
        loaded_model = CropModel(num_classes=num_classes)
    
    loaded_model.create_trainer()

    return loaded_model

def train(model, train_dir, val_dir, comet_logger=None, fast_dev_run=False, max_epochs=10, batch_size=4, workers=0, lr=0.0001):
    """Train a model on labeled images."""
    model.batch_size = batch_size
    model.num_workers = workers
    model.lr = lr

    devices = torch.cuda.device_count()
    print(f"[train] Using {devices} CUDA devices.")

    model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs, num_nodes=1, devices = devices, enable_checkpointing=False)

    # Check if the model has been trained on a different set of classes
    if hasattr(model, 'label_dict') and model.label_dict:
        classes_in_checkpoint = list(model.label_dict.keys())
        print(f"[train] Classes in checkpoint: {len(classes_in_checkpoint)}")
        model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
        classes_in_new_data = list(model.train_ds.class_to_idx.keys())
        print(f"[train] Classes in new data: {len(classes_in_new_data)}")

        # If there are new classes in the training data, update the model
        if set(classes_in_new_data).issubset(set(classes_in_checkpoint)):
            print("[train] No new classes found, using existing model.")
            finetune_model = model
        else:
            print("[train] New classes found! Updating model output layer.")
            combined_classes = set(classes_in_new_data).union(set(classes_in_checkpoint))
            print(f"[train] Combined classes: {len(combined_classes)}")
            finetune_model = CropModel(num_classes=len(classes_in_new_data))
            finetune_model.load_from_disk(train_dir=train_dir, val_dir=val_dir)

            # Strip the last layer off the checkpoint model and replace with new layer
            num_ftrs = model.model.fc.in_features
            print(f"[train] Setting model output layer to {len(classes_in_new_data)} classes.")
            model.model.fc = torch.nn.Linear(num_ftrs, len(classes_in_new_data))
            finetune_model.model = model.model
            finetune_model.label_dict = model.train_ds.class_to_idx
            finetune_model.numeric_to_label_dict = {v: k for k, v in model.train_ds.class_to_idx.items()}
            print(f"[train] label_dict set to {len(finetune_model.label_dict)} classes.")
            finetune_model.create_trainer(logger=comet_logger, fast_dev_run=fast_dev_run, max_epochs=max_epochs, num_nodes=1, devices = devices, enable_checkpointing=False)
            if len(finetune_model.label_dict) != len(classes_in_new_data):
                print(f"[train][WARNING] label_dict ({len(finetune_model.label_dict)}) does not match classes_in_new_data ({len(classes_in_new_data)})!")
    else:
        print("[train] No label_dict found, loading from disk.")
        finetune_model = model
        finetune_model.load_from_disk(train_dir=train_dir, val_dir=val_dir)
        print(f"[train] label_dict set to {len(finetune_model.label_dict)} classes.")

    print(f"[train] Fitting model with {finetune_model.model.fc.out_features} output features and {len(finetune_model.label_dict)} labels.")
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
    checkpoint_num_classes=None,
):
    """Preprocess data and train a crop model."""
    if checkpoint:
        if checkpoint_num_classes is None:
            raise ValueError("checkpoint_num_classes must be provided if checkpoint is passed.")
        print(f"[preprocess_and_train] Using checkpoint: {checkpoint} with checkpoint_num_classes: {checkpoint_num_classes}")
    
    if train_df is None:
        num_classes = checkpoint_num_classes
        print(f"[preprocess_and_train] No train_df provided, using num_classes from checkpoint: {num_classes}")
        loaded_model = load(
            checkpoint=checkpoint,
            num_classes=num_classes
        )
    else:
        num_classes = len(train_df["label"].unique())
        print(f"[preprocess_and_train] Found {num_classes} unique labels in train_df.")

        loaded_model = load(
            checkpoint=checkpoint,
            num_classes=num_classes,
        )

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

    if train_df is None:
        print("[preprocess_and_train] Training with no train_df, using loaded model.")
        trained_model = train(
            batch_size=batch_size,
            lr=lr,
            train_dir=train_crop_image_dir,
            val_dir=val_crop_image_dir,
            model=loaded_model,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            comet_logger=comet_logger,
            workers=workers)
        if trained_model.trainer.global_rank == 0:
            print(f"[preprocess_and_train] saving model to checkpoint {checkpoint_dir}")
            classification_checkpoint_path = save_model(trained_model, checkpoint_dir, comet_logger.experiment.id)
            comet_logger.experiment.log_asset(file_data=classification_checkpoint_path, file_name="classification_model.ckpt")
    elif preprocessed_train is not None and preprocessed_validation is not None:
        print("[preprocess_and_train] Training with preprocessed train and validation data.")
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
        if trained_model.trainer.global_rank == 0:
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
