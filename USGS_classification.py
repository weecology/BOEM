import pandas as pd
import glob
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train
import hydra
from omegaconf import DictConfig
import os

# Create train test split, split each class into 90% train and 10% test with a minimum of 5 images per class for test and a max of 100
def train_test_split(df, test_size=0.1, min_test_images=5, max_test_images=100):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        test_count = max(min_test_images, int(len(class_df) * test_size))
        test_count = min(test_count, max_test_images)
        
        test_class_df = class_df.sample(n=test_count)
        train_class_df = class_df.drop(test_class_df.index)
        
        train_df = pd.concat([train_df, train_class_df])
        test_df = pd.concat([test_df, test_class_df])
    
    return train_df, test_df

@hydra.main(config_path="boem_conf", config_name="boem_config")
def main(cfg: DictConfig):
    # Override the classification_model config with USGS.yaml
    cfg = hydra.compose(config_name="boem_config", overrides=["classification_model=USGS"])
        
    # From the detection script
    crop_annotations = glob.glob("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/*.csv")
    crop_annotations = [pd.read_csv(x) for x in crop_annotations]
    crop_annotations = pd.concat(crop_annotations)
    
    # Keep labels with more than 25 images
    crop_annotations = crop_annotations.groupby("label").filter(lambda x: len(x) > 25)

    # Only keep two word labels
    crop_annotations = crop_annotations[crop_annotations["label"].str.contains(" ")]
    crop_annotations = crop_annotations[~crop_annotations.label.isin([0,"0","FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
    
    def normalize_label(l):
        if pd.isna(l):
            return l
        s = str(l).strip()
        s = s.replace("/", " ")
        s = " ".join(s.split()[:2])   # keep first two tokens if that is your convention
        return s
    
    crop_annotations["label"] = crop_annotations["label"].apply(normalize_label)

    # Two word labels
    crop_annotations["label"] = crop_annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
    crop_annotations = crop_annotations[crop_annotations["label"].apply(lambda x: len(x.split()) == 2)]

    # Remove any crop_annotations with empty boxes
    crop_annotations = crop_annotations[(crop_annotations['xmin'] != 0) & (crop_annotations['ymin'] != 0) & (crop_annotations['xmax'] != 0) & (crop_annotations['ymax'] != 0)]

    # Remove any negative values
    crop_annotations = crop_annotations[(crop_annotations['xmin'] >= 0) & (crop_annotations['ymin'] >= 0) & (crop_annotations['xmax'] >= 0) & (crop_annotations['ymax'] >= 0)]

    # Expand bounding boxes by 30 pixels on all sides
    crop_annotations["xmin"] -= 30
    crop_annotations["ymin"] -= 30
    crop_annotations["xmax"] += 30
    crop_annotations["ymax"] += 30

    train_df, validation_df = train_test_split(crop_annotations)

    comet_logger = CometLogger(project_name=cfg.comet.project, workspace=cfg.comet.workspace)

    # Log train and val dataframes to comet
    train_csv_path = "/tmp/train_annotations.csv"
    val_csv_path = "/tmp/val_annotations.csv"
    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(val_csv_path, index=False)
    comet_logger.experiment.log_parameters(cfg.classification_model)
    comet_logger.experiment.log_parameter("num_classes", len(train_df['label'].unique()))
    comet_logger.experiment.log_table("train_annotations.csv", train_df)
    comet_logger.experiment.log_table("val_annotations.csv", validation_df)

    comet_logger.experiment.add_tag("classification")
    comet_id = comet_logger.experiment.id

    # Add a experiment stamp to not use the same image_dir for different runs
    train_crop_image_dir = os.path.join(cfg.classification_model.train_crop_image_dir, comet_id)
    os.makedirs(train_crop_image_dir, exist_ok=True)

    val_crop_image_dir = os.path.join(cfg.classification_model.val_crop_image_dir, comet_id)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        comet_logger=comet_logger,
        checkpoint=cfg.classification_model.checkpoint,
        checkpoint_dir=cfg.classification_model.checkpoint_dir,
        image_dir=cfg.classification_model.image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        fast_dev_run=cfg.classification_model.fast_dev_run,
        max_epochs=cfg.classification_model.max_epochs,
        lr=cfg.classification_model.lr,
        batch_size=cfg.classification_model.batch_size,
        workers=cfg.classification_model.workers,
    )
    
    checkpoint_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints/"
    trained_model.trainer.save_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"))

    image_dataset, true_label, predicted_label = trained_model.val_dataset_confusion(return_images=True)
    comet_logger.experiment.log_confusion_matrix(
            y_true=true_label,
            y_predicted=predicted_label,
            images=image_dataset,
            max_categories=len(trained_model.label_dict.keys()),
            labels=list(trained_model.label_dict.keys()),
        )
if __name__ == "__main__":
    main()

