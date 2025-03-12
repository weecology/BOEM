from deepforest import model
import pandas as pd
import glob
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train
import hydra
from omegaconf import DictConfig
import os

# Create train test split, split each class into 90% train and 10% test with a minimum of 10 images per class for test and a max of 100
def train_test_split(df, test_size=0.1, min_test_images=10, max_test_images=100):
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

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Override the classification_model config with USGS.yaml
    cfg = hydra.compose(config_name="config", overrides=["classification_model=USGS"])
        
    # From the detection script
    crop_annotations = glob.glob("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/*.csv")
    crop_annotations = [pd.read_csv(x) for x in crop_annotations]
    crop_annotations = pd.concat(crop_annotations)
    
    # Keep labels with more than 100 images
    crop_annotations = crop_annotations.groupby("label").filter(lambda x: len(x) > 100)

    # Only keep two word labels
    crop_annotations = crop_annotations[crop_annotations["label"].str.contains(" ")]

    # Expand bounding boxes by 30 pixels on all sides
    crop_annotations["xmin"] -= 30
    crop_annotations["ymin"] -= 30
    crop_annotations["xmax"] += 30
    crop_annotations["ymax"] += 30
    
    train_df, validation_df = train_test_split(crop_annotations)
    
    comet_logger = CometLogger(project_name=cfg.comet.project, workspace=cfg.comet.workspace)
    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        comet_logger=comet_logger,
        **cfg.classification_model
    )

    comet_id = comet_logger.experiment.id
    checkpoint_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints/"
    trained_model.trainer.save_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"))

    # Confirm it can be loaded
    confirmed_load = model.CropModel.load_from_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"), num_classes=trained_model.num_classes)

if __name__ == "__main__":
    main()

