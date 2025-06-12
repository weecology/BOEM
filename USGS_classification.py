from deepforest import model
import pandas as pd
import glob
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train
import hydra
from omegaconf import DictConfig
import os
import cv2
import torch.nn.functional as F

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

    # Expand bounding boxes by 30 pixels on all sides
    crop_annotations["xmin"] -= 30
    crop_annotations["ymin"] -= 30
    crop_annotations["xmax"] += 30
    crop_annotations["ymax"] += 30

    train_df, validation_df = train_test_split(crop_annotations)

    # Plot all the Tursiops truncatus images as a sanity check
    import matplotlib.pyplot as plt

    # Get unique image paths for Tursiops truncatus in the training set
    tursiops_images = train_df[train_df['label'] == 'Tursiops truncatus']['image_path'].unique()
    n_plot = min(20, len(tursiops_images))

    for img_path in pd.Series(tursiops_images).sample(n=n_plot, random_state=42):
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_rgb)
        plt.title("Tursiops truncatus")
        plt.axis('off')
        # Save to a temporary file
        tmp_img_path = "/tmp/tursiops_plot.png"
        plt.savefig(tmp_img_path, bbox_inches='tight')
        plt.close()
        # Log image to comet
        comet_logger.experiment.log_image(
            tmp_img_path,
            metadata={"name": os.path.basename(img_path), "context": "tursiops_sanity_check"}
        )


    comet_logger = CometLogger(project_name=cfg.comet.project, workspace=cfg.comet.workspace)
    comet_logger.experiment.add_tag("classification")
    comet_id = comet_logger.experiment.id
    trained_model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        comet_logger=comet_logger,
        checkpoint=cfg.classification_model.checkpoint,
        checkpoint_dir=cfg.classification_model.checkpoint_dir,
        image_dir=cfg.classification_model.image_dir,
        train_crop_image_dir=cfg.classification_model.train_crop_image_dir,
        val_crop_image_dir=cfg.classification_model.val_crop_image_dir,
        fast_dev_run=cfg.classification_model.fast_dev_run,
        max_epochs=cfg.classification_model.max_epochs,
        lr=cfg.classification_model.lr,
        batch_size=cfg.classification_model.batch_size,
        workers=cfg.classification_model.workers
    )

    checkpoint_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints/"
    trained_model.trainer.save_checkpoint(os.path.join(checkpoint_dir,f"{comet_id}.ckpt"))
            

if __name__ == "__main__":
    main()

