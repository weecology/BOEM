from deepforest import model
import pandas as pd
import os
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train_classification
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf/classification_model", config_name="USGS")
def main(cfg: DictConfig):
    classification_cfg = cfg.classification
    savedir = classification_cfg.savedir
    train = pd.read_csv(os.path.join(savedir, "train.csv"))
    test = pd.read_csv(os.path.join(savedir, "test.csv"))

    comet_logger = CometLogger(project_name=classification_cfg.project_name, workspace=classification_cfg.workspace)
    preprocess_and_train_classification(
        config=cfg,
        train_df=train,
        validation_df=test,
        comet_logger=comet_logger
    )

if __name__ == "__main__":
    main()

