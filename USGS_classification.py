from deepforest import model
import pandas as pd
import os
import comet_ml
from pytorch_lightning.loggers import CometLogger
from src.classification import preprocess_and_train_classification
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Override the classification_model config with USGS.yaml
    cfg = hydra.compose(config_name="config", overrides=["classification_model=USGS"])
    
    classification_cfg = cfg.classification_model
    
    # From the detection script
    savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
    train = pd.read_csv(os.path.join(savedir, "train.csv"))
    test = pd.read_csv(os.path.join(savedir, "test.csv"))

    comet_logger = CometLogger(project_name=cfg.project, workspace=cfg.workspace)
    preprocess_and_train_classification(
        config=cfg,
        train_df=train,
        validation_df=test,
        comet_logger=comet_logger
    )

if __name__ == "__main__":
    main()

