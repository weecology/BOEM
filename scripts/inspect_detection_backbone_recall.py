import os
import sys
from pathlib import Path

# Resolve config from project root (script may run from any cwd)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import hydra
from omegaconf import DictConfig
import pandas as pd
from deepforest import main
from deepforest.visualize import plot_results

@hydra.main(config_path=str(PROJECT_ROOT / "boem_conf"), config_name="boem_config.yaml")
def main_inspect(cfg: DictConfig):
    # Load model from checkpoint
    checkpoint_path = "/blue/ewhite/b.weinstein/BOEM/training/checkpoints/9a203c4f18b942f3a946ea5db0670524.pl"
    print(f"Loading model from checkpoint: {checkpoint_path}")
    m = main.deepforest.load_from_checkpoint(checkpoint_path)

    m.label_dict = {"Object": 0}
    m.numeric_to_label_dict = {0: "Object"}

    savedir = "/blue/ewhite/b.weinstein/BOEM/training/crops"
    m.config["validation"]["csv_file"] = os.path.join(savedir,"test.csv")
    m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/BOEM/training/crops"

    # Evaluate
    print("Running evaluation...")
    results = m.evaluate(
        csv_file=m.config["validation"]["csv_file"],
        root_dir= m.config["validation"]["root_dir"],
        batch_size=cfg.predict.batch_size
    )
    print("Evaluation results:", results)

    # Print recall scores
    print(f"Recall: {results['box_recall']}")
    print(f"Recall: {results['box_precision']}")

    missing_images = results["results"][results['results']["match"] == False]

    

if __name__ == "__main__":
    main_inspect()