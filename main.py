import hydra
import os
from omegaconf import DictConfig
from src.pipeline import Pipeline
from src.label_studio import get_api_key

@hydra.main(config_path="boem_conf", config_name="boem_config")
def main(cfg: DictConfig):
    """Main entry point for the application"""
    api_key = get_api_key()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    if api_key is None:
        print("Warning: No Label Studio API key found in .comet.config")
        return None

    # Initialize and run pipeline
    pipeline = Pipeline(cfg=cfg)
    results = pipeline.run()

if __name__ == "__main__":
    main()
