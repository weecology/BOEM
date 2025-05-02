import hydra
import os
from omegaconf import DictConfig
from src.pipeline import Pipeline
from src.label_studio import get_api_key
from src.cluster import start

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the application"""
    api_key = get_api_key()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    if api_key is None:
        print("Warning: No Label Studio API key found in .comet.config")
        return None
    
    if cfg.pipeline.gpus > 1:
        dask_client = start(gpus=cfg.pipeline.gpus, mem_size="100GB")

        def test_path():
            import sys

            # Add the path to the src directory to the Python path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
            
            return sys.path
        
        print("dask response for path {}".format(dask_client.run(test_path)))
        
        def test_src():
            from src import detection

        print("dask response for src {}".format(dask_client.run(test_src)))
        
    else:
        dask_client = None

    # Initialize and run pipeline
    pipeline = Pipeline(cfg=cfg, dask_client=dask_client)
    results = pipeline.run()

if __name__ == "__main__":
    main()
