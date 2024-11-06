import hydra
from omegaconf import DictConfig
from src.pipeline import Pipeline

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the application"""
    
    # Initialize and run pipeline
    pipeline = Pipeline(config=cfg)
    results = pipeline.run(model_path=cfg.model.path)
    
    # Log results
    print(f"Images needing review: {len(results['needs_review'])}")
    print(f"Images not needing review: {len(results['no_review_needed'])}")

if __name__ == "__main__":
    main()
