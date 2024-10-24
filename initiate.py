import hydra
from omegaconf import DictConfig
import torch
from src.model_deployment import ModelDeployment
from src.data_processing import DataProcessing

@hydra.main(version_base=None, config_path="conf", config_name="config")
def initiate(cfg: DictConfig):
    # Load the model
    model_path = cfg.model.path
    model = torch.load(model_path)
    
    # Prepare data for prediction
    data_processing = DataProcessing()
    test_data = data_processing.load_test_data(cfg.data.test_path)
    
    # Deploy model
    model_deployment = ModelDeployment()
    deployed_model = model_deployment.deploy_model(model)
    
    # Generate predictions
    predictions = deployed_model.predict(test_data)
    
    # Save predictions
    output_path = cfg.output.predictions_path
    torch.save(predictions, output_path)

    #Upload predictions to Label Studio
    
    
    print(f"Predictions generated and saved to {output_path}")

if __name__ == "__main__":
    initiate()
