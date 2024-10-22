import hydra
from omegaconf import DictConfig
from data_ingestion import DataIngestion
from data_processing import DataProcessing
from model_training import ModelTraining
from model_evaluation import ModelEvaluation
from model_deployment import ModelDeployment
from monitoring import Monitoring
from annotation.pipeline import iterate as annotation_pipeline

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Run the original pipeline
    data_ingestion = DataIngestion()
    raw_data = data_ingestion.ingest_data()

    data_processing = DataProcessing()
    processed_data = data_processing.process_data(raw_data)

    model_training = ModelTraining()
    trained_model = model_training.train_model(processed_data)

    model_evaluation = ModelEvaluation()
    evaluation_results = model_evaluation.evaluate_model(trained_model, processed_data)

    if evaluation_results['performance'] >= cfg.model.performance_threshold:
        model_deployment = ModelDeployment()
        deployed_model = model_deployment.deploy_model(trained_model)

        monitoring = Monitoring()
        monitoring.start_monitoring(deployed_model)
    else:
        print(f"Model performance below threshold of {cfg.model.performance_threshold}. Not deploying.")

    # Run the annotation pipeline
    annotation_pipeline(**cfg)

if __name__ == "__main__":
    main()
