import hydra
from omegaconf import DictConfig
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.pipeline_evaluation import PipelineEvaluation
from src.model_deployment import ModelDeployment
from src.monitoring import Monitoring
from src.annotation.pipeline import iterate as annotation_pipeline
from src.pre_annotation_prediction import PreAnnotationPrediction
from src.reporting import Reporting

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    existing_model = cfg.model.path
    pipeline_evaluation = PipelineEvaluation(existing_model)
    evaluation_results = pipeline_evaluation.human_review()
    if evaluation_results is None:
        raise ValueError(f"Pipeline evaluation is currently waiting for human review. There have been {pipeline_evaluation.num_annotations_submitted()} annotations submitted with a minimum of {pipeline_evaluation.min_annotations_needed()} needed for review")
    success = pipeline_evaluation.check_performance(evaluation_results, cfg.model.performance_threshold)

    if success:
        reporting = Reporting()
        reporting.generate_reports(pipeline_evaluation)
    else:
        data_ingestion = DataIngestion(evaluation_results)
        raw_data = data_ingestion.ingest_data()

        data_processing = DataProcessing()
        images_to_annotate = data_processing.process_data(raw_data)
        

        # Use the PreAnnotationPrediction class
        pre_annotation = PreAnnotationPrediction(existing_model)
        images_for_human_review, auto_annotated_images = pre_annotation.predict_and_divide(images_to_annotate)

        print(f"Images requiring human review: {len(images_for_human_review)}")
        print(f"Images auto-annotated: {len(auto_annotated_images)}")

        # Update images_to_annotate to only include those requiring human review
        images_to_annotate = images_for_human_review
        # Run the annotation pipeline
        annotations = annotation_pipeline(images_to_annotate, **cfg)

        if annotations is None:
            raise ValueError("Image annotation is currently waiting for human review. There have been {annotation_pipeline.num_annotations_submitted()} annotations submitted with a minimum of {annotation_pipeline.min_annotations_needed()} needed for model training")
        
        model_training = ModelTraining()
        trained_model = model_training.train_model(annotations)

        model_deployment = ModelDeployment()
        deployed_model = model_deployment.compare_models(trained_model, existing_model)

        # Update the model path
        cfg.model.path = deployed_model


if __name__ == "__main__":
    # The workflow starts with an existing model and predictions
    main()
