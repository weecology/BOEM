import hydra
from omegaconf import DictConfig
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.pipeline_evaluation import PipelineEvaluation
from src.active_learning import choose_images
from src.reporting import Reporting
from src.pre_annotation_prediction import PreAnnotationPrediction
from src.label_studio import check_for_new_annotations, upload_to_label_studio
from src.model import Model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    # Check for new annotations
    new_annotations = check_for_new_annotations(**cfg.label_studio)
    if new_annotations is None:
        print("No new annotations, exiting")
        return None
    
    model_training = Model()
    trained_model = model_training.train_model(annotations)

    # Update the model path
    cfg.model.path = trained_model

    existing_model = cfg.model.path

    pipeline_monitor = PipelineEvaluation(trained_model)
    pipeline_monitor.review()
    performance = pipeline_monitor.check_performance(cfg.model.performance_threshold)
    reporting = Reporting()
    reporting.generate_reports(performance)

    if performance.success:
        print("Pipeline performance is satisfactory, exiting")
        return None
    else:
        images_to_annotate = choose_images(**cfg.choose_images)

        data_ingestion = DataIngestion(images_to_annotate)
        raw_data = data_ingestion.ingest_data()

        data_processing = DataProcessing()
        images_to_annotate = data_processing.process_data(raw_data)
        
        pre_annotation = PreAnnotationPrediction(existing_model)
        images_for_human_review, auto_annotated_images = pre_annotation.predict_and_divide(images_to_annotate)

        print(f"Images requiring human review: {len(images_for_human_review)}")
        print(f"Images auto-annotated: {len(auto_annotated_images)}")

        # Run the annotation pipeline
        annotations = upload_to_label_studio(images_for_human_review, **cfg)
        reporting.generate_reports(trained_model)

if __name__ == "__main__":
    # The workflow starts with an existing model and predictions
    main()
