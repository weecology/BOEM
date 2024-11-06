import hydra
from omegaconf import DictConfig
from src.data_ingestion import DataIngestion
from src.pipeline_evaluation import PipelineEvaluation
from src.active_learning import choose_images
from src.reporting import Reporting
from src.pre_annotation_prediction import PreAnnotationPrediction
from src.label_studio import check_for_new_annotations, upload_to_label_studio
from src.model import preprocess_and_train
from src.data_processing import preprocess_images

class Pipeline:
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_client = None
        self.pre_annotation_predictor = None

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_annotations = check_for_new_annotations(**self.config.label_studio)
            if new_annotations is None:
                print("No new annotations, exiting")
                return None
        trained_model = preprocess_and_train(self.config)

        # Update the model path
        self.config.model.path = trained_model
        existing_model = self.config.model.path

        pipeline_monitor = PipelineEvaluation(trained_model)
        pipeline_monitor.review()
        performance = pipeline_monitor.check_performance(self.config.model.performance_threshold)
        reporting = Reporting()
        reporting.generate_reports(performance)

        if performance.success:
            print("Pipeline performance is satisfactory, exiting")
            return None
        else:
            images_to_annotate = choose_images(**self.config.choose_images)

            data_ingestion = DataIngestion(images_to_annotate)
            raw_data = data_ingestion.ingest_data()

            images_to_annotate = preprocess_images(raw_data)
            
            pre_annotation = PreAnnotationPrediction(existing_model)
            images_for_human_review, auto_annotated_images = pre_annotation.predict_and_divide(images_to_annotate)

            print(f"Images requiring human review: {len(images_for_human_review)}")
            print(f"Images auto-annotated: {len(auto_annotated_images)}")

            # Run the annotation pipeline
            annotations = upload_to_label_studio(images_for_human_review, **self.config)
            reporting.generate_reports(trained_model)