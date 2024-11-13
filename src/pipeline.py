from omegaconf import DictConfig
from src.data_ingestion import DataIngestion
from src.pipeline_evaluation import PipelineEvaluation
from src.active_learning import choose_images
from src.reporting import Reporting
from src.pre_annotation_prediction import PreAnnotationPrediction
from src.label_studio import check_for_new_annotations, upload_to_label_studio, create_sftp_client, connect_to_label_studio
from src.model import preprocess_and_train
from src.data_processing import preprocess_images
import pandas as pd
from datetime import datetime
import os

class Pipeline:
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_project = connect_to_label_studio(url=self.config.label_studio.url, project_name=self.config.label_studio.project_name)
        self.sftp_client = create_sftp_client(**self.config.server) 
    
    def save_model(self, model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.config.train.checkpoint_dir, f"model_{timestamp}.ckpt")
        model.trainer.save_checkpoint(checkpoint_path)

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_annotations = check_for_new_annotations(**self.config.label_studio)
            if new_annotations is None:
                print("No new annotations, exiting")
                return None
            
        if self.config.train.validation_csv_path is not None:
            validation_df = pd.read_csv(self.config.validation_csv_path)
        else:
            validation_df = None

        trained_model = preprocess_and_train(self.config, validation_df=validation_df)
        self.save_model(trained_model)

        pipeline_monitor = PipelineEvaluation(model=trained_model, **self.config.pipeline_evaluation)
        performance = pipeline_monitor.evaluate()

        reporting = Reporting()
        reporting.generate_reports(pipeline_monitor)

        if performance.success:
            print("Pipeline performance is satisfactory, exiting")
            return None
        else:
            images_to_annotate = choose_images(performance, **self.config.choose_images)

            data_ingestion = DataIngestion(images_to_annotate)
            raw_data = data_ingestion.ingest_data()

            images_to_annotate = preprocess_images(raw_data)
            
            pre_annotation = PreAnnotationPrediction(trained_model)
            images_for_human_review, auto_annotated_images = pre_annotation.predict_and_divide(images_to_annotate)

            print(f"Images requiring human review: {len(images_for_human_review)}")
            print(f"Images auto-annotated: {len(auto_annotated_images)}")

            # Run the annotation pipeline
            annotations = upload_to_label_studio(self.sftp_client, images_for_human_review, **self.config)
            reporting.generate_reports(trained_model)