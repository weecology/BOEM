from datetime import datetime
import os

import pandas as pd
from omegaconf import DictConfig

from src.active_learning import choose_train_images, choose_test_images
from src import propagate
from src import label_studio
from src.model import preprocess_and_train
from src.pipeline_evaluation import PipelineEvaluation
from src.pre_annotation_prediction import PreAnnotationPrediction
from src.reporting import Reporting

class Pipeline:
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_project = label_studio.connect_to_label_studio(url=self.config.label_studio.url, project_name=self.config.label_studio.project_name)
        self.sftp_client = label_studio.create_sftp_client(**self.config.server) 
    
    def save_model(self, model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.config.train.checkpoint_dir, f"model_{timestamp}.ckpt")
        model.trainer.save_checkpoint(checkpoint_path)

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_annotations = label_studio.check_for_new_annotations(**self.config.label_studio)
            if new_annotations is None:
                print("No new annotations, exiting")
                return None
            
            # Given new annotations, propogate labels to nearby images
            label_propagator = propagate.LabelPropagator(**self.config.propagate)
            label_propagator.through_time(new_annotations)
            
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
            train_images_to_annotate = choose_train_images(performance, trained_model, **self.config.active_learning)
            test_images_to_annotate = choose_test_images(performance, **self.config.active_testing)

            pre_annotation = PreAnnotationPrediction(train_images_to_annotate)
            confident_annotations, uncertain_annotations = pre_annotation.predict_and_divide(train_images_to_annotate)

            print(f"Images requiring human review: {len(confident_annotations)}")
            print(f"Images auto-annotated: {len(uncertain_annotations)}")

            # Run the annotation pipeline
            label_studio.upload_to_label_studio(self.sftp_client, uncertain_annotations, **self.config)
            label_studio.upload_to_label_studio(self.sftp_client, test_images_to_annotate, **self.config)

            reporting.generate_reports(trained_model)
