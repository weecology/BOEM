from datetime import datetime
import os

import pandas as pd
from omegaconf import DictConfig

from src.active_learning import choose_train_images, choose_test_images
from src import propagate
from src import label_studio
from src.model import preprocess_and_train, predict
from src.pipeline_evaluation import PipelineEvaluation
from src.reporting import Reporting

class Pipeline:
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_project = label_studio.connect_to_label_studio(url=self.config.label_studio.url, project_name=self.config.label_studio.project_name)
        self.sftp_client = label_studio.create_sftp_client(**self.config.server) 
    
    def save_model(self, model, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(directory, f"model_{timestamp}.ckpt")
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
            
        if self.config.detection_model.validation_csv_path is not None:
            validation_df = pd.read_csv(self.config.detection_model.validation_csv_path)
        else:
            validation_df = None

        trained_detection_model = preprocess_and_train(self.config, validation_df=validation_df, model_type="detection")
        trained_classification_model = preprocess_and_train(self.config, validation_df=validation_df, model_type="classification")

        self.save_model(trained_detection_model, self.config.detection_model.checkpoint_dir)
        self.save_model(trained_classification_model, self.config.classification_model.checkpoint_dir)

        pipeline_monitor = PipelineEvaluation(model=trained_detection_model, **self.config.pipeline_evaluation)
        performance = pipeline_monitor.evaluate()

        reporting = Reporting(self.config.reporting.report_dir)
        reporting.generate_reports(pipeline_monitor)

        if pipeline_monitor.check_success():
            print("Pipeline performance is satisfactory, exiting")
            return None
        else:
            train_images_to_annotate = choose_train_images(performance, trained_detection_model, **self.config.active_learning)
            test_images_to_annotate = choose_test_images(performance, **self.config.active_testing)

            predictions = predict(
                m=trained_detection_model,
                crop_model=trained_classification_model,
                image_paths= train_images_to_annotate, 
                patch_size=self.config.active_learning.patch_size, 
                patch_overlap=self.config.active_learning.patch_overlap, 
            )
            combined_predictions = pd.concat(predictions)

            # Split predictions into confident and uncertain
            confident_predictions = combined_predictions[combined_predictions["score"] > self.config.active_learning.confident_threshold]
            uncertain_predictions = combined_predictions[combined_predictions["score"] <= self.config.active_learning.confident_threshold]

            print(f"Images requiring human review: {len(confident_predictions)}")
            print(f"Images auto-annotated: {len(uncertain_predictions)}")

            # Run the annotation pipeline
            label_studio.upload_to_label_studio(self.sftp_client, uncertain_predictions, **self.config)
            label_studio.upload_to_label_studio(self.sftp_client, test_images_to_annotate, **self.config)