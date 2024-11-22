from datetime import datetime
import os

import pandas as pd
from omegaconf import DictConfig

from src.active_learning import choose_train_images, choose_test_images, predict_and_divide
from src import propagate
from src import label_studio
from src.classification import preprocess_and_train_classification
from src.data_processing import density_cropping
from src.detection import preprocess_and_train
from src.pipeline_evaluation import PipelineEvaluation
from src.reporting import Reporting


class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_project = label_studio.connect_to_label_studio(
            url=self.config.label_studio.url,
            project_name=self.config.label_studio.project_name)
        self.sftp_client = label_studio.create_sftp_client(
            **self.config.server)

    def save_model(self, model, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(directory, f"model_{timestamp}.ckpt")
        model.trainer.save_checkpoint(checkpoint_path)

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_annotations = label_studio.check_for_new_annotations(
                **self.config.label_studio)
            if new_annotations is None:
                print("No new annotations, exiting")
                return None

            # Given new annotations, propogate labels to nearby images
            # label_propagator = propagate.LabelPropagator(
            #     **self.config.propagate)
            # label_propagator.through_time(new_annotations)

        if self.config.detection_model.validation_csv_path is not None:
            validation_df = pd.read_csv(
                self.config.detection_model.validation_csv_path)
        else:
            validation_df = None

        trained_detection_model = preprocess_and_train(
            self.config, validation_df=validation_df)
        trained_classification_model = preprocess_and_train_classification(
            self.config, validation_df=validation_df)

        self.save_model(trained_detection_model,
                        self.config.detection_model.checkpoint_dir)
        self.save_model(trained_classification_model,
                        self.config.classification_model.checkpoint_dir)

        pipeline_monitor = PipelineEvaluation(
            model=trained_detection_model,
            crop_model=trained_classification_model,
            **self.config.pipeline_evaluation)
        
        performance = pipeline_monitor.evaluate()

        reporter = Reporting(self.config.reporting.report_dir,
                            self.config.reporting.image_dir,
                            pipeline_monitor)

        if pipeline_monitor.check_success():
            print("Pipeline performance is satisfactory, exiting")
            reporter.generate_report()
            return None
        else:
            train_images_to_annotate = choose_train_images(
                evaluation=performance,
                image_dir=self.config.active_learning.image_dir,
                model=trained_detection_model,
                strategy=self.config.active_learning.strategy,
                n=self.config.active_learning.n_images,
                patch_size=self.config.active_learning.patch_size,
                patch_overlap=self.config.active_learning.patch_overlap,
                min_score=self.config.active_learning.min_score
            )
            
            test_images_to_annotate = choose_test_images(
                image_dir=self.config.active_testing.image_dir,
                model=trained_detection_model,
                strategy=self.config.active_testing.strategy,
                n=self.config.active_testing.n_images,
                patch_size=self.config.active_testing.patch_size,
                patch_overlap=self.config.active_testing.patch_overlap,
                min_score=self.config.active_testing.min_score)


            confident_predictions, uncertain_predictions = predict_and_divide(
                trained_detection_model, trained_classification_model,
                train_images_to_annotate, self.config.active_learning.patch_size,
                self.config.active_learning.patch_overlap,
                self.config.active_learning.confident_threshold)

            reporter.confident_predictions = confident_predictions
            reporter.uncertain_predictions = uncertain_predictions

            print(f"Images requiring human review: {len(confident_predictions)}")
            print(f"Images auto-annotated: {len(uncertain_predictions)}")

            # Intelligent cropping
            image_paths = uncertain_predictions["image_path"].unique()
            # cropped_image_annotations = density_cropping(
            # image_paths, uncertain_predictions, **self.config.intelligent_cropping)

            # Align the predictions with the cropped images
            # Run the annotation pipeline
            label_studio.upload_to_label_studio(self.sftp_client,
                                                uncertain_predictions,
                                                **self.config)
            label_studio.upload_to_label_studio(self.sftp_client,
                                                test_images_to_annotate,
                                                **self.config)
            reporter.generate_report()

