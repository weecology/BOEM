from datetime import datetime
import os

from omegaconf import DictConfig

from src.active_learning import generate_training_pool_predictions, select_train_images, choose_test_images, human_review
from src import label_studio
from src import detection
from src import classification
from src.pipeline_evaluation import PipelineEvaluation
from src.reporting import Reporting
from src.cluster import start
from pytorch_lightning.loggers import CometLogger
import glob

class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.sftp_client = label_studio.create_sftp_client(
            **self.config.server)
        
        # Create reporting object
        self.create_reporter()

        # Pool of all images
        self.all_images = glob.glob(os.path.join(self.config.active_learning.image_dir, "*.jpg"))

        self.comet_logger = CometLogger(project_name=self.comet.project, workspace=self.comet.workspace)


    def save_model(self, model, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(directory, f"model_{timestamp}.ckpt")
        model.trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path
    
    def create_reporter(self):
        self.reporter = Reporting(
                report_dir=self.config.reporting.report_dir,
                image_dir=self.config.active_learning.image_dir,
                thin_factor=self.config.reporting.thin_factor,
                patch_overlap=self.config.active_learning.patch_overlap,
                patch_size=self.config.active_learning.patch_size,
                metadata_csv=self.config.reporting.metadata,
                batch_size=self.config.predict.batch_size
                )
        
    def check_new_annotations(self, instance_name):
        instance_config = self.config.label_studio.instances[instance_name]
        return label_studio.check_for_new_annotations(
            sftp_client=self.sftp_client,
            url=self.config.label_studio.url,
            csv_dir=instance_config.csv_dir,
            project_name=instance_config.project_name,
            folder_name=self.config.label_studio.folder_name,
            images_to_annotate_dir=self.config.label_studio.images_to_annotate_dir,
            annotated_images_dir=self.config.label_studio.annotated_images_dir,
        )

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_train_annotations = self.check_new_annotations("train")

            # Validation
            new_val_annotations = self.check_new_annotations("validation")

            # Human review 
            new_review_annotations = self.check_new_annotations("review")
            self.review_annotations = label_studio.gather_data(self.label_studio.instances.review.csv_dir)
            
            if new_val_annotations is None:
                if self.config.force_upload:
                    print("No new annotations, but force_upload is set to True, continuing")
                elif not self.config.force_training:
                    print("No new annotations, exiting")
                    return None
                else:
                    print(f"No new annotations, but force training is {self.config.force_training} and force upload is {self.config.force_upload}, continuing")
            else:   
                try:
                    print(f"New train annotations found: {len(new_train_annotations)}")
                except:
                    pass
                print(f"New val annotations found: {len(new_val_annotations)}")

        if self.config.force_training:
            trained_detection_model = detection.preprocess_and_train(self.config, comet_logger=self.comet_logger)
            
            # No reason re-compute detection results later for validation
            try:
                detection_results = {"recall":trained_detection_model.trainer.logger.experiment.metrics["box_recall"],
                                 "precision":trained_detection_model.trainer.logger.experiment.metrics["box_precision"]}
            except:
                detection_results = None

            trained_classification_model = classification.preprocess_and_train_classification(self.config, self.comet_logger)

            detection_checkpoint_path = self.save_model(trained_detection_model,
                            self.config.detection_model.checkpoint_dir)
            
            classification_checkpoint_path = self.save_model(trained_classification_model,
                            self.config.classification_model.checkpoint_dir)      
 
            self.reporter.detection_checkpoint_path = detection_checkpoint_path
            self.reporter.classification_checkpoint_path = classification_checkpoint_path

        else:
            detection_checkpoint_path = self.config.detection_model.checkpoint

            trained_detection_model = detection.load(
                checkpoint = self.config.detection_model.checkpoint)
            
            if self.config.classification_model.checkpoint is not None:
                trained_classification_model = classification.load(
                    self.config.classification_model.checkpoint, checkpoint_dir=self.config.classification_model.checkpoint_dir, annotations=None)
            else:
                annotations = label_studio.gather_data(self.config.classification_model.train_csv_folder)
                trained_classification_model = classification.load(
                    checkpoint = None, checkpoint_dir=self.config.classification_model.checkpoint_dir, annotations=annotations)

        pipeline_monitor = PipelineEvaluation(
            model=trained_detection_model,
            crop_model=trained_classification_model,
            batch_size=self.config.predict.batch_size,
            detection_results=detection_results,
            comet_logger=self.comet_logger,
            **self.config.pipeline_evaluation)

        performance = pipeline_monitor.evaluate()

        if pipeline_monitor.check_success():
            print("Pipeline performance is satisfactory, exiting")
            return None
        if self.config.active_learning.gpus > 1:
            dask_client = start(gpus=self.config.active_learning.gpus, mem_size="70GB")
        else:
            dask_client = None
            
        test_images_to_annotate, preannotations = choose_test_images(
            image_dir=self.config.active_testing.image_dir,
            model=trained_detection_model,
            strategy=self.config.active_testing.strategy,
            n=self.config.active_testing.n_images,
            patch_size=self.config.active_testing.patch_size,
            patch_overlap=self.config.active_testing.patch_overlap,
            min_score=self.config.active_testing.min_score,
            batch_size=self.config.predict.batch_size,
            comet_logger=self.comet_logger
            )
        
        
        label_studio_project = label_studio.connect_to_label_studio(url=self.config.label_studio.url, project_name=self.config.label_studio.instances.validation.project_name)
        label_studio.upload_to_label_studio(images=test_images_to_annotate,
                                    sftp_client=self.sftp_client,
                                    label_studio_project=label_studio_project,
                                    images_to_annotate_dir=self.config.active_testing.image_dir,
                                    folder_name=self.config.label_studio.folder_name,
                                    preannotations=None)

        # Generate predictions for the training pool
        training_pool_predictions = generate_training_pool_predictions(
            image_dir=self.config.active_learning.image_dir,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.active_learning.min_score,
            model=trained_detection_model,
            model_path=detection_checkpoint_path,
            dask_client=dask_client,
            batch_size=self.config.predict.batch_size,
            comet_logger=self.comet_logger
        )

        # Select images to annotate based on the strategy
        train_images_to_annotate, preannotations = select_train_images(
            preannotations=training_pool_predictions,
            strategy=self.config.active_learning.strategy,
            n=self.config.active_learning.n_images,
            target_labels=self.config.active_learning.target_labels
        )
        
        # Training annotation pipeline
        full_image_paths = [os.path.join(self.config.active_learning.image_dir, image) for image in train_images_to_annotate]
        label_studio.upload_to_label_studio(images=full_image_paths, 
                                            sftp_client=self.sftp_client, 
                                            label_studio_project=self.label_studio_project_train, 
                                            images_to_annotate_dir=self.config.active_learning.image_dir, 
                                            folder_name=self.config.label_studio.folder_name, 
                                            preannotations=preannotations)
        
        # Prediction annotation pipeline, choose images for humans to review
        confident_predictions, uncertain_predictions = human_review(
            detection_model=trained_detection_model,
            classification_model=trained_classification_model,
            image_paths=train_images_to_annotate,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            confident_threshold=self.config.pipeline.confidence_threshold,
            min_score=self.config.active_learning.min_score,
            batch_size=self.config.predict.batch_size,
            existing_predictions=training_pool_predictions
        )


        # Human review - to be replaced by AWS for NJ Audubon
        chosen_uncertain_images = uncertain_predictions.sort_values(by="score", ascending=False).head(self.config.human_review.n)["image_path"].tolist()
        chosen_preannotations = uncertain_predictions[uncertain_predictions.image_path.isin(chosen_uncertain_images)]
        label_studio.upload_to_label_studio(images=chosen_uncertain_images, 
                                            sftp_client=self.sftp_client, 
                                            label_studio_project=self.label_studio_project_review, 
                                            images_to_annotate_dir=self.config.active_learning.image_dir, 
                                            folder_name=self.config.label_studio.folder_name, 
                                            preannotations=chosen_preannotations)

        print(f"Images requiring human review: {len(uncertain_predictions)}")
        print(f"Images auto-annotated: {len(confident_predictions)}")
        self.chosen_uncertain_images = chosen_preannotations 
        self.reporter.generate_report(create_video=False)