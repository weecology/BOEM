from datetime import datetime
import os

from omegaconf import DictConfig

from src.active_learning import choose_train_images, choose_test_images, predict_and_divide
from src import label_studio
from src import detection
from src import classification
from src.pipeline_evaluation import PipelineEvaluation
from src.reporting import Reporting
from src.cluster import start

class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.label_studio_project_train = label_studio.connect_to_label_studio(
            url=self.config.label_studio.url,
            project_name=self.config.label_studio.project_name_train)

        self.label_studio_project_validation = label_studio.connect_to_label_studio(
            url=self.config.label_studio.url,
            project_name=self.config.label_studio.project_name_validation)
        self.sftp_client = label_studio.create_sftp_client(
            **self.config.server)
        
        # Create reporting object
        self.create_reporter()

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
        
    def run(self):
        # Check for new annotations if the check_annotations flag is set
        if self.config.check_annotations:
            new_train_annotations = label_studio.check_for_new_annotations(
                sftp_client=self.sftp_client,
                url=self.config.label_studio.url,
                csv_dir=self.config.label_studio.csv_dir_train,
                project_name=self.config.label_studio.project_name_train,
                folder_name=self.config.label_studio.folder_name,
                images_to_annotate_dir=self.config.label_studio.images_to_annotate_dir,
                annotated_images_dir=self.config.label_studio.annotated_images_dir,
            )

            # Validation
            new_val_annotations = label_studio.check_for_new_annotations(
                sftp_client=self.sftp_client,
                url=self.config.label_studio.url,
                csv_dir=self.config.label_studio.csv_dir_validation,
                project_name=self.config.label_studio.project_name_validation,
                folder_name=self.config.label_studio.folder_name,
                images_to_annotate_dir=self.config.label_studio.images_to_annotate_dir,
                annotated_images_dir=self.config.label_studio.annotated_images_dir,
            )
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
            trained_detection_model = detection.preprocess_and_train(
                self.config)
            
            # No reason re-compute detection results later for validation
            try:
                detection_results = {"recall":trained_detection_model.trainer.logger.experiment.metrics["box_recall"],
                                 "precision":trained_detection_model.trainer.logger.experiment.metrics["box_precision"]}
            except:
                detection_results = None

            trained_classification_model = classification.preprocess_and_train_classification(
                self.config)

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
            **self.config.pipeline_evaluation)

        performance = pipeline_monitor.evaluate()
        self.reporter.detection_model = trained_detection_model
        self.reporter.classification_model = trained_classification_model
        self.reporter.pipeline_monitor = pipeline_monitor
        self.reporter.performance = performance

        if pipeline_monitor.check_success():
            print("Pipeline performance is satisfactory, exiting")
            return None
        if self.config.active_learning.gpus > 1:
            dask_client = start(gpus=self.config.active_learning.gpus, mem_size="70GB")
        else:
            dask_client = None
            
        test_images_to_annotate, testing_pool_predictions = choose_test_images(
            image_dir=self.config.active_testing.image_dir,
            model=trained_detection_model,
            strategy=self.config.active_testing.strategy,
            n=self.config.active_testing.n_images,
            patch_size=self.config.active_testing.patch_size,
            patch_overlap=self.config.active_testing.patch_overlap,
            min_score=self.config.active_testing.min_score,
            batch_size=self.config.predict.batch_size
            )
        
        self.reporter.testing_pool_predictions = testing_pool_predictions

        label_studio.upload_to_label_studio(images=test_images_to_annotate,
                                    sftp_client=self.sftp_client,
                                    label_studio_project=self.label_studio_project_validation,
                                    images_to_annotate_dir=self.config.active_testing.image_dir,
                                    folder_name=self.config.label_studio.folder_name,
                                    preannotations=None)

        train_images_to_annotate, training_pool_predictions = choose_train_images(
            evaluation=performance,
            image_dir=self.config.active_learning.image_dir,
            model_path=detection_checkpoint_path,
            model=trained_detection_model,
            strategy=self.config.active_learning.strategy,
            n=self.config.active_learning.n_images,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.active_learning.min_score,
            target_labels=self.config.active_learning.target_labels,
            pool_limit=self.config.active_learning.pool_limit,
            dask_client=dask_client,
            batch_size=self.config.predict.batch_size
        )

        self.reporter.training_pool_predictions = training_pool_predictions

        if len(train_images_to_annotate) > 0:
            confident_predictions, uncertain_predictions = predict_and_divide(
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

            print(f"Images requiring human review: {len(uncertain_predictions)}")
            print(f"Images auto-annotated: {len(confident_predictions)}")

            image_paths = uncertain_predictions["image_path"].unique()

            # Align the predictions with the cropped images
            # Run the annotation pipeline
            if len(image_paths) > 0:
                full_image_paths = [os.path.join(self.config.active_learning.image_dir, image) for image in image_paths]
                preannotations = [uncertain_predictions[uncertain_predictions["image_path"] == image_path] for image_path in image_paths]
                label_studio.upload_to_label_studio(images=full_image_paths, 
                                                    sftp_client=self.sftp_client, 
                                                    label_studio_project=self.label_studio_project_train, 
                                                    images_to_annotate_dir=self.config.active_learning.image_dir, 
                                                    folder_name=self.config.label_studio.folder_name, 
                                                    preannotations=preannotations)

        else:
            print("No images to annotate")
            confident_predictions = None
            uncertain_predictions = None
        
        self.reporter.generate_report(create_video=False)

