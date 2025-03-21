import comet_ml
import os
import geopandas as gpd
from omegaconf import DictConfig

from src.active_learning import generate_pool_predictions, select_images, human_review
from src import label_studio
from src import detection
from src import classification
from src.visualization import crop_images
from src.pipeline_evaluation import PipelineEvaluation
from pytorch_lightning.loggers import CometLogger
import glob
import pandas as pd
import random

class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig, dask_client=None):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.sftp_client = label_studio.create_sftp_client(
            **self.config.server)

        self.dask_client = dask_client

        # Pool of all images
        self.all_images = glob.glob(os.path.join(self.config.image_dir, "*.jpg"))

        self.comet_logger = CometLogger(project_name=self.config.comet.project, workspace=self.config.comet.workspace)
        self.comet_logger.experiment.add_tag("pipeline")
        flight_name = os.path.basename(self.config.image_dir)
        self.comet_logger.experiment.add_tag(flight_name)
        self.comet_logger.experiment.log_parameters(self.config)
        self.comet_logger.experiment.log_parameter("flight_name", flight_name)

        # The folders are relative to flight name
        self.config.label_studio.instances.train.csv_dir = os.path.join(self.config.label_studio.instances.train.csv_dir, flight_name)
        self.config.label_studio.instances.validation.csv_dir = os.path.join(self.config.label_studio.instances.validation.csv_dir, flight_name)
        self.config.label_studio.instances.review.csv_dir = os.path.join(self.config.label_studio.instances.review.csv_dir, flight_name)

        # Create the directories for the annotations
        os.makedirs(self.config.label_studio.instances.train.csv_dir, exist_ok=True)
        os.makedirs(self.config.label_studio.instances.validation.csv_dir, exist_ok=True)
        os.makedirs(self.config.label_studio.instances.review.csv_dir, exist_ok=True)

        self.config.detection_model.crop_image_dir = os.path.join(self.config.detection_model.crop_image_dir, flight_name)
        self.config.detection_model.checkpoint_dir = os.path.join(self.config.detection_model.checkpoint_dir, flight_name)
        self.config.classification_model.checkpoint_dir = os.path.join(self.config.classification_model.checkpoint_dir, flight_name)
        self.config.detection_model.crop_image_dir = os.path.join(self.config.detection_model.crop_image_dir, flight_name)
        self.config.classification_model.train_crop_image_dir = os.path.join(self.config.classification_model.train_crop_image_dir, flight_name)
        self.config.classification_model.val_crop_image_dir = os.path.join(self.config.classification_model.val_crop_image_dir, flight_name)

        # make sure the directories exist
        os.makedirs(self.config.detection_model.crop_image_dir, exist_ok=True)
        os.makedirs(self.config.detection_model.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.classification_model.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.detection_model.crop_image_dir, exist_ok=True)
        os.makedirs(self.config.classification_model.train_crop_image_dir, exist_ok=True)

        # Log src folder code
        self.comet_logger.experiment.log_code(folder=os.path.join(os.path.dirname(__file__), "../src"), overwrite=True)
        
    def check_new_annotations(self, instance_name):
        instance_config = self.config.label_studio.instances[instance_name]
        return label_studio.check_for_new_annotations(
            url=self.config.label_studio.url,
            csv_dir=instance_config.csv_dir,
            project_name=instance_config.project_name,
            image_dir=self.config.image_dir,
            )
    
    def check_annotations(self):

        if self.config.check_annotations:
            self.check_new_annotations("train")
            self.check_new_annotations("validation")
            self.check_new_annotations("review")

        self.existing_training = label_studio.gather_data(self.config.label_studio.instances.train.csv_dir, image_dir=self.config.image_dir)
        self.existing_validation = label_studio.gather_data(self.config.label_studio.instances.validation.csv_dir, image_dir=self.config.image_dir)
        self.existing_reviewed = label_studio.gather_data(self.config.label_studio.instances.review.csv_dir, image_dir=self.config.image_dir)
        
        self.comet_logger.experiment.log_table(tabular_data=self.existing_reviewed, filename="human_reviewed_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_training, filename="training_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_validation, filename="validation_annotations.csv")
        
        # If a brand new folder, there are no annotations, we need to start the pipeline from scratch, upload random images to label studio
        if self.existing_training is None and self.existing_validation is None and self.existing_reviewed is None:
            self.existing_images = None
            print("No existing annotations, starting from scratch")
            for instance in ["train", "validation", "review"]:
                if self.config.debug:
                    continue
                full_image_paths = [os.path.join(self.config.image_dir, image) for image in self.all_images]
                # Select 5 random images
                images_to_annotate = random.sample(full_image_paths, 5)
                full_image_paths = [os.path.join(self.config.image_dir, image) for image in images_to_annotate]
                label_studio.upload_to_label_studio(images=full_image_paths,
                                sftp_client=self.sftp_client,
                                url=self.config.label_studio.url,
                                project_name=self.config.label_studio.instances[instance].project_name,
                                images_to_annotate_dir=self.config.image_dir,
                                folder_name=self.config.label_studio.folder_name,
                                preannotations=None)
            return False

        else:
            if self.existing_training is not None:
                print(f"Training annotations shape: {self.existing_training.shape}")
            if self.existing_validation is not None:
                print(f"Validation annotations shape: {self.existing_validation.shape}")
            if self.existing_reviewed is not None:
                print(f"Reviewed annotations shape: {self.existing_reviewed.shape}")

        self.existing_images = list(set(
            (self.existing_training.image_path.tolist() if self.existing_training is not None else []) +
            (self.existing_validation.image_path.tolist() if self.existing_validation is not None else []) +
            (self.existing_reviewed.image_path.tolist() if self.existing_reviewed is not None else [])
        ))

        return True

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        status = self.check_annotations()
        
        # If no data is available, exit
        if not status:
            return None

        if self.config.force_training:
            trained_detection_model = detection.preprocess_and_train(
                train_annotations=self.existing_training,
                validation_annotations=self.existing_validation,
                train_image_dir=self.config.image_dir,
                crop_image_dir=self.config.detection_model.crop_image_dir,
                patch_size=self.config.detection_model.patch_size,
                patch_overlap=self.config.detection_model.patch_overlap,
                limit_empty_frac=self.config.detection_model.limit_empty_frac,
                checkpoint=self.config.detection_model.checkpoint,
                checkpoint_dir=self.config.detection_model.checkpoint_dir,
                trainer_config=self.config.detection_model.trainer,
                comet_logger=self.comet_logger)

            trained_classification_model = classification.preprocess_and_train(
                train_df=self.existing_training,
                validation_df=self.existing_validation,
                image_dir=self.config.image_dir,
                **self.config.classification_model,
                comet_logger=self.comet_logger)

        else:
            trained_detection_model = detection.load(checkpoint = self.config.detection_model.checkpoint)
            trained_classification_model = classification.load(
                self.config.classification_model.checkpoint,
                checkpoint_dir=self.config.classification_model.checkpoint_dir,
                annotations=self.existing_training,
                checkpoint_num_classes=self.config.classification_model.checkpoint_num_classes,
                checkpoint_train_dir=self.config.classification_model.checkpoint_train_dir)
            
        detection_checkpoint_path = self.comet_logger.experiment.get_parameter("detection_checkpoint_path")

        # Predict entire flightline
        trained_classification_model.num_workers = 0
        flightline_predictions = generate_pool_predictions(
            image_dir=self.config.image_dir,
            pool_limit=self.config.active_learning.pool_limit,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.predict.min_score,
            model=trained_detection_model,
            model_path=detection_checkpoint_path,
            dask_client=self.dask_client,
            batch_size=self.config.predict.batch_size,
            crop_model=trained_classification_model,
        )
        flightline_predictions["comet_id"] = self.comet_logger.experiment.id

        if self.config.debug:
            # To be a minimum images to debug the pipeline, get the first 5 evaluation images
            image_paths = [os.path.join(self.config.image_dir, x) for x in self.existing_validation.image_path.head(5).tolist()]
            
            evaluation_predictions = detection.predict(
                image_paths=image_paths,
                m=trained_detection_model,
                model_path=detection_checkpoint_path,
                dask_client=self.dask_client,
                crop_model=trained_classification_model,
                patch_size=self.config.active_learning.patch_size,
                patch_overlap=self.config.active_learning.patch_overlap,
                batch_size=self.config.predict.batch_size
            )
            if len(evaluation_predictions) == 0:
                evaluation_predictions = None
            else:
                evaluation_predictions = gpd.GeoDataFrame(pd.concat(evaluation_predictions), geometry="geometry")
                evaluation_predictions["comet_id"] = self.comet_logger.experiment.id
        else:
            evaluation_predictions = flightline_predictions[flightline_predictions.image_path.isin(self.existing_validation.image_path)]

        evaluation_annotations = self.existing_validation.copy(deep=True)
        
        if evaluation_annotations.empty:
            print("No annotations")
        else:
            pipeline_monitor = PipelineEvaluation(
                predictions=evaluation_predictions,
                annotations=evaluation_annotations,
                classification_label_dict=trained_classification_model.label_dict,
                **self.config.pipeline_evaluation)

            performance = pipeline_monitor.evaluate()
            self.comet_logger.experiment.log_metrics(performance)

            if pipeline_monitor.check_success():
                print("Pipeline performance is satisfactory, exiting")
                return None

        if self.config.debug:
            test_preannotations =  evaluation_predictions
        else:
            test_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images)]

        if test_preannotations is not None:
            test_images_to_annotate, preannotations = select_images(
                preannotations=test_preannotations,
                strategy=self.config.active_testing.strategy,
                n=self.config.active_testing.n_images,
                )
        else:
            test_images_to_annotate = []
        
        if len(test_images_to_annotate) == 0:
            print("No images to annotate in the test set")
        else:
            full_image_paths = [os.path.join(self.config.image_dir, image) for image in test_images_to_annotate]
            label_studio.upload_to_label_studio(images=full_image_paths,
                                        sftp_client=self.sftp_client,
                                        url=self.config.label_studio.url,
                                        project_name=self.config.label_studio.instances.validation.project_name,
                                        images_to_annotate_dir=self.config.image_dir,
                                        folder_name=self.config.label_studio.folder_name,
                                        preannotations=None)

        # Select images to annotate based on the strategy
        training_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate)]
        
        train_images_to_annotate, preannotations = select_images(
            preannotations=training_preannotations,
            strategy=self.config.active_learning.strategy,
            n=self.config.active_learning.n_images,
            target_labels=self.config.active_learning.target_labels
        )
        
        if len(train_images_to_annotate) == 0:
            print("No images to annotate in the training set")
        else:
            print(f"Training images to annotate: {len(train_images_to_annotate)}")
            # Training annotation pipeline
            full_image_paths = [os.path.join(self.config.image_dir, image) for image in train_images_to_annotate]
            preannotations_dict = {image_path: group for image_path, group in preannotations.groupby("image_path")}
            label_studio.upload_to_label_studio(images=full_image_paths, 
                                                url=self.config.label_studio.url,
                                                sftp_client=self.sftp_client, 
                                                project_name=self.config.label_studio.instances.train.project_name, 
                                                images_to_annotate_dir=self.config.image_dir, 
                                                folder_name=self.config.label_studio.folder_name, 
                                                preannotations=preannotations_dict)
        if self.config.debug:
            human_review_pool =  evaluation_predictions
        else:
            human_review_pool = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate + train_images_to_annotate)]
        
        confident_predictions, uncertain_predictions = human_review(
            confident_threshold=self.config.pipeline.confidence_threshold,
            min_score=self.config.active_learning.min_classification_score,
            predictions=human_review_pool,
        )

        self.comet_logger.experiment.log_table(tabular_data=confident_predictions, filename="confident_predictions.csv")
        self.comet_logger.experiment.log_table(tabular_data=uncertain_predictions, filename="uncertain_predictions.csv") 

        # Human review 
        if len(uncertain_predictions) == 0:
            print("No images to review")
        else:
            chosen_uncertain_images = uncertain_predictions.sort_values(by="score", ascending=False).head(self.config.human_review.n)["image_path"].unique()
            chosen_preannotations = uncertain_predictions[uncertain_predictions.image_path.isin(chosen_uncertain_images)]
            chosen_preannotations_dict = {image_path: group for image_path, group in chosen_preannotations.groupby("image_path")}
            full_image_paths = [os.path.join(self.config.image_dir, image) for image in chosen_uncertain_images]
            label_studio.upload_to_label_studio(images=full_image_paths, 
                                                sftp_client=self.sftp_client, 
                                                url=self.config.label_studio.url,
                                                project_name=self.config.label_studio.instances.review.project_name, 
                                                images_to_annotate_dir=self.config.image_dir, 
                                                folder_name=self.config.label_studio.folder_name, 
                                                preannotations=chosen_preannotations_dict)

        print(f"Images requiring human review: {len(uncertain_predictions)}")
        print(f"Images auto-annotated: {len(confident_predictions)}")

        # Construct the final predictions, which are the existing train, test and human review overriding the auto-annotations
        final_predictions = flightline_predictions.copy(deep=True)
        final_predictions["set"] = "prediction"
        
        for dataset, label in [("existing_training", "train"), ("existing_validation", "validation"), ("existing_reviewed", "review")]:
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "cropmodel_label"] = getattr(self, dataset)["label"]
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "score"] = None
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "set"] = label
        
        # Reset index
        final_predictions = final_predictions.reset_index(drop=True)
        
        # Write crops to disk
        urls = crop_images(final_predictions, root_dir=self.config.image_dir, experiment=self.comet_logger.experiment)
        final_predictions["crop_api_path"] = urls

        # crop_image_id
        final_predictions["crop_image_id"] = final_predictions.apply(
            lambda row: f"{os.path.splitext(os.path.basename(row['image_path']))[0]}_{row['cropmodel_label']}_{row.name}.png", axis=1)
        
        # give it a complete tag
        self.comet_logger.experiment.log_table(tabular_data=final_predictions, filename="final_predictions.csv")
        self.comet_logger.experiment.add_tag("complete")

        return None
