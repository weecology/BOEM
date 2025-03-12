import comet_ml
import os
import geopandas as gpd
from omegaconf import DictConfig

from src.active_learning import generate_pool_predictions, select_images, human_review
from src import label_studio
from src import detection
from src import classification
from src.pipeline_evaluation import PipelineEvaluation
from src.cluster import start
from pytorch_lightning.loggers import CometLogger
import glob
import pandas as pd

class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg
        self.sftp_client = label_studio.create_sftp_client(
            **self.config.server)

        # Pool of all images
        self.all_images = glob.glob(os.path.join(self.config.active_learning.image_dir, "*.jpg"))

        self.comet_logger = CometLogger(project_name=self.config.comet.project, workspace=self.config.comet.workspace)
        self.comet_logger.experiment.add_tag("pipeline")
        flight_name = os.path.basename(self.config.label_studio.images_to_annotate_dir)
        self.comet_logger.experiment.add_tag(flight_name)
        self.comet_logger.experiment.log_parameters(self.config)
        self.comet_logger.experiment.log_parameter("flight_name", flight_name)

    def save_model(self, model, directory):
        id = self.comet_logger.experiment.id
        checkpoint_path = os.path.join(directory, f"{id}.ckpt")
        if not os.path.exists(checkpoint_path):
            model.trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path
        
    def check_new_annotations(self, instance_name):
        instance_config = self.config.label_studio.instances[instance_name]
        return label_studio.check_for_new_annotations(
            url=self.config.label_studio.url,
            csv_dir=instance_config.csv_dir,
            project_name=instance_config.project_name,
            image_dir=self.config.active_learning.image_dir,
            )
    
    def check_annotations(self):

        if self.config.check_annotations:
            new_train_annotations = self.check_new_annotations("train")
            new_val_annotations = self.check_new_annotations("validation")
            new_review_annotations = self.check_new_annotations("review")

        self.existing_training = label_studio.gather_data(self.config.label_studio.instances.train.csv_dir, image_dir=self.config.active_learning.image_dir)
        self.existing_validation = label_studio.gather_data(self.config.label_studio.instances.validation.csv_dir, image_dir=self.config.active_learning.image_dir)
        self.existing_reviewed = label_studio.gather_data(self.config.label_studio.instances.review.csv_dir, image_dir=self.config.active_learning.image_dir)
        
        self.comet_logger.experiment.log_table(tabular_data=self.existing_reviewed, filename="human_reviewed_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_training, filename="training_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_validation, filename="validation_annotations.csv")
        
        print(f"Training annotations shape: {self.existing_training.shape}")
        print(f"Validation annotations shape: {self.existing_validation.shape}")
        print(f"Reviewed annotations shape: {self.existing_reviewed.shape}")

        self.existing_images = list(set(self.existing_training.image_path.tolist() + self.existing_validation.image_path.tolist() + self.existing_reviewed.image_path.tolist()))

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        self.check_annotations()
        
        # Assert no train in test
        #assert len(set(self.existing_training.image_path.tolist()).intersection(set(self.existing_validation.image_path.tolist()))) == 0

        if self.config.force_training:
            trained_detection_model = detection.preprocess_and_train(
                train_annotations=self.existing_training,
                validation_annotations=self.existing_validation,
                train_image_dir=self.config.active_learning.image_dir,
                crop_image_dir=self.config.detection_model.crop_image_dir,
                patch_size=self.config.detection_model.patch_size,
                patch_overlap=self.config.detection_model.patch_overlap,
                limit_empty_frac=self.config.detection_model.limit_empty_frac,
                checkpoint=self.config.detection_model.checkpoint,
                checkpoint_dir=self.config.detection_model.checkpoint_dir,
                trainer_config=self.config.detection_model.trainer,
                comet_logger=self.comet_logger)
            
            detection_checkpoint_path = self.save_model(trained_detection_model, self.config.detection_model.checkpoint_dir)
            self.comet_logger.experiment.log_asset(file_data=detection_checkpoint_path)

            # Remove the empty frames and object only labels
            classification_train = self.existing_training[~self.existing_training.label.isin(["FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
            classification_val = self.existing_validation[~self.existing_validation.label.isin(["FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
            
            classification_train = classification_train[classification_train.xmin != 0]
            classification_val = classification_val[classification_val.xmin != 0]
            
            if classification_train.empty:
                print("No training data for classification")
                trained_classification_model = classification.load(
                    self.config.classification_model.checkpoint,
                    checkpoint_dir=self.config.classification_model.checkpoint_dir,
                    num_classes=len(self.existing_training.label.unique()),
                    annotations=None,
                    checkpoint_num_classes=self.config.classification_model.checkpoint_num_classes,
                    checkpoint_train_dir=self.config.classification_model.checkpoint_train_dir)
            else:
                trained_classification_model = classification.preprocess_and_train(
                    train_df=classification_train,
                    validation_df=classification_val,
                    **self.config.classification_model,
                    comet_logger=self.comet_logger)            
                
                classification_checkpoint_path = self.save_model(trained_classification_model, self.config.classification_model.checkpoint_dir)      

                self.comet_logger.experiment.log_asset(file_data=classification_checkpoint_path)

        else:
            trained_detection_model = detection.load(checkpoint = self.config.detection_model.checkpoint)
            trained_classification_model = classification.load(
                self.config.classification_model.checkpoint,
                checkpoint_dir=self.config.classification_model.checkpoint_dir,
                annotations=self.existing_training,
                num_classes=self.config.classification_model.checkpoint_num_classes,
                checkpoint_train_dir=self.config.classification_model.checkpoint_train_dir)
            
            detection_checkpoint_path = self.config.detection_model.checkpoint
            classification_checkpoint_path = self.config.classification_model.checkpoint

        if self.config.pipeline.gpus > 1:
            dask_client = start(gpus=self.config.pipeline.gpus, mem_size="70GB")
        else:
            dask_client = None

        # Predict entire flightline
        flightline_predictions = generate_pool_predictions(
            image_dir=self.config.active_learning.image_dir,
            pool_limit=self.config.active_learning.pool_limit,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.active_learning.min_score,
            model=trained_detection_model,
            model_path=detection_checkpoint_path,
            dask_client=dask_client,
            batch_size=self.config.predict.batch_size,
            crop_model=trained_classification_model
        )

        if self.config.debug:
            # To be a minimum images to debug the pipeline, get the first 5 evaluation images
            image_paths = [os.path.join(self.config.active_learning.image_dir, x) for x in self.existing_validation.image_path.head(3).tolist()]
            evaluation_predictions = detection.predict(
                image_paths=image_paths,
                m=trained_detection_model,
                model_path=detection_checkpoint_path,
                dask_client=dask_client,
                crop_model=trained_classification_model,
                patch_size=self.config.active_learning.patch_size,
                patch_overlap=self.config.active_learning.patch_overlap,
                batch_size=self.config.predict.batch_size
            )
            evaluation_predictions = gpd.GeoDataFrame(pd.concat(evaluation_predictions), geometry="geometry")

        else:
            evaluation_predictions = flightline_predictions[flightline_predictions.image_path.isin(self.existing_validation.image_path)]
        
        detection_annotations = self.existing_validation
        classification_annotations = self.existing_training.copy(deep=True)
        
        # Remove empty frames from classification annotations
        classification_annotations = classification_annotations[~classification_annotations.label.isin(["FalsePositive", "Object","Bird", "Reptile", "Turtle", "Mammal","Artificial"])]
        classification_annotations = classification_annotations[classification_annotations.xmin != 0]
        classification_annotations = classification_annotations[~classification_annotations.label.isnull()]
        
        if classification_annotations.empty:
            print("No evaluation annotations")
        else:
            evaluation_predictions = evaluation_predictions[evaluation_predictions.xmin != 0]   
            evaluation_predictions = evaluation_predictions[~evaluation_predictions.label.isnull()]
            
            # All classifications should have numeric labels
            classification_annotations["cropmodel_label"] = classification_annotations["label"].apply(lambda x: trained_classification_model.label_dict[x])
            evaluation_predictions["cropmodel_label"] = evaluation_predictions["cropmodel_label"].apply(lambda x: trained_classification_model.label_dict[x])

            pipeline_monitor = PipelineEvaluation(
                predictions=evaluation_predictions,
                detection_annotations=detection_annotations,
                classification_annotations=classification_annotations,
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

        test_images_to_annotate, preannotations = select_images(
            preannotations=test_preannotations,
            strategy=self.config.active_testing.strategy,
            n=self.config.active_testing.n_images,
            )
        
        if len(test_images_to_annotate) == 0:
            print("No images to annotate in the test set")
        else:
            full_image_paths = [os.path.join(self.config.active_testing.image_dir, image) for image in test_images_to_annotate]
            label_studio.upload_to_label_studio(images=full_image_paths,
                                        sftp_client=self.sftp_client,
                                        url=self.config.label_studio.url,
                                        project_name=self.config.label_studio.instances.validation.project_name,
                                        images_to_annotate_dir=self.config.active_testing.image_dir,
                                        folder_name=self.config.label_studio.folder_name,
                                        preannotations=None)

        # Select images to annotate based on the strategy
        if self.config.debug:
            training_preannotations =  evaluation_predictions
        else:
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
            # Training annotation pipeline
            full_image_paths = [os.path.join(self.config.active_learning.image_dir, image) for image in train_images_to_annotate]
            #preannotations["cropmodel_label"] = preannotations["cropmodel_label"].apply(lambda x: trained_classification_model.numeric_to_label_dict[x])
            preannotations_list = [group for _, group in preannotations.groupby("image_path")]
            label_studio.upload_to_label_studio(images=full_image_paths, 
                                                url=self.config.label_studio.url,
                                                sftp_client=self.sftp_client, 
                                                project_name=self.config.label_studio.instances.train.project_name, 
                                                images_to_annotate_dir=self.config.active_learning.image_dir, 
                                                folder_name=self.config.label_studio.folder_name, 
                                                preannotations=preannotations_list)
        if self.config.debug:
            human_review_pool =  evaluation_predictions
        else:
            human_review_pool = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate + train_images_to_annotate)]
        
        confident_predictions, uncertain_predictions = human_review(
            confident_threshold=self.config.pipeline.confidence_threshold,
            min_score=self.config.active_learning.min_score,
            predictions=human_review_pool,
        )

        self.comet_logger.experiment.log_table(tabular_data=confident_predictions, filename="confident_predictions.csv")
        self.comet_logger.experiment.log_table(tabular_data=uncertain_predictions, filename="uncertain_predictions.csv") 

        # Human review - to be replaced by AWS for NJ Audubon
        if len(uncertain_predictions) == 0:
            print("No images to review")
        else:
            chosen_uncertain_images = uncertain_predictions.sort_values(by="score", ascending=False).head(self.config.human_review.n)["image_path"].unique()
            chosen_preannotations = uncertain_predictions[uncertain_predictions.image_path.isin(chosen_uncertain_images)]
            chosen_preannotations = [group for _, group in chosen_preannotations.groupby("image_path")]
            full_image_paths = [os.path.join(self.config.active_learning.image_dir, image) for image in chosen_uncertain_images]
            #chosen_preannotations["cropmodel_label"] = chosen_preannotations["cropmodel_label"].apply(lambda x: trained_classification_model.numeric_to_label_dict[x])
            label_studio.upload_to_label_studio(images=full_image_paths, 
                                                sftp_client=self.sftp_client, 
                                                url=self.config.label_studio.url,
                                                project_name=self.config.label_studio.instances.review.project_name, 
                                                images_to_annotate_dir=self.config.active_learning.image_dir, 
                                                folder_name=self.config.label_studio.folder_name, 
                                                preannotations=chosen_preannotations)

        print(f"Images requiring human review: {len(uncertain_predictions)}")
        print(f"Images auto-annotated: {len(confident_predictions)}")

        # Construct the final predictions, which are the existing train, test and human review overriding the auto-annotations
        if self.config.debug:
            final_predictions = evaluation_predictions.copy(deep=True)
        else:
            final_predictions = flightline_predictions.copy(deep=True)
            #final_predictions["cropmodel_label"] = final_predictions["cropmodel_label"].apply(lambda x: trained_classification_model.numeric_to_label_dict[x])
            final_predictions["set"] = "prediction"
        
        for dataset, label in [("existing_training", "train"), ("existing_validation", "validation"), ("existing_reviewed", "review")]:
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "cropmodel_label"] = getattr(self, dataset)["label"]
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "score"] = None
            final_predictions.loc[final_predictions.image_path.isin(getattr(self, dataset).image_path), "set"] = label

        self.comet_logger.experiment.log_table(tabular_data=final_predictions, filename="final_predictions.csv")