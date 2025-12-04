import comet_ml
import os
from omegaconf import DictConfig

from src.active_learning import generate_pool_predictions, select_images, human_review
from deepforest.model import CropModel
from src import label_studio
from src import sagemaker_gt
from src.annotators import get_annotator
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
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg

        # Generic annotator
        self.annotator = get_annotator(self.config)

        # Pool of all images
        self.all_images = glob.glob(os.path.join(self.config.image_dir, "*.jpg")) + glob.glob(os.path.join(self.config.image_dir, "*.JPG"))

        self.comet_logger = CometLogger(project_name=self.config.comet.project, workspace=self.config.comet.workspace)
        self.comet_logger.experiment.add_tag("pipeline")
        flight_name = os.path.basename(self.config.image_dir)
        self.comet_logger.experiment.add_tag(flight_name)
        self.comet_logger.experiment.log_parameters(self.config)
        self.comet_logger.experiment.log_parameter("flight_name", flight_name)

        # Directories prepared by the annotator impl (for LS)

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
        os.makedirs(self.config.classification_model.val_crop_image_dir, exist_ok=True)

        # Log src folder code
        self.comet_logger.experiment.log_code(folder=os.path.join(os.path.dirname(__file__), "../src"), overwrite=True)
        
    def check_new_annotations(self, instance_name):
        return self.annotator.check_for_new_annotations(instance_name, image_dir=self.config.image_dir)
    
    def check_annotations(self):

        if self.config.check_annotations:
            self.check_new_annotations("train")
            self.check_new_annotations("validation")
            self.check_new_annotations("review")

        self.existing_training = self.annotator.gather_data("train", image_dir=self.config.image_dir)
        self.existing_validation = self.annotator.gather_data("validation", image_dir=self.config.image_dir)
        self.existing_reviewed = self.annotator.gather_data("review", image_dir=self.config.image_dir)

        self.comet_logger.experiment.log_table(tabular_data=self.existing_reviewed, filename="human_reviewed_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_training, filename="training_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_validation, filename="validation_annotations.csv")
        
        # If a brand new folder, there are no annotations, we need to start the pipeline from scratch, upload random images to label studio
        if self.existing_training is None and self.existing_validation is None and self.existing_reviewed is None:
            self.existing_images = None
            print("No existing annotations, starting from scratch")
            # Do not auto-upload on empty start
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

        # If there are no annotations in any set, turn off force training
        if len(self.existing_images) == 0:
            self.config.force_training = False
            print("No existing annotations, turning off force training")
        
        all_training = pd.concat([self.existing_training, self.existing_reviewed])
        
        if self.config.detection_model.force_train:
            trained_detection_model = detection.preprocess_and_train(
                train_annotations=all_training,
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
        else:
            trained_detection_model = detection.load(checkpoint=self.config.detection_model.checkpoint)
            self.comet_logger.experiment.log_parameter("detection_checkpoint_path",self.config.detection_model.checkpoint)
        
        classification_backend = getattr(self.config.classification_model, "backend", "deepforest")

        # Create crop image directories if they don't exist
        train_crop_image_dir = os.path.join(self.config.classification_model.train_crop_image_dir, self.comet_logger.experiment.id)
        os.makedirs(train_crop_image_dir, exist_ok=True)

        val_crop_image_dir = os.path.join(self.config.classification_model.val_crop_image_dir, self.comet_logger.experiment.id)
        os.makedirs(val_crop_image_dir, exist_ok=True)

        if classification_backend == "deepforest":
            # If there are no train annotations, turn off force training
            if all_training.xmin[all_training.xmin != 0].empty:
                self.config.classification_model.force_train = False
                print("No training annotations, turning off force training")
            
            # If there are no validation annotations, turn off force training
            if self.existing_validation is None:
                self.config.classification_model.force_train = False
                print("No validation annotations, turning off force training")
            elif self.existing_validation.xmin[self.existing_validation.xmin!= 0].empty:
                self.config.classification_model.force_train = False
                print("No validation annotations, turning off force training")
            
            if self.config.classification_model.force_train:
                trained_classification_model = classification.preprocess_and_train(
                    train_df=all_training,
                    validation_df=self.existing_validation,
                    image_dir=self.config.image_dir,
                    checkpoint=self.config.classification_model.checkpoint,
                    checkpoint_num_classes=self.config.classification_model.checkpoint_num_classes,
                    checkpoint_dir=self.config.classification_model.checkpoint_dir,
                    train_crop_image_dir=train_crop_image_dir,
                    val_crop_image_dir=val_crop_image_dir,
                    fast_dev_run=self.config.classification_model.fast_dev_run,
                    max_epochs=self.config.classification_model.max_epochs,
                    lr=self.config.classification_model.lr,
                    batch_size=self.config.classification_model.batch_size,
                    workers=self.config.classification_model.workers,
                    comet_logger=self.comet_logger)
            else:
                # HOT FIX: For old deepforest, add num_classes to the checkpoint
                trained_classification_model = CropModel.load_from_checkpoint(self.config.classification_model.checkpoint)
        else:
            raise NotImplementedError("Only deepforest classification backend is currently implemented")

        pool = glob.glob(os.path.join(self.config.image_dir, "*.jpg")) + glob.glob(os.path.join(self.config.image_dir, "*.JPG"))  + glob.glob(os.path.join(self.config.image_dir, "*.jpeg")) + glob.glob(os.path.join(self.config.image_dir, "*.JPEG"))  # Get all images in the data directory
        pool = [image for image in pool if not image.endswith('.csv')]
        pool = [image for image in pool if image not in self.existing_images]

        if self.config.debug:
            if self.existing_validation is not None:
                non_empty_validation = self.existing_validation[~(self.existing_validation.xmin==0)]
                pool = list(non_empty_validation.image_path.unique())
                pool = [os.path.join(self.config.image_dir, image) for image in pool][:10]
            else:
                pool = random.sample(pool, 10)

        flightline_predictions = generate_pool_predictions(
            pool=pool,
            pool_limit=self.config.active_learning.pool_limit,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.predict.min_score,
            model=trained_detection_model,
            batch_size=self.config.predict.batch_size,
            crop_model=trained_classification_model,
        )

        if flightline_predictions is None:
            print("No predictions")
            return None
        
        flightline_predictions["comet_id"] = self.comet_logger.experiment.id

        if self.existing_validation is None:
            print("No validation annotations, skipping evaluation")       
        else:
            evaluation_annotations = self.existing_validation.copy(deep=True)
            evaluation_predictions = flightline_predictions[flightline_predictions.image_path.isin(self.existing_validation.image_path)]


            label_dict = trained_classification_model.label_dict
                
            pipeline_monitor = PipelineEvaluation(
                predictions=evaluation_predictions,
                annotations=evaluation_annotations,
                classification_label_dict=label_dict,
                **self.config.pipeline_evaluation)

            performance = pipeline_monitor.evaluate()
            self.comet_logger.experiment.log_metrics(performance)

            if pipeline_monitor.check_success():
                print("Pipeline performance is satisfactory, exiting")
                return None

        test_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images)]
        test_images_to_annotate, preannotations = select_images(
            preannotations=test_preannotations,
            strategy="random",
           n=self.config.active_testing.n_images,
        )

        print(f"Test images to annotate: {len(test_images_to_annotate)}")

        # Select images to annotate based on the strategy
        training_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate)]
        
        train_images_to_annotate, preannotations = select_images(
            preannotations=training_preannotations,
            strategy=self.config.active_learning.strategy,
            n=self.config.active_learning.n_images,
        )
        
        print(f"Training images to annotate: {len(train_images_to_annotate)}")
        human_review_pool = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate + train_images_to_annotate)]
        
        confident_predictions, uncertain_predictions = human_review(
            confident_threshold=self.config.pipeline.confidence_threshold,
            min_classification_score=self.config.active_learning.min_classification_score,
            min_detection_score=self.config.active_learning.min_detection_score,
            predictions=human_review_pool.copy(deep=True),
        )

        self.comet_logger.experiment.log_table(tabular_data=confident_predictions, filename="confident_predictions.csv")
        self.comet_logger.experiment.log_table(tabular_data=uncertain_predictions, filename="uncertain_predictions.csv") 

        # Human review 
        if len(uncertain_predictions) == 0:
            print("No images to review")
            review_images_to_annotate = []
        else:
            review_images_to_annotate = uncertain_predictions.sort_values(by="score", ascending=False).head(self.config.human_review.n)["image_path"].unique()

        print(f"Images requiring human review: {len(uncertain_predictions.image_path.unique())}")
        print(f"Images auto-annotated: {len(confident_predictions.image_path.unique())}")

        # Construct the final predictions, which are the existing train, test and human review overriding the auto-annotations
        final_predictions = flightline_predictions.copy(deep=True)
        final_predictions["set"] = "prediction"

        # Add in the existing training and validation annotations
        if self.existing_training is not None:
            existing_training = self.existing_training.copy(deep=True)
            existing_training["set"] = "train"
            # Change label to cropmodel_label
            existing_training.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a cropmodel_score column
            existing_training["cropmodel_score"] = 1.0
            # Add an object score
            existing_training["label"] = "Object"
            existing_training["score"] = 1.0

            if self.config.debug:
                existing_training = existing_training[:10]

            final_predictions = pd.concat([final_predictions, existing_training], ignore_index=True)

        if self.existing_validation is not None:
            existing_validation = self.existing_validation.copy(deep=True)
            existing_validation["set"] = "validation"
            # Change label to cropmodel_label
            existing_validation.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a cropmodel_score column
            existing_validation["cropmodel_score"] = 1.0

            # Add an object score
            existing_validation["label"] = "Object"
            existing_validation["score"] = 1.0

            if self.config.debug:
                existing_validation = existing_validation[:10]
            final_predictions = pd.concat([final_predictions, existing_validation], ignore_index=True)

        # Add in the human reviewed annotations
        if self.existing_reviewed is not None:
            existing_reviewed = self.existing_reviewed.copy(deep=True)
            # Change label to cropmodel_label
            existing_reviewed.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a set column
            existing_reviewed["cropmodel_score"] = 1.0
            
            #Set detection column
            existing_reviewed["label"] = "Object"
            # Add an object score
            existing_reviewed["score"] = 1.0
            existing_reviewed["set"] = "reviewed"

            if self.config.debug:
                existing_reviewed = existing_reviewed[:10]

            final_predictions = pd.concat([final_predictions, existing_reviewed], ignore_index=True)
        
        # Reset index
        final_predictions = final_predictions.reset_index(drop=True)

        # add comet_id
        final_predictions["comet_id"] = self.comet_logger.experiment.id
        
        # Remove False Positives and empty images
        final_predictions = final_predictions[~final_predictions.cropmodel_label.isin(["FalsePositive", "0",0,"Object"])]

        if final_predictions.empty:
            print("No predictions")
            return None
        
        # Write crops to disk
        image_paths = crop_images(final_predictions, root_dir=self.config.image_dir, experiment=self.comet_logger.experiment, expand=self.config.predict.buffer)

        # crop_image_id
        final_predictions["crop_image_id"] = image_paths
        
        # give it a complete tag
        self.comet_logger.experiment.log_table(tabular_data=final_predictions, filename="final_predictions.csv")

        # Use generic annotator to upload for each instance
        for instance, image_basenames in {"train": train_images_to_annotate, "validation": test_images_to_annotate, "review": review_images_to_annotate}.items():
            if len(image_paths) == 0:
                print(f"No images to upload for instance {instance}, skipping")
                continue

            image_paths = [os.path.join(self.config.image_dir, x) for x in image_basenames]
            preannotations = final_predictions[final_predictions.image_path.isin(image_basenames)].copy(deep=True)

            # As a dict of lists
            preannotations = {image_path: group for image_path, group in preannotations.groupby("image_path")}
            self.annotator.upload(images=image_paths, instance_name=instance, preannotations=preannotations)

        self.comet_logger.experiment.add_tag("complete")
        return None
