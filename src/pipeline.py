import comet_ml
import os
import shutil
from omegaconf import DictConfig

from src.active_learning import (
    generate_pool_predictions,
    human_review,
    load_species_to_genus_from_csv,
    select_images,
)
from deepforest.model import CropModel
from src import sagemaker_gt
from src.annotators import get_annotator, LabelStudioAnnotator, SageMakerAnnotator
from src import label_studio as ls_mod
from src import detection
from src import classification
from src import hierarchical
from src.usgs_annotations import load_usgs_annotated_image_paths
from src.visualization import crop_images, convert_codec
from src.report import generate_report, load_geospatial_metadata, georeference_predictions
from src import bulk_annotations as bulk_mod
from src.pipeline_evaluation import PipelineEvaluation
from src.spatiotemporal_metadata import metadata_lookup_for_images
from pytorch_lightning.loggers import CometLogger
import glob
import pandas as pd
import random
import tempfile

_IMAGE_EXTS = ("jpg", "JPG", "jpeg", "JPEG", "tif", "TIF", "tiff", "TIFF")


def _collect_images(image_dir: str) -> list[str]:
    return [p for ext in _IMAGE_EXTS for p in glob.glob(os.path.join(image_dir, f"*.{ext}"))]


class Pipeline:
    """Pipeline for training and evaluating a detection and classification model"""
    def __init__(self, cfg: DictConfig):
        """Initialize the pipeline with optional configuration"""
        self.config = cfg

        # Generic annotator
        self.annotator = get_annotator(self.config)

        # Pool of all images
        self.all_images = _collect_images(self.config.image_dir)

        self.comet_logger = CometLogger(project_name=self.config.comet.project, workspace=self.config.comet.workspace)
        self.comet_logger.experiment.add_tag("pipeline")
        flight_name = self._flight_name()
        self.comet_logger.experiment.add_tag(flight_name)
        self.comet_logger.experiment.log_parameters(self.config)
        self.comet_logger.experiment.log_parameter("flight_name", flight_name)
        self.config.detection_model.crop_image_dir = os.path.join(self.config.detection_model.crop_image_dir, flight_name)
        self.config.detection_model.checkpoint_dir = os.path.join(self.config.detection_model.checkpoint_dir, flight_name)
        self.config.classification_model.checkpoint_dir = os.path.join(self.config.classification_model.checkpoint_dir, flight_name)
        self.config.classification_model.train_crop_image_dir = os.path.join(self.config.classification_model.train_crop_image_dir, flight_name)
        self.config.classification_model.val_crop_image_dir = os.path.join(self.config.classification_model.val_crop_image_dir, flight_name)
        os.makedirs(self.config.detection_model.crop_image_dir, exist_ok=True)
        os.makedirs(self.config.detection_model.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.classification_model.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.classification_model.train_crop_image_dir, exist_ok=True)
        os.makedirs(self.config.classification_model.val_crop_image_dir, exist_ok=True)
        self.comet_logger.experiment.log_code(folder=os.path.join(os.path.dirname(__file__), "../src"), overwrite=True)

    def _comet_experiment_id(self) -> str:
        """Return Comet experiment id as string (handles both property and method API)."""
        eid = self.comet_logger.experiment.id
        return eid() if callable(eid) else eid

    def _flight_name(self):
        """Name used for crop/checkpoint dirs, caches and metadata lookup.

        Defaults to the image_dir basename. Datasets whose leaf directories repeat
        across flights (neaq: every date has a "Belly camera edited") must override
        this, or all dates collide into one namespace and overwrite each other.
        """
        override = getattr(self.config, "flight_name", None)
        return str(override) if override else os.path.basename(self.config.image_dir)

    def _classification_metadata_dir(self):
        cls_metadata_dir = getattr(self.config.classification_model, "metadata_dir", None)
        if cls_metadata_dir:
            return cls_metadata_dir
        report_cfg = getattr(self.config, "report", None)
        return getattr(report_cfg, "metadata_dir", None) if report_cfg else None

    def _use_classification_metadata(self) -> bool:
        return bool(getattr(self.config.classification_model, "use_metadata", False))

    def _metadata_lookup_for_pool(self, pool):
        if not self._use_classification_metadata():
            return None
        metadata_dir = self._classification_metadata_dir()
        if not metadata_dir:
            raise ValueError(
                "classification_model.use_metadata=True requires report.metadata_dir "
                "or classification_model.metadata_dir"
            )
        lookup = metadata_lookup_for_images(
            image_paths=pool,
            metadata_dir=metadata_dir,
            default_flight_name=self._flight_name(),
        )
        if pool and not lookup:
            raise ValueError(
                "classification_model.use_metadata=True but no image metadata "
                f"matched flight {self._flight_name()}"
            )
        return lookup

    def check_new_annotations(self, instance_name):
        return self.annotator.check_for_new_annotations(instance_name, image_dir=self.config.image_dir)
    
    def check_annotations(self):

        if self.config.check_annotations:
            self.check_new_annotations("train")
            self.check_new_annotations("validation")
            self.check_new_annotations("review")

        if isinstance(self.annotator, SageMakerAnnotator):
            train_dir = str(self.config.annotation.sagemaker.instances.train.csv_dir)
            annotations_base = os.path.dirname(os.path.dirname(train_dir))
            sagemaker_root = os.path.join(annotations_base, "sagemaker")
            sagemaker_gt.normalize_sagemaker_output_to_annotation_dirs(
                sagemaker_output_dir=os.path.join(sagemaker_root, "output"),
                sagemaker_input_dir=os.path.join(sagemaker_root, "input"),
                annotations_base_dir=annotations_base,
            )

        self.existing_training = self.annotator.gather_data("train", image_dir=self.config.image_dir)
        self.existing_validation = self.annotator.gather_data("validation", image_dir=self.config.image_dir)
        self.existing_reviewed = self.annotator.gather_data("review", image_dir=self.config.image_dir)

        # Apply Serenity bulk annotator overrides (and optionally add new rows from predictions manifest)
        server_cfg = getattr(self.config, "server", None)
        bulk_path = getattr(server_cfg, "bulk_annotations_path", None) if server_cfg is not None else None
        if server_cfg is not None and bulk_path:
            bulk_df = bulk_mod.fetch_bulk_annotations_csv(server_cfg, bulk_path)
            if bulk_df is not None and not bulk_df.empty:
                bulk_reduced = bulk_mod.reduce_bulk_to_latest(bulk_df)
                predictions_path = getattr(server_cfg, "bulk_predictions_path", None)
                predictions_df = bulk_mod.fetch_predictions_csv(server_cfg, predictions_path) if predictions_path else None
                train, val, review, n_overrides, n_added = bulk_mod.apply_bulk_overrides(
                    self.existing_training,
                    self.existing_validation,
                    self.existing_reviewed,
                    bulk_reduced,
                    self.config.image_dir,
                    predictions_df=predictions_df,
                )
                self.existing_training = train
                self.existing_validation = val
                self.existing_reviewed = review
                print(f"Applied {n_overrides} bulk annotation overrides from Serenity.")
                if n_added > 0:
                    print(f"Added {n_added} new annotation rows from bulk (predictions manifest).")

        self.comet_logger.experiment.log_table(tabular_data=self.existing_reviewed, filename="human_reviewed_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_training, filename="training_annotations.csv")
        self.comet_logger.experiment.log_table(tabular_data=self.existing_validation, filename="validation_annotations.csv")
        
        # If a brand new folder, there are no annotations. Continue in inference-only mode:
        # run existing model on full pool and review imagery logic; skip training and eval.
        if self.existing_training is None and self.existing_validation is None and self.existing_reviewed is None:
            self.existing_images = []
            print("No existing annotations; running inference-only (no training or eval)")
            return True

        else:
            if self.existing_training is not None:
                print(f"Training annotations shape: {self.existing_training.shape}")
            if self.existing_validation is not None:
                print(f"Validation annotations shape: {self.existing_validation.shape}")
            if self.existing_reviewed is not None:
                print(f"Reviewed annotations shape: {self.existing_reviewed.shape}")

        # deepforest.read_file keeps image_path as basenames (root_dir stored as metadata).
        # Normalize everything to basenames so pool/flightline comparisons work consistently.
        all_paths = (
            (self.existing_training.image_path.tolist() if self.existing_training is not None else []) +
            (self.existing_validation.image_path.tolist() if self.existing_validation is not None else []) +
            (self.existing_reviewed.image_path.tolist() if self.existing_reviewed is not None else [])
        )
        self.existing_images = list(set(os.path.basename(p) for p in all_paths))
        # Include USGS/UBFAI annotations from crops/train.csv and test.csv so we don't re-review
        usgs_train_paths, usgs_test_paths = load_usgs_annotated_image_paths(self.config.image_dir)
        if usgs_train_paths or usgs_test_paths:
            print(f"Found {len(usgs_train_paths)} USGS/UBFAI train paths and {len(usgs_test_paths)} USGS/UBFAI test paths")
            self.existing_images = list(set(self.existing_images + [os.path.basename(p) for p in usgs_train_paths + usgs_test_paths]))

        return True

    def upload_full_flight(self, project_name=None, skip_annotated=False):
        """Upload all flight detections to a dedicated Label Studio project.

        Combines existing annotations (train/eval/review) with model predictions
        for unannotated images. All images that have detections or annotations are
        uploaded. Intended for creating a clean annotated flight video.

        Args:
            project_name: Label Studio project name override.
                Defaults to "BOEM - Full Flight - {flight_name}".
            skip_annotated: If True, exclude already-annotated images from the
                upload entirely. Only unannotated images with model predictions
                are uploaded.
        """
        if not isinstance(self.annotator, LabelStudioAnnotator):
            raise ValueError("upload_full_flight requires a LabelStudioAnnotator")

        self.check_annotations()

        flight_name = self._flight_name()
        if project_name is None:
            project_name = f"BOEM - Full Flight - {flight_name}"

        # Run predictions on images not yet annotated (existing_images uses basenames)
        unannotated = [img for img in self.all_images if os.path.basename(img) not in self.existing_images]
        full_flight_cache = os.path.join(self.config.image_dir, ".full_flight_predictions.csv")
        pool_predictions = None
        if unannotated:
            # Only the full-flight cache is valid here. The active-learning pool cache
            # covers a random pool_limit-sized sample, not the flight.
            if os.path.isfile(full_flight_cache):
                print(f"Loading cached full-flight predictions from {full_flight_cache}")
                pool_predictions = pd.read_csv(full_flight_cache)
            else:
                # No cache — load models and run predictions
                detection_model = detection.load(checkpoint=self.config.detection_model.checkpoint)
                classification_model = CropModel.load_from_checkpoint(
                    self.config.classification_model.checkpoint
                )
                classification_model.config["cropmodel"]["expand"] = self.config.classification_model.expand

                hcast_checkpoint = getattr(self.config.hierarchical, "checkpoint", None)
                hcast_label_csv = getattr(self.config.hierarchical, "label_csv", None)
                hcast_model = None
                if hcast_checkpoint:
                    if not hcast_label_csv:
                        raise ValueError("hierarchical.label_csv is required when hierarchical.checkpoint is set")
                    hcast_model = hierarchical.load_hcast_model(
                        checkpoint_path=hcast_checkpoint,
                        label_csv=hcast_label_csv,
                    )
                hcast_batch_size = getattr(self.config.hierarchical, "batch_size", 16)
                hcast_workers = getattr(self.config.hierarchical, "workers", 4)

                print(f"Running predictions on {len(unannotated)} unannotated images")
                pool_predictions = generate_pool_predictions(
                    pool=unannotated,
                    pool_limit=None,
                    patch_size=self.config.active_learning.patch_size,
                    patch_overlap=self.config.active_learning.patch_overlap,
                    min_score=self.config.predict.min_score,
                    model=detection_model,
                    batch_size=self.config.predict.batch_size,
                    crop_model=classification_model,
                    hcast_model=hcast_model,
                    image_dir=self.config.image_dir,
                    hcast_batch_size=hcast_batch_size,
                    hcast_workers=hcast_workers,
                    workers=self.config.predict.workers,
                    metadata_lookup=self._metadata_lookup_for_pool(unannotated),
                )
                if pool_predictions is not None and not pool_predictions.empty:
                    pool_predictions.to_csv(full_flight_cache, index=False)
                    print(f"Cached full-flight predictions to {full_flight_cache}")

        # Build per-image preannotation dict, starting with human annotations
        preannotations = {}
        images_with_content = set()

        ann_sources = (
            []
            if skip_annotated
            else [self.existing_training, self.existing_validation, self.existing_reviewed]
        )
        for ann_df in ann_sources:
            if ann_df is None or ann_df.empty:
                continue
            ann = ann_df.copy()
            if "label" in ann.columns:
                ann = ann[ann["label"] != "FalsePositive"]
            if ann.empty:
                continue
            ann.rename(columns={"label": "cropmodel_label"}, inplace=True)
            ann["label"] = "Object"
            ann["score"] = 2.0
            ann["cropmodel_score"] = 2.0
            if "comet_id" not in ann.columns:
                ann["comet_id"] = "human_annotation"
            for img_path, group in ann.groupby("image_path"):
                basename = os.path.basename(img_path)
                if basename in preannotations:
                    preannotations[basename] = pd.concat(
                        [preannotations[basename], group], ignore_index=True
                    )
                else:
                    preannotations[basename] = group
                images_with_content.add(os.path.join(self.config.image_dir, basename))

        # Fill in model predictions for unannotated images (never overwrite human labels)
        if pool_predictions is not None and not pool_predictions.empty:
            pool_predictions["comet_id"] = self._comet_experiment_id()
            for img_path, group in pool_predictions.groupby("image_path"):
                basename = os.path.basename(img_path)
                if basename not in preannotations:
                    preannotations[basename] = group
                images_with_content.add(os.path.join(self.config.image_dir, basename))

        images_to_upload = sorted(images_with_content)
        print(
            f"Uploading {len(images_to_upload)} images to Label Studio project '{project_name}' "
            f"({len(preannotations)} with pre-annotations)"
        )

        ls_cfg = self.config.annotation.label_studio
        ls_mod.upload_to_label_studio(
            images=images_to_upload,
            sftp_client=self.annotator.sftp_client,
            url=ls_cfg.url,
            project_name=project_name,
            images_to_annotate_dir=self.config.image_dir,
            folder_name=ls_cfg.folder_name,
            preannotations=preannotations,
        )

        self.comet_logger.experiment.log_parameter("full_flight_project", project_name)
        self.comet_logger.experiment.log_parameter("full_flight_n_images", len(images_to_upload))
        print(f"Full flight upload complete: {len(images_to_upload)} images uploaded.")

    def run(self):
        # Check for new annotations if the check_annotations flag is set
        status = self.check_annotations()
        
        if not status:
            return None

        # existing_images is always a list after check_annotations ([] when no annotations)
        if len(self.existing_images) == 0:
            self.config.detection_model.force_train = False
            print("No existing annotations, turning off force training")

        parts = [x for x in [self.existing_training, self.existing_reviewed] if x is not None]
        all_training = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        
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
            actual_detection_checkpoint = os.path.join(
                self.config.detection_model.checkpoint_dir,
                self._comet_experiment_id() + ".ckpt",
            )
        else:
            trained_detection_model = detection.load(checkpoint=self.config.detection_model.checkpoint)
            actual_detection_checkpoint = self.config.detection_model.checkpoint
            self.comet_logger.experiment.log_parameter("detection_checkpoint_path", self.config.detection_model.checkpoint)

        classification_backend = getattr(self.config.classification_model, "backend", "deepforest")

        # Create crop image directories if they don't exist
        train_crop_image_dir = os.path.join(self.config.classification_model.train_crop_image_dir, self.comet_logger.experiment.id)
        os.makedirs(train_crop_image_dir, exist_ok=True)

        val_crop_image_dir = os.path.join(self.config.classification_model.val_crop_image_dir, self.comet_logger.experiment.id)
        os.makedirs(val_crop_image_dir, exist_ok=True)

        if classification_backend == "deepforest":
            # If there are no train annotations, turn off force training
            if all_training.empty or (all_training.xmin[all_training.xmin != 0].empty):
                self.config.classification_model.force_train = False
                print("No training annotations, turning off force training")

            # If there are no validation annotations, turn off force training (eval skipped later)
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
                    expand_pixels=int(getattr(self.config.classification_model, "expand", 30)),
                    fast_dev_run=self.config.classification_model.fast_dev_run,
                    max_epochs=self.config.classification_model.max_epochs,
                    lr=self.config.classification_model.lr,
                    batch_size=self.config.classification_model.batch_size,
                    workers=self.config.classification_model.workers,
                    comet_logger=self.comet_logger,
                    balance_classes=getattr(self.config.classification_model, "balance_classes", False),
                    use_metadata=self._use_classification_metadata(),
                    metadata_dir=self._classification_metadata_dir(),
                    flight_name=self._flight_name(),
                    metadata_dim=getattr(self.config.classification_model, "metadata_dim", 32),
                    metadata_dropout=getattr(self.config.classification_model, "metadata_dropout", 0.5))
            else:
                trained_classification_model = CropModel.load_from_checkpoint(self.config.classification_model.checkpoint)
        else:
            raise NotImplementedError("Only deepforest classification backend is currently implemented")

        pool = _collect_images(self.config.image_dir)
        pool = [image for image in pool if os.path.basename(image) not in self.existing_images]
        pool_limit = getattr(self.config.active_learning, "pool_limit", None)
        print(f"Pool: {len(pool)} images (after excluding existing), pool_limit={pool_limit}")

        if self.config.debug:
            if self.existing_validation is not None:
                non_empty_validation = self.existing_validation[~(self.existing_validation.xmin==0)]
                pool = list(non_empty_validation.image_path.unique())
                pool = [os.path.join(self.config.image_dir, image) for image in pool][:10]
            else:
                pool = random.sample(pool, min(10, len(pool))) if pool else []

        # Prediction cache: reuse detection results when only classification model changes.
        # Cache is invalidated when detection checkpoint changes.
        cache_dir = os.path.join(self.config.image_dir, ".prediction_cache")
        prediction_pool = pool
        use_cached_pool = False
        if os.path.isdir(cache_dir):
            ckpt_file = os.path.join(cache_dir, "detection_checkpoint.txt")
            pred_file = os.path.join(cache_dir, "pool_predictions.csv")
            min_score_file = os.path.join(cache_dir, "min_score.txt")
            if os.path.isfile(ckpt_file) and os.path.isfile(pred_file):
                with open(ckpt_file) as f:
                    cached_ckpt = f.read().strip()
                # The cache stores only boxes >= the min_score it was written at, so it is
                # only valid when that threshold is <= the current one (cache is a superset).
                # A lower current min_score (or an unmarked/old cache) must force re-prediction.
                try:
                    with open(min_score_file) as f:
                        cached_min_score = float(f.read().strip())
                except (OSError, ValueError):
                    cached_min_score = None
                min_score_ok = (
                    cached_min_score is not None
                    and cached_min_score <= self.config.predict.min_score
                )
                if cached_ckpt == actual_detection_checkpoint and min_score_ok:
                    cached_df = pd.read_csv(pred_file)
                    if "score" in cached_df.columns:
                        min_score = self.config.predict.min_score
                        cached_above = cached_df[cached_df["score"] >= min_score]
                        cached_paths = set(cached_above["image_path"].astype(str).unique())
                        cached_basenames = {os.path.basename(p) for p in cached_paths}
                        prediction_pool = [
                            p for p in pool
                            if p in cached_paths or os.path.basename(p) in cached_basenames
                        ]
                        if prediction_pool:
                            use_cached_pool = True
                            print(
                                f"Using prediction cache (same detection checkpoint): "
                                f"running detection+classification on {len(prediction_pool)} images "
                                f"(previously had ≥{min_score} detections) instead of {len(pool)}"
                            )
        if not use_cached_pool:
            # The cache block above narrows prediction_pool to images that previously scored
            # >=min_score. When that set is empty (e.g. a flight where the last run found
            # nothing) we must fall back to the full pool, or we would predict on 0 images.
            prediction_pool = pool
            if os.path.isdir(cache_dir):
                print("Prediction cache skipped (new or different detection checkpoint, or no cached predictions)")

        trained_classification_model.config["cropmodel"]["expand"] = self.config.classification_model.expand

        hcast_checkpoint = getattr(self.config.hierarchical, "checkpoint", None)
        hcast_label_csv = getattr(self.config.hierarchical, "label_csv", None)
        hcast_model = None
        if hcast_checkpoint:
            if not hcast_label_csv:
                raise ValueError("hierarchical.label_csv is required when hierarchical.checkpoint is set")
            hcast_model = hierarchical.load_hcast_model(
                checkpoint_path=hcast_checkpoint,
                label_csv=hcast_label_csv,
            )
        hcast_batch_size = getattr(self.config.hierarchical, "batch_size", 16)
        hcast_workers = getattr(self.config.hierarchical, "workers", 4)
        species_to_genus = load_species_to_genus_from_csv(hcast_label_csv)

        # Apply pool_limit: when using cache, still honor config so debug/small runs stay fast
        configured_pool_limit = getattr(self.config.active_learning, "pool_limit", None)
        if use_cached_pool and configured_pool_limit is not None:
            pool_limit = min(configured_pool_limit, len(prediction_pool))
        elif use_cached_pool:
            pool_limit = len(prediction_pool)
        else:
            pool_limit = configured_pool_limit
        flightline_predictions = generate_pool_predictions(
            pool=prediction_pool,
            pool_limit=pool_limit,
            patch_size=self.config.active_learning.patch_size,
            patch_overlap=self.config.active_learning.patch_overlap,
            min_score=self.config.predict.min_score,
            model=trained_detection_model,
            batch_size=self.config.predict.batch_size,
            crop_model=trained_classification_model,
            hcast_model=hcast_model,
            image_dir=self.config.image_dir,
            hcast_batch_size=hcast_batch_size,
            hcast_workers=hcast_workers,
            workers=self.config.predict.workers,
            metadata_lookup=self._metadata_lookup_for_pool(prediction_pool),
        )

        if flightline_predictions is None:
            print("No predictions")
            return None

        n_images_with_detections = flightline_predictions["image_path"].nunique()
        n_detections = len(flightline_predictions)
        print(f"Pool: {n_images_with_detections} images with ≥{self.config.predict.min_score} detections ({n_detections} total boxes)")

        # Save prediction cache for this flight (detection checkpoint + pool predictions CSV)
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "detection_checkpoint.txt"), "w") as f:
            f.write(actual_detection_checkpoint)
        with open(os.path.join(cache_dir, "min_score.txt"), "w") as f:
            f.write(str(self.config.predict.min_score))
        flightline_predictions.to_csv(os.path.join(cache_dir, "pool_predictions.csv"), index=False)

        flightline_predictions["comet_id"] = self.comet_logger.experiment.id

        if self.existing_validation is None:
            print("No validation annotations, skipping evaluation")       
        else:
            evaluation_annotations = self.existing_validation.copy(deep=True)
            evaluation_predictions = flightline_predictions[flightline_predictions.image_path.isin(self.existing_validation.image_path)]


            label_dict = trained_classification_model.label_dict
            pipeline_monitor = PipelineEvaluation(  # species_to_genus added for hierarchical metrics
                predictions=evaluation_predictions,
                annotations=evaluation_annotations,
                classification_label_dict=label_dict,
                species_to_genus=species_to_genus,
                **self.config.pipeline_evaluation)

            performance = pipeline_monitor.evaluate()
            self.comet_logger.experiment.log_metrics(performance)

            if pipeline_monitor.check_success():
                print("Pipeline performance is satisfactory, exiting")
                return None

        test_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images)]
        test_images_to_annotate, preannotations, _al_stats_test = select_images(
            preannotations=test_preannotations,
            strategy="random",
           n=self.config.active_testing.n_images,
        )
        if len(test_images_to_annotate) < self.config.active_testing.n_images and len(test_images_to_annotate) > 0:
            print(f"Test: requested {self.config.active_testing.n_images} images but only {flightline_predictions['image_path'].nunique()} images have any detection ≥min_score; using all {len(test_images_to_annotate)}")

        print(f"Test images to annotate: {len(test_images_to_annotate)}")

        # Select images to annotate based on the strategy
        training_preannotations = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate)]
        
        # Default taxonomy path: project root transformed_taxonomy.json (for strategy "taxonomy")
        _default_taxonomy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transformed_taxonomy.json")
        disagreement_target_labels = getattr(self.config.active_learning, "disagreement_target_labels", None)
        if disagreement_target_labels is not None and len(disagreement_target_labels) == 0:
            disagreement_target_labels = None
        train_images_to_annotate, preannotations, al_stats_train = select_images(
            preannotations=training_preannotations,
            strategy=self.config.active_learning.strategy,
            n=self.config.active_learning.n_images,
            min_score=self.config.predict.min_score,
            target_labels=getattr(self.config.active_learning, "target_labels", None),
            drop_n_most_common=getattr(self.config.active_learning, "drop_n_most_common", 1),
            rarest_confidence_selection=getattr(self.config.active_learning, "rarest_confidence_selection", "lowest"),
            min_classification_score=getattr(self.config.active_learning, "min_classification_score", None),
            taxonomy_path=getattr(self.config.active_learning, "taxonomy_path", None) or _default_taxonomy_path,
            taxonomy_aliases=getattr(self.config.active_learning, "taxonomy_aliases", None),
            valid_labels=list(trained_classification_model.label_dict.keys()),
            ensemble_target_mode=getattr(self.config.active_learning, "ensemble_target_mode", "crop_only"),
            species_to_genus=species_to_genus if species_to_genus else None,
            disagreement_require_genus_mismatch=getattr(
                self.config.active_learning, "disagreement_require_genus_mismatch", False
            ),
            disagreement_target_labels=disagreement_target_labels,
        )
        if al_stats_train:
            numeric_metrics = {
                k: float(v) if isinstance(v, bool) else v
                for k, v in al_stats_train.items()
                if isinstance(v, (int, float))
            }
            if numeric_metrics:
                self.comet_logger.experiment.log_metrics(numeric_metrics)
            mode = al_stats_train.get("al_ensemble_target_mode")
            if mode is not None:
                self.comet_logger.experiment.log_parameter("al_ensemble_target_mode", mode)
        if len(train_images_to_annotate) == 0 and training_preannotations.empty:
            print("Training images to annotate: 0 (all images with detections ≥min_score were assigned to test)")
        
        print(f"Training images to annotate: {len(train_images_to_annotate)}")
        human_review_pool = flightline_predictions[~flightline_predictions.image_path.isin(self.existing_images + test_images_to_annotate + train_images_to_annotate)]
        
        confident_predictions, uncertain_predictions = human_review(
            min_detection_score=getattr(
                self.config.human_review, "min_detection_score", self.config.predict.min_score),
            review_low=getattr(self.config.human_review, "review_low", 0.3),
            review_high=getattr(self.config.human_review, "review_high", 0.6),
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
        final_predictions["flight_name"] = self._flight_name()

        # Add in the existing training and validation annotations
        if self.existing_training is not None:
            existing_training = self.existing_training.copy(deep=True)
            existing_training["set"] = "train"
            # Change label to cropmodel_label
            existing_training.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a cropmodel_score column
            existing_training["cropmodel_score"] = 2.0
            # Add an object score
            existing_training["label"] = "Object"
            existing_training["score"] = 2.0

            if self.config.debug:
                existing_training = existing_training[:10]

            final_predictions = pd.concat([final_predictions, existing_training], ignore_index=True)

        if self.existing_validation is not None:
            existing_validation = self.existing_validation.copy(deep=True)
            existing_validation["set"] = "validation"
            # Change label to cropmodel_label
            existing_validation.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a cropmodel_score column
            existing_validation["cropmodel_score"] = 2.0

            # Add an object score
            existing_validation["label"] = "Object"
            existing_validation["score"] = 2.0

            if self.config.debug:
                existing_validation = existing_validation[:10]
            final_predictions = pd.concat([final_predictions, existing_validation], ignore_index=True)

        # Add in the human reviewed annotations
        if self.existing_reviewed is not None:
            existing_reviewed = self.existing_reviewed.copy(deep=True)
            # Change label to cropmodel_label
            existing_reviewed.rename(columns={"label": "cropmodel_label"}, inplace=True)
            # Add a set column
            existing_reviewed["cropmodel_score"] = 2.0
            
            #Set detection column
            existing_reviewed["label"] = "Object"
            # Add an object score
            existing_reviewed["score"] = 2.0
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

        # Add geospatial coordinates so the dashboard has lat/lon per detection
        report_cfg = getattr(self.config, "report", None)
        metadata_dir = getattr(report_cfg, "metadata_dir", None) if report_cfg else None
        if metadata_dir:
            try:
                captures, img_w, img_h = load_geospatial_metadata(
                    self._flight_name(), metadata_dir,
                )
                final_predictions = georeference_predictions(
                    final_predictions, captures, img_w, img_h,
                )
            except FileNotFoundError:
                print("Could not load geospatial metadata for final_predictions")

        # Write crops to disk
        image_paths = crop_images(final_predictions, root_dir=self.config.image_dir, experiment=self.comet_logger.experiment, expand=self.config.predict.buffer)

        # crop_image_id
        final_predictions["crop_image_id"] = image_paths
        
        # give it a complete tag
        self.comet_logger.experiment.log_table(tabular_data=final_predictions, filename="final_predictions.csv")

        # Generate report
        if getattr(self.config, "report", None) and getattr(self.config.report, "enabled", False):
            generate_report(
                predictions=final_predictions,
                config=self.config,
                comet_logger=self.comet_logger,
                image_dir=self.config.image_dir,
            )

        # Flythrough video (uses cached pool_predictions) and upload to Comet
        flythrough_cfg = getattr(self.config, "flythrough_video", None)
        if flythrough_cfg and getattr(flythrough_cfg, "enabled", False):
            try:
                import importlib.util
                script_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "scripts", "flight_flythrough_video.py",
                )
                if os.path.isfile(script_path):
                    spec = importlib.util.spec_from_file_location(
                        "flight_flythrough_video", script_path,
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    output_dir = getattr(flythrough_cfg, "output_dir", None)
                    video_path = mod.generate_flythrough(
                        flight_dir=self.config.image_dir,
                        output_dir=output_dir,
                        predictions=final_predictions,
                    )
                    if video_path and os.path.isfile(video_path):
                        # Re-encode to H.264 for Streamlit/browser playback (script writes mp4v).
                        # Requires ffmpeg (e.g. module load ffmpeg); raises if not H.264.
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                            tmp_path = f.name
                        try:
                            convert_codec(video_path, tmp_path)
                            # Use shutil.move so cross-device (e.g. /scratch -> /blue) works
                            shutil.move(tmp_path, video_path)
                        except Exception:
                            if os.path.isfile(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass
                            raise
                        self.comet_logger.experiment.log_asset(
                            file_data=video_path,
                            file_name=os.path.basename(video_path),
                        )
                        print(f"Logged flythrough video to Comet: {video_path}")
                    else:
                        print(f"Flythrough video not produced: {video_path}")
                else:
                    print(f"Flythrough script not found: {script_path}")
            except Exception as e:
                print(f"Flythrough video failed (non-fatal): {e}")

        # Use generic annotator to upload for each instance
        for instance, image_basenames in {"train": train_images_to_annotate, "validation": test_images_to_annotate, "review": review_images_to_annotate}.items():
            if len(image_basenames) == 0:
                print(f"No images to upload for instance {instance}, skipping")
                continue

            image_paths = [os.path.join(self.config.image_dir, x) for x in image_basenames]
            # Normalize image_path in final_predictions to basenames for comparison
            final_predictions_normalized = final_predictions.copy()
            final_predictions_normalized["image_path_basename"] = final_predictions_normalized["image_path"].apply(
                lambda x: os.path.basename(x) if os.path.sep in str(x) else x
            )
            preannotations_df = final_predictions_normalized[
                final_predictions_normalized["image_path_basename"].isin(image_basenames)
            ].copy(deep=True)

            # As a dict with basename as key (matching what SageMaker upload expects)
            preannotations = {}
            for basename, group in preannotations_df.groupby("image_path_basename"):
                # Ensure image_path column uses basename for manifest writing
                group = group.copy()
                group["image_path"] = basename
                # Use basename as key to match the format expected by SageMaker upload
                preannotations[basename] = group.drop(columns=["image_path_basename"], errors="ignore")
            self.annotator.upload(
                images=image_paths,
                instance_name=instance,
                preannotations=preannotations,
                species_to_genus=species_to_genus if species_to_genus else None,
            )

        self.comet_logger.experiment.add_tag("complete")
        return None