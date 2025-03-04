from src.label_studio import gather_data
from torchmetrics.classification import Accuracy
from deepforest.evaluate import evaluate_boxes
from src.detection import predict, fix_taxonomy

from deepforest.utilities import read_file
import pandas as pd
import torch
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher

import os

class PipelineEvaluation:
    def __init__(self, model, crop_model, image_dir, detect_ground_truth_dir, classify_ground_truth_dir, comet_logger, detection_true_positive_threshold=0.85, classification_avg_score=0.5, patch_size=450, patch_overlap=0, min_score=0.5, debug=False, batch_size=16, detection_results=None):
        """Initialize pipeline evaluation.
        
        Args:
            model: Trained model for making predictions
            crop_model: Trained model for making crop predictions
            image_dir (str): Directory containing images
            detect_ground_truth_dir (str): Directory containing detection ground truth annotation CSV files
            classify_ground_truth_dir (str): Directory containing confident classification ground truth annotation CSV files
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            comet_logger: CometLogger object for logging
            classification_threshold (float): Threshold for classification confidence score
            patch_size (int): Size of image patches for prediction
            patch_overlap (int): Overlap between patches
            min_score (float): Minimum confidence score threshold for predictions
            batch_size (int): Batch size for prediction
            detection_results (dict): Dictionary containing detection results, optional
        """
        self.detection_true_positive_threshold = detection_true_positive_threshold 
        self.classification_avg_score = classification_avg_score
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_score = min_score
        self.detection_ground_truth_dir = detect_ground_truth_dir
        self.classify_ground_truth_dir = classify_ground_truth_dir
        self.image_dir = image_dir
        self.classification_model = crop_model
        self.model = model
        self.debug = debug
        self.batch_size = batch_size
        self.comet_logger = comet_logger

        # Gather data
        self.detection_annotations = gather_data(detect_ground_truth_dir)
        self.detection_annotations= fix_taxonomy(self.detection_annotations)

        self.detection_annotations = self.detection_annotations[self.detection_annotations.label.isin(self.model.label_dict.keys())]

        self.classification_annotations = gather_data(classify_ground_truth_dir)
        
        # There is one caveat for empty frames, assign a label which the dict contains
        self.classification_annotations.loc[self.classification_annotations.label.astype(str)=='0',"label"] = self.model.numeric_to_label_dict[0]

        # No need to evaluate if there are no annotations for classification
        self.classification_annotations = self.classification_annotations.loc[
            ~(
            (self.classification_annotations.xmin == 0) &
            (self.classification_annotations.ymin == 0) &
            (self.classification_annotations.xmax == 0) &
            (self.classification_annotations.ymax == 0)
            )
        ]

        # Prediction container
        self.predictions = []
        
        self.confident_predictions, self.uncertain_predictions = self.predict_classification()
        self.num_classes = len(self.classification_annotations["label"].unique())
            
        if self.num_classes == 1:
            self.num_classes = 2

        # Metrics
        self.confident_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)
        self.uncertain_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)

    def _format_targets(self, annotations_df):
        targets = {}
        
        if annotations_df.empty:
            targets["boxes"] = torch.tensor([])
            targets["labels"] = torch.tensor([])
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor([])
        elif (annotations_df.xmin == 0).all():
            # Torchmetrics expects empty tensors, see https://github.com/Lightning-AI/torchmetrics/pull/624/files
            targets["boxes"] = torch.tensor([])
            targets["labels"] = torch.tensor([])
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor([])
        else:
            targets["boxes"] = torch.tensor(annotations_df[["xmin", "ymin", "xmax","ymax"]].values.astype("float32"))
            targets["labels"] = torch.tensor(annotations_df["label"].astype(int).values)
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor(annotations_df["score"].tolist())

        return targets

    def predict_classification(self):
        full_image_paths = [self.image_dir + "/" + image_path for image_path in self.classification_annotations.drop_duplicates("image_path").image_path.tolist()] 
        # Add a debug to check performance
        if self.debug:
            full_image_paths = full_image_paths[:3]

        predictions = predict(
            m=self.model,
            crop_model=self.classification_model,
            image_paths=full_image_paths, 
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap, 
            batch_size=self.batch_size
        )
        combined_predictions = pd.concat(predictions)

        # Split into confident and uncertain based on average score
        average_score = combined_predictions.groupby("image_path").apply(lambda x: x["score"].mean())

        # Which images have a score above the average
        confident_images = average_score[average_score > self.classification_avg_score].index

        # Select the annotations for confident and uncertain
        confident_predictions = combined_predictions[combined_predictions.image_path.isin(confident_images)]
        uncertain_predictions = combined_predictions[~combined_predictions.image_path.isin(confident_images)]

        self.comet_logger.experiment.log_table("validation_confident_predictions", confident_predictions)
        self.comet_logger.experiment.log_table("validation_uncertain_predictions", uncertain_predictions)

        return confident_predictions, uncertain_predictions
    
    def match_predictions_and_targets(self, pred, target):
        """
        Matches predicted bounding boxes with source bounding boxes using Intersection over Union (IoU).
        Args:
            pred (Tensor): A tensor containing the source bounding boxes.
            target (Tensor): A tensor containing the predicted bounding boxes.
        Returns:
            DataFrame: A dataframe containing the matched predictions and targets.
        """
        
        # Match predictions and targets
        matcher = Matcher(
            0.3,
            0.3,
            allow_low_quality_matches=False)
        
        pred_boxes = pred["boxes"]
        src_boxes = target["boxes"]

        match_quality_matrix = box_iou(
            src_boxes,
            pred_boxes)
        
        results = matcher(match_quality_matrix)

        matched_pred = []
        matched_target = []

        for i, match in enumerate(results):
            if match >= 0:
                matched_pred.append(int(pred["labels"][i].item()))
                matched_target.append(int(target["labels"][match].item()))
            else:
                matched_pred.append(int(pred["labels"][i].item()))
                matched_target.append(None)

        matches = pd.DataFrame({"pred": matched_pred, "target": matched_target})

        # Remove the None values for predicted, can't get class scores if the box doesn't match
        matches = matches.dropna(subset=["target"])
        
        return matches
            
    def evaluate_confident_classification(self):
        """Evaluate confident classification performance"""
        return self._evaluate_classification(self.confident_predictions, self.confident_classification_accuracy)

    def evaluate_uncertain_classification(self):
        """Evaluate uncertain classification performance"""
        return self._evaluate_classification(self.uncertain_predictions, self.uncertain_classification_accuracy)

    def _evaluate_classification(self, predictions, accuracy_metric):
        """Helper function to evaluate classification performance.
        
        Args:
            predictions (DataFrame): DataFrame containing the predictions.
            accuracy_metric (torchmetrics.Metric): Metric to evaluate accuracy.
        
        Returns:
            dict: Dictionary containing the computed accuracy metric.
        """
        for image_path in predictions.drop_duplicates("image_path").image_path.tolist():
            image_targets = self.classification_annotations.loc[self.classification_annotations.image_path == os.path.basename(image_path)]
            image_predictions = predictions.loc[predictions.image_path == os.path.basename(image_path)]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            
            if image_predictions.empty:
                continue

            # Labels as numeric
            image_targets["label"] = image_targets.label.apply(lambda x: self.classification_model.label_dict[x])
            if not pd.api.types.is_numeric_dtype(image_targets["label"]):
                image_targets["label"] = image_targets["label"].apply(lambda x: self.classification_model.label_dict[x])
            if not pd.api.types.is_numeric_dtype(image_predictions["cropmodel_label"]):
                image_predictions["label"] = image_predictions["cropmodel_label"].apply(lambda x: self.classification_model.label_dict[x])
            else:
                image_predictions["label"] = image_predictions["cropmodel_label"]
            
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            if len(pred["labels"]) == 0:
                continue
            matches = self.match_predictions_and_targets(pred, target)
            self.predictions.append(matches)
            if matches.empty:
                continue
            accuracy_metric.update(preds=torch.tensor(matches["pred"].values), target=torch.tensor(matches["target"].values))
        
        # To do average score of true positive and false positive
        results = {f"{accuracy_metric.__class__.__name__.lower()}": accuracy_metric.compute(), 
                   "avg_score_true_positive": None, 
                   "avg_score_false_positive": None}
        
        self.comet_logger.experiment.log_metrics(results)

        return results
    
    def evaluate_detection(self):
        """Evaluate detection performance"""
        
        full_image_paths = [self.image_dir + "/" + image_path for image_path in self.detection_annotations.image_path.tolist()] 
        
        if self.debug:
            full_image_paths = full_image_paths[:3]
        
        predictions = predict(
            m=self.model,
            image_paths=full_image_paths, 
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap, 
        )
        
        combined_predictions = pd.concat(predictions)
        combined_predictions["workflow"] = "detection"
        self.predictions.append(combined_predictions)
        
        # Remove empty predictions, needs to be confirmed for edge cases
        combined_predictions = combined_predictions[~combined_predictions["score"].isna()]

        combined_predictions = read_file(combined_predictions, self.image_dir)
        ground_truth = self.detection_annotations
        if "geometry" not in ground_truth.columns:
            ground_truth = read_file(ground_truth, self.image_dir)

        iou_results = evaluate_boxes(
            combined_predictions,
            ground_truth,
            iou_threshold=self.detection_true_positive_threshold,
            root_dir=self.image_dir)
        
        non_empty_results = iou_results["results"][~iou_results["results"]["score"].isna()]
        if non_empty_results.empty:
            return {"recall": None, "precision": None, "avg_score_true_positive": None, "avg_score_false_positive": None}
        else:
            #Convert match to boolean
            non_empty_results["match"] = non_empty_results["match"].astype(bool)
            avg_score_true_positive = non_empty_results.loc[non_empty_results["match"]].score.mean()
            avg_score_false_positive = non_empty_results.loc[~non_empty_results["match"]].score.mean()
        
        results = {"recall": iou_results["box_recall"], "precision": iou_results["box_precision"], "avg_score_true_positive": avg_score_true_positive, "avg_score_false_positive": avg_score_false_positive}

        with self.comet_logger.experiment.context_manager("detection"):
            self.comet_logger.experiment.log_metrics(results)

        return results

    def evaluate(self):
        """
        Evaluate pipeline performance for both detection and classification
            
        """
        self.results = {}
        detection_results = self.evaluate_detection()

        if self.classification_annotations.empty:
            results = {"confident_classification_accuracy": None, "uncertain_classification_accuracy": None}
            return results
        else:
            confident_classification_results = self.evaluate_confident_classification()
            uncertain_classification_results = self.evaluate_uncertain_classification()

        self.results = {"detection": detection_results, "confident_classification":confident_classification_results, "uncertain_classification":uncertain_classification_results}
        
        return self.results
    
    def check_success(self):
        """Check if pipeline performance is satisfactory"""
        # For each metric, check if it is above the threshold
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No results found, run evaluate() first")

        if self.results['detection']["recall"] > self.detection_true_positive_threshold:
            return True
        else:
            return False