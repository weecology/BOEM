from torchmetrics.classification import Accuracy
from deepforest.evaluate import evaluate_boxes

from deepforest.utilities import read_file
import pandas as pd
import torch
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
import geopandas as gpd

import os

class PipelineEvaluation:
    def __init__(self, predictions, detection_annotations, classification_annotations, detection_true_positive_threshold=0.85, classification_threshold=0.5):
        """Initialize pipeline evaluation.
        
        Args:
            detect_ground_truth_dir (str): Directory containing detection ground truth annotation CSV files
            predictions: DataFrame containing the predictions
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            classification_threshold (float): Threshold for classification confidence score

        """
        self.detection_true_positive_threshold = detection_true_positive_threshold 
        self.classification_threshold = classification_threshold 
        self.predictions = predictions
        self.detection_annotations = detection_annotations
        self.classification_annotations = classification_annotations
      
        self.confident_predictions, self.uncertain_predictions = self.split_predictions()
        self.num_classes = len(self.classification_annotations["label"].unique())
            
        if self.num_classes == 1:
            self.num_classes = 2

        # Metrics
        self.confident_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)
        self.uncertain_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)

    def _format_targets(self, annotations_df, label_column="cropmodel_label"):
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
            targets["labels"] = torch.tensor(annotations_df[label_column].astype(int).values)
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor(annotations_df["score"].tolist())

        return targets

    def split_predictions(self):
        # Split into confident and uncertain based on classification threshold
        confident_predictions = self.predictions[self.predictions.cropmodel_score > self.classification_threshold]
        uncertain_predictions = self.predictions[self.predictions.cropmodel_score <= self.classification_threshold]

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
            if image_predictions.empty:
                continue
            
            target = self._format_targets(image_targets, label_column="label")
            pred = self._format_targets(image_predictions, label_column="cropmodel_label")
            if len(target["labels"]) == 0:
                continue
            matches = self.match_predictions_and_targets(pred, target)
            if matches.empty:
                continue
            accuracy_metric.update(preds=torch.tensor(matches["pred"].values), target=torch.tensor(matches["target"].values))
        
        # To do average score of true positive and false positive
        results = {f"{accuracy_metric.__class__.__name__.lower()}": accuracy_metric.compute(), 
                   "avg_score_true_positive": None, 
                   "avg_score_false_positive": None}
        
        return results
    
    def evaluate_detection(self):
        """Evaluate detection performance"""
        detection_predictions = self.predictions[self.predictions.image_path.isin(self.detection_annotations.image_path)]
        combined_predictions = gpd.GeoDataFrame(detection_predictions)

        # When you concat geodataframes, you get pandas dataframes
        combined_predictions = combined_predictions[~combined_predictions["score"].isna()]

        # Check if geometry is string or polygon
        combined_predictions = combined_predictions
        ground_truth = self.detection_annotations

        iou_results = evaluate_boxes(
            combined_predictions,
            ground_truth,
            iou_threshold=self.detection_true_positive_threshold,
            root_dir=None)
        
        non_empty_results = iou_results["results"][~iou_results["results"]["score"].isna()]
        if non_empty_results.empty:
            return {"recall": None, "precision": None, "avg_score_true_positive": None, "avg_score_false_positive": None}
        else:
            #Convert match to boolean
            non_empty_results["match"] = non_empty_results["match"].astype(bool)
            avg_score_true_positive = non_empty_results.loc[non_empty_results["match"]].score.mean()
            avg_score_false_positive = non_empty_results.loc[~non_empty_results["match"]].score.mean()
        
        results = {"recall": iou_results["box_recall"], "precision": iou_results["box_precision"], "avg_score_true_positive": avg_score_true_positive, "avg_score_false_positive": avg_score_false_positive}

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