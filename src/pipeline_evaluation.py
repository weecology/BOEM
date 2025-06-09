import os
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from torchmetrics.classification import Accuracy
from deepforest.evaluate import evaluate_boxes
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher

class PipelineEvaluation:
    def __init__(self, predictions, annotations, classification_label_dict, detection_true_positive_threshold=0.85, classification_threshold=0.5):
        """Initialize pipeline evaluation.
        
        Args:
            predictions: DataFrame containing the predictions
            annotations: DataFrame containing the annotations
            classification_label_dict: Dictionary mapping classification labels to integers 
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            classification_threshold (float): Threshold for classification confidence score
        """
        self.detection_true_positive_threshold = detection_true_positive_threshold 
        self.classification_threshold = classification_threshold 
        self.predictions = predictions
        self.annotations = annotations
        self.classification_label_dict = classification_label_dict

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
            targets["boxes"] = torch.tensor(annotations_df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32"))
            targets["labels"] = torch.tensor(annotations_df[label_column].astype(int).values)
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor(annotations_df["score"].tolist())

        return targets
    
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
        matcher = Matcher(0.3, 0.3, allow_low_quality_matches=False)
        
        pred_boxes = pred["boxes"]
        src_boxes = target["boxes"]

        match_quality_matrix = box_iou(src_boxes, pred_boxes)
        
        results = matcher(match_quality_matrix)

        matched_pred = []
        matched_target = []
        matched_score = []

        for i, match in enumerate(results):
            if match >= 0:
                matched_pred.append(int(pred["labels"][i].item()))
                matched_target.append(int(target["labels"][match].item()))
                matched_score.append(pred["scores"][i].item())
            else:
                matched_pred.append(int(pred["labels"][i].item()))
                matched_target.append(None)
                matched_score.append(pred["scores"][i].item())

        matches = pd.DataFrame({"pred": matched_pred, "target": matched_target, "score":matched_score})
        matches = matches.dropna(subset=["target"])

        return matches
            
    def evaluate_confident_classification(self, predictions):
        """Evaluate confident classification performance"""
        return self._evaluate_classification(predictions)

    def evaluate_uncertain_classification(self, predictions):
        """Evaluate uncertain classification performance"""
        return self._evaluate_classification(predictions)

    def _evaluate_classification(self, predictions):
        """Helper function to evaluate classification performance.
        
        Args:
            predictions (DataFrame): DataFrame containing the predictions.
            accuracy_metric (torchmetrics.Metric): Metric to evaluate accuracy.
        
        Returns:
            dict: Dictionary containing the computed accuracy metric.
        """
        self.classification_annotations = self.annotations.copy(deep=True)

        # Remove empty frames from classification annotations
        self.classification_annotations = self.annotations.copy(deep=True)
        self.classification_annotations = self.classification_annotations[~self.classification_annotations.label.isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"])]
        self.classification_annotations = self.classification_annotations[self.classification_annotations.xmin != 0]
        self.classification_annotations = self.classification_annotations[~self.classification_annotations.label.isnull()]
        
        if self.classification_annotations.empty:
            return {"accuracy": None, "avg_score_true_positive": None, "avg_score_false_positive": None}
        
        # Only two word labels
        self.classification_annotations["label"] = self.classification_annotations["label"].apply(lambda x: ' '.join(x.split()[:2]))
        self.classification_annotations = self.classification_annotations[self.classification_annotations["label"].apply(lambda x: len(x.split()) == 2)]
    
        if self.classification_annotations.empty:
            return {"accuracy": None, "avg_score_true_positive": None, "avg_score_false_positive": None}

        # Only evaluate classes that are in the label dict?
        self.classification_annotations = self.classification_annotations[self.classification_annotations["label"].isin(self.classification_label_dict.keys())]
        self.classification_annotations["cropmodel_label"] = self.classification_annotations["label"].apply(lambda x: self.classification_label_dict[x])
        predictions["cropmodel_label"] = predictions["cropmodel_label"].apply(lambda x: self.classification_label_dict[x])

        # Metrics
        num_classes = len(self.classification_annotations["cropmodel_label"].unique())
        
        if num_classes == 0:
            return {"accuracy": None, "avg_score_true_positive": None, "avg_score_false_positive": None}
        elif num_classes == 1:
            micro_accuracy = Accuracy(average="micro", task="binary")
            # In a single class scenario, micro and macro accuracy are the same
            macro_accuracy = micro_accuracy
        else:  
            micro_accuracy = Accuracy(average="micro", task="multiclass", num_classes=num_classes)
            macro_accuracy = Accuracy(average="macro", task="multiclass", num_classes=num_classes)
        
        true_positive_scores = []
        false_positive_scores = []
        for image_path in predictions.drop_duplicates("image_path").image_path.tolist():
            image_targets = self.classification_annotations.loc[self.classification_annotations.image_path == os.path.basename(image_path)]
            image_predictions = predictions.loc[predictions.image_path == os.path.basename(image_path)]            
            if image_predictions.empty:
                continue
            
            target = self._format_targets(image_targets, label_column="cropmodel_label")
            pred = self._format_targets(image_predictions, label_column="cropmodel_label")
            if len(target["labels"]) == 0:
                continue
            matches = self.match_predictions_and_targets(pred, target)
            if matches.empty:
                continue
            
            true_positive_scores.append(matches.loc[matches.target == matches.pred,"score"])
            false_positive_scores.append(matches.loc[~(matches.target == matches.pred),"score"])

            micro_accuracy.update(preds=torch.tensor(matches["pred"].values), target=torch.tensor(matches["target"].values))
            macro_accuracy.update(preds=torch.tensor(matches["pred"].values), target=torch.tensor(matches["target"].values))

        results = {
            "micro_accuracy": micro_accuracy.compute(), 
            "macro_accuracy": macro_accuracy.compute(),
            "avg_true_classification_score": round(float(np.mean([score.mean() for score in true_positive_scores])), 2), 
            "avg_false_classification_score": round(float(np.mean([score.mean() for score in false_positive_scores])), 2)
        }
        
        return results
    
    def evaluate_detection(self):
        """Evaluate detection performance"""
        detection_predictions = self.predictions[self.predictions.image_path.isin(self.annotations.image_path)]
        detection_predictions = detection_predictions[detection_predictions.xmin != 0]   
        detection_predictions = detection_predictions[~detection_predictions.label.isnull()]
        combined_predictions = gpd.GeoDataFrame(detection_predictions)

        # When you concat geodataframes, you get pandas dataframes
        combined_predictions = combined_predictions[~combined_predictions["score"].isna()]

        # Check if geometry is string or polygon
        combined_predictions = combined_predictions
        ground_truth = self.annotations.copy(deep=True)

        iou_results = evaluate_boxes(
            combined_predictions,
            ground_truth,
            iou_threshold=self.detection_true_positive_threshold
        )
        
        non_empty_results = iou_results["results"][~iou_results["results"]["score"].isna()]
        if non_empty_results.empty:
            return {"recall": None, "precision": None, "avg_score_true_positive": None, "avg_score_false_positive": None}
        else:
            # Convert match to boolean
            non_empty_results["match"] = non_empty_results["match"].astype(bool)
            avg_score_true_positive = non_empty_results.loc[non_empty_results["match"]].score.mean()
            avg_score_false_positive = non_empty_results.loc[~non_empty_results["match"]].score.mean()
        
        results = {
            "recall": iou_results["box_recall"], 
            "precision": iou_results["box_precision"], 
            "avg_score_true_positive": avg_score_true_positive, 
            "avg_score_false_positive": avg_score_false_positive
        }

        return results

    def evaluate(self):
        """Evaluate pipeline performance for both detection and classification"""
        if self.predictions is None:
            self.results = {"detection": {"recall": 0, "precision": None, "avg_score_true_positive": None, "avg_score_false_positive": None},
            "classification": {"confident": {"micro_accuracy": 0,"macro_accuracy": 0, "avg_score_true_positive": None, "avg_score_false_positive": None}, "uncertain": {"micro_accuracy": 0, "macro_accuracy":0, "avg_score_true_positive": None, "avg_score_false_positive": None}}
            }

        else:
            self.confident_predictions = self.predictions[self.predictions.cropmodel_score > self.classification_threshold]
            self.uncertain_predictions = self.predictions[self.predictions.cropmodel_score <= self.classification_threshold]
            self.predictions["cropmodel_label"] = self.predictions["cropmodel_label"].apply(lambda x: self.classification_label_dict[x])

            detection_results = self.evaluate_detection()
            confident_classification_results = self.evaluate_confident_classification(self.confident_predictions)
            uncertain_classification_results = self.evaluate_uncertain_classification(self.uncertain_predictions)
            self.results = {
                "detection": detection_results, 
                "classification": {
                    "confident": confident_classification_results, 
                    "uncertain": uncertain_classification_results
                }
            }

        return self.results

    def check_success(self):
        """Check if pipeline performance is satisfactory"""
        # For each metric, check if it is above the threshold
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No results found, run evaluate() first")

        if self.results["detection"]["recall"] is None:
            return False
        if self.results['detection']["recall"] > self.detection_true_positive_threshold:
            return True
        else:
            return False