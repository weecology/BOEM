from src.label_studio import gather_data
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics.functional import confusion_matrix
from src.detection import predict
import pandas as pd
import torch
import os

class PipelineEvaluation:
    def __init__(self, model, crop_model, image_dir, detect_ground_truth_dir, classify_ground_truth_dir, detection_true_positive_threshold=0.8, detection_false_positive_threshold=0.5, classification_avg_score=0.5, patch_size=450, patch_overlap=0, min_score=0.5, debug=False):
        """Initialize pipeline evaluation.
        
        Args:
            model: Trained model for making predictions
            crop_model: Trained model for making crop predictions
            image_dir (str): Directory containing images
            detect_ground_truth_dir (str): Directory containing detection ground truth annotation CSV files
            classify_ground_truth_dir (str): Directory containing confident classification ground truth annotation CSV files
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            detection_false_positive_threshold (float): IoU threshold for considering a detection a false positive
            classification_threshold (float): Threshold for classification confidence score
            patch_size (int): Size of image patches for prediction
            patch_overlap (int): Overlap between patches
            min_score (float): Minimum confidence score threshold for predictions
        """
        self.detection_true_positive_threshold = detection_true_positive_threshold 
        self.detection_false_positive_threshold = detection_false_positive_threshold
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

        # Gather data
        self.detection_annotations = gather_data(detect_ground_truth_dir)
        self.detection_annotations = self.detection_annotations[self.detection_annotations.label.isin(["Bird","Cetacean","Turtle"])]
        self.classification_annotations = gather_data(classify_ground_truth_dir)
        
        # There is one caveat for empty frames, assign a label which the dict contains
        self.classification_annotations.loc[self.classification_annotations.label.astype(str)=='0',"label"] = self.model.numeric_to_label_dict[0]

        # Prediction container
        self.predictions = []
        
        self.confident_predictions, self.uncertain_predictions = self.predict_classification()
        self.num_classes = len(self.classification_annotations["label"].unique())
            
        if self.num_classes == 1:
            self.num_classes = 2

    def _format_targets(self, annotations_df):
        targets = {}
        
        if annotations_df.xmin.isna().all():
            # Torchmetrics expects empty tensors, see https://github.com/Lightning-AI/torchmetrics/pull/624/files
            targets["boxes"] = torch.tensor([])
            targets["labels"] = torch.tensor([])
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor([])
        elif annotations_df.empty:
            targets["boxes"] = torch.tensor([])
            targets["labels"] = torch.tensor([])
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor([])
        else:
            targets["boxes"] = torch.tensor(annotations_df[["xmin", "ymin", "xmax","ymax"]].values.astype("float32"))
            targets["labels"] = torch.tensor([self.model.label_dict[x] for x in annotations_df["label"].tolist()])
            if "score" in annotations_df.columns:
                targets["scores"] = torch.tensor(annotations_df["score"].tolist())

        return targets

    def evaluate_detection(self):
        """Evaluate detection performance"""
        
        # Metrics
        self.mAP = MeanAveragePrecision(box_format="xyxy",extended_summary=True)

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

        targets = []
        preds = []
        for image_predictions in predictions:
            # Min score for predictions
            image_targets = self.detection_annotations.loc[self.detection_annotations.image_path == image_predictions["image_path"].iloc[0]]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            targets.append(target)
            preds.append(pred)
        
        # Minimum 
        self.mAP.update(preds=preds, target=targets)

        results = {"mAP": self.mAP.compute()}

        return results

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
        )
        combined_predictions = pd.concat(predictions)
        self.predictions.append(combined_predictions)

        # Split into confident and uncertain based on average score
        average_score = combined_predictions.groupby("image_path").apply(lambda x: x["score"].mean())

        # Which images have a score above the average
        confident_images = average_score[average_score > self.classification_avg_score].index

        # Select the annotations for confident and uncertain
        confident_predictions = combined_predictions[combined_predictions.image_path.isin(confident_images)]
        uncertain_predictions = combined_predictions[~combined_predictions.image_path.isin(confident_images)]

        return confident_predictions, uncertain_predictions
    
    def evaluate_confident_classification(self):
        """Evaluate confident classification performance"""

        targets = []
        preds = []
        for image_path in self.confident_predictions.drop_duplicates("image_path").image_path.tolist():
            # Min score for predictions
            image_targets = self.classification_annotations.loc[self.classification_annotations.image_path == os.path.basename(image_path)]
            image_predictions = self.confident_predictions.loc[self.confident_predictions.image_path == os.path.basename(image_path)]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            if len(pred["labels"]) == 0:
                continue
            targets.append(target)
            preds.append(pred)
        
        if len(preds) == 0:
            return {"confident_classification_accuracy": None}
        else:
            # Classification is just the labels dict
            target_labels = torch.stack([x["labels"] for x in targets])
            pred_labels = torch.stack([x["labels"] for x in preds])

            self.confident_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)

            self.confident_classification_accuracy.update(preds=pred_labels, target=target_labels)
            results = {"confident_classification_accuracy": self.confident_classification_accuracy.compute()}

        return results

    def evaluate_uncertain_classification(self):
        """Evaluate uncertain classification performance"""

        self.uncertain_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=self.num_classes)

        targets = []
        preds = []
        for image_path in self.uncertain_predictions.drop_duplicates("image_path").image_path.tolist():
            image_targets = self.classification_annotations.loc[self.classification_annotations.image_path == os.path.basename(image_path)]
            image_predictions = self.uncertain_predictions.loc[self.uncertain_predictions.image_path == os.path.basename(image_path)]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            
            if len(pred["labels"]) == 0:
                    continue
            targets.append(target)
            preds.append(pred)

        if len(preds) == 0:
            
            return {"uncertain_classification_accuracy": None}
        else:
            # Classification is just the labels dict
            target_labels = torch.stack([x["labels"] for x in targets])
            pred_labels = torch.stack([x["labels"] for x in preds])
            
            self.uncertain_classification_accuracy.update(preds=pred_labels, target=target_labels)
            results = {"uncertain_classification_accuracy": self.uncertain_classification_accuracy.compute()}
            
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

        if self.results['detection']["mAP"]["map"] > self.detection_true_positive_threshold:
            return True
        else:
            return False