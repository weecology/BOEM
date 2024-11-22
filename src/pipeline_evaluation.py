from src.label_studio import gather_data
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics.functional import confusion_matrix
from src.detection import predict
import pandas as pd
import torch
class PipelineEvaluation:
    def __init__(self, model, crop_model, image_dir, detect_ground_truth_dir=None, classify_confident_ground_truth_dir=None, classify_uncertain_ground_truth_dir=None, detection_true_positive_threshold=0.8, detection_false_positive_threshold=0.5, classification_avg_score=0.5, patch_size=450, patch_overlap=0, min_score=0.5):
        """Initialize pipeline evaluation.
        
        Args:
            model: Trained model for making predictions
            crop_model: Trained model for making crop predictions
            image_dir (str): Directory containing images
            detect_ground_truth_dir (str): Directory containing detection ground truth annotation CSV files
            classify_confident_ground_truth_dir (str): Directory containing confident classification ground truth annotation CSV files
            classify_uncertain_ground_truth_dir (str): Directory containing uncertain classification ground truth annotation CSV files
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            detection_false_positive_threshold (float): IoU threshold for considering a detection a false positive
            classification_avg_score (float): Threshold for classification confidence score
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
        self.confident_classification_ground_truth_dir = classify_confident_ground_truth_dir
        self.uncertain_classification_ground_truth_dir = classify_uncertain_ground_truth_dir
        self.image_dir = image_dir
        self.classification_model = crop_model

        # Gather data
        self.detection_annotations_df = gather_data(detect_ground_truth_dir)
        self.confident_classification_annotations_df = gather_data(classify_confident_ground_truth_dir)
        self.uncertain_classification_annotations_df = gather_data(classify_uncertain_ground_truth_dir)

        # If no annotations, raise errors
        if self.detection_annotations_df.empty:
            raise ValueError("No detection annotations found")
        if self.confident_classification_annotations_df.empty:
            raise ValueError("No confident classification annotations found")
        if self.uncertain_classification_annotations_df.empty:
            raise ValueError("No uncertain classification annotations found")

        self.model = model

        # Metrics
        self.mAP = MeanAveragePrecision(box_format="xyxy",extended_summary=True)
        self.confident_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=len(self.confident_classification_annotations_df["label"].unique()))
        self.uncertain_classification_accuracy = Accuracy(average="micro", task="multiclass", num_classes=len(self.uncertain_classification_annotations_df["label"].unique()))

        # Prediction container
        self.predictions = []

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
        full_image_paths = [self.image_dir + "/" + image_path for image_path in self.detection_annotations_df.image_path.tolist()] 
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
            image_targets = self.detection_annotations_df.loc[self.detection_annotations_df.image_path == image_predictions["image_path"].iloc[0]]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            targets.append(target)
            preds.append(pred)
        
        # Minimum 
        self.mAP.update(preds=preds, target=targets)

        results = {"mAP": self.mAP.compute()}

        return results

    def evaluate_confident_classification(self):
        """Evaluate confident classification performance"""

        full_image_paths = [self.image_dir + "/" + image_path for image_path in self.confident_classification_annotations_df.image_path.tolist()] 
        predictions = predict(
            m=self.model,
            crop_model=self.classification_model,
            image_paths=full_image_paths, 
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap, 
        )
        combined_predictions = pd.concat(predictions)
        combined_predictions["workflow"] = "confident_classification"
        self.predictions.append(combined_predictions)

        targets = []
        preds = []
        for image_predictions in predictions:
            # Min score for predictions
            image_targets = self.confident_classification_annotations_df.loc[self.confident_classification_annotations_df.image_path == image_predictions["image_path"].iloc[0]]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            targets.append(target)
            preds.append(pred)
        
        # Classification is just the labels dict
        target_labels = torch.stack([x["labels"] for x in targets])
        pred_labels = torch.stack([x["labels"] for x in preds])

        self.confident_classification_accuracy.update(preds=pred_labels, target=target_labels)
        results = {"confident_classification_accuracy": self.confident_classification_accuracy.compute()}


        return results

    def evaluate_uncertain_classification(self):
        """Evaluate uncertain classification performance"""

        full_image_paths = [self.image_dir + "/" + image_path for image_path in self.uncertain_classification_annotations_df.image_path.tolist()] 
        predictions = predict(
            m=self.model,
            crop_model=self.classification_model,
            image_paths=full_image_paths, 
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap, 
        )
        combined_predictions = pd.concat(predictions)
        combined_predictions["workflow"] = "uncertain_classification"
        self.predictions.append(combined_predictions)

        targets = []
        preds = []
        for image_predictions in predictions:
            # Min score for predictions
            image_targets = self.uncertain_classification_annotations_df.loc[self.uncertain_classification_annotations_df.image_path == image_predictions["image_path"].iloc[0]]
            image_predictions = image_predictions[image_predictions.score > self.min_score]
            target = self._format_targets(image_targets)
            pred = self._format_targets(image_predictions)
            targets.append(target)
            preds.append(pred)

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