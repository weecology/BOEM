from src.label_studio import gather_data
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics.functional import confusion_matrix
import pandas as pd

class PipelineEvaluation:
    def __init__(self, model, detect_ground_truth_dir=None, classify_confident_ground_truth_dir=None, classify_uncertain_ground_truth_dir=None, detection_true_positive_threshold=0.8, detection_false_positive_threshold=0.5, classification_avg_score=0.5, target_labels=None, patch_size=450, patch_overlap=0, min_score=0.5):
        """Initialize pipeline evaluation.
        
        Args:
            model: Trained model for making predictions
            detect_ground_truth_dir (str): Directory containing detection ground truth annotation CSV files
            classify_confident_ground_truth_dir (str): Directory containing confident classification ground truth annotation CSV files
            classify_uncertain_ground_truth_dir (str): Directory containing uncertain classification ground truth annotation CSV files
            detection_true_positive_threshold (float): IoU threshold for considering a detection a true positive
            detection_false_positive_threshold (float): IoU threshold for considering a detection a false positive
            classification_avg_score (float): Threshold for classification confidence score
            target_labels (list): List of target class labels to evaluate
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

        self.detection_annotations_df = gather_data(detect_ground_truth_dir)
        self.classification_confident_annotations_df = gather_data(classify_confident_ground_truth_dir)
        self.classification_uncertain_annotations_df = gather_data(classify_uncertain_ground_truth_dir)

        self.model = model

        # Metrics
        self.mAP = MeanAveragePrecision(box_format="xyxy",extended_summary=True, iou_threshold=detection_true_positive_threshold)
        self.classification_accuracy = Accuracy(average="micro", num_classes=len(target_labels))

    def _format_targets(self, annotations_df):
        targets = {}
        targets["boxes"] = annotations_df[["xmin", "ymin", "xmax",
                                                  "ymax"]].values.astype("float32")
        targets["labels"] = [self.model.label_dict[x] for x in annotations_df["label"].tolist()]

        return targets

    def evaluate_detection(self):
        preds = self.model.predict(
            self.detection_annotations_df.image_path.tolist(), 
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap, 
            min_score=self.min_score
        )
        targets = self._format_targets(self.detection_annotations_df)

        self.mAP.update(preds=preds, target=targets)

        return self.mAP.compute()

    def confident_classification_accuracy(self):
        self.classification_accuracy.update(self.classification_confident_annotations_df)
        return self.classification_accuracy.compute()

    def uncertain_classification_accuracy(self):
        self.classification_accuracy.update(self.classification_uncertain_annotations_df)
        return self.classification_accuracy.compute()

    def target_classification_accuracy(self):
        # Combine confident and uncertain classifications
        combined_annotations_df = pd.concat([self.classification_confident_annotations_df, self.classification_uncertain_annotations_df])
        if self.target_classes is not None:
            self.classification_accuracy.update(combined_annotations_df, self.target_classes)
            return self.classification_accuracy.compute()
        else:
            return None

    def evaluate_pipeline(self):
        """
        Evaluate pipeline performance for both detection and classification
            
        """
        detection_results = self.evaluate_detection()
        confident_classification_results = self.confident_classification_accuracy()
        uncertain_classification_results = self.uncertain_classification_accuracy()

        results = {
            'detection': detection_results,
            'confident_classification': confident_classification_results,
            'uncertain_classification': uncertain_classification_results
        }
        
        return results
