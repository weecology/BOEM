from src.label_studio import gather_data
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics.functional import confusion_matrix

class PipelineEvaluation:
    def __init__(self, model, detection_annotation_dir=None, classification_annotation_dir=None, iou_threshold=0.4, target_classes=None):
        """Initialize pipeline evaluation"""
        self.iou_threshold = iou_threshold 
        self.detection_annotations_df = gather_data(detection_annotation_dir)
        self.classification_annotations_df = gather_data(classification_annotation_dir)

        self.model = model

        # Metrics
        self.mAP = MeanAveragePrecision(box_format="xyxy",extended_summary=True)
        self.classification_accuracy = Accuracy()

    def _format_targets(self, annotations_df):
        return {"boxes": annotations_df["bbox"].tolist(),
                "labels": annotations_df["label"].tolist()}

    def evaluate_detection(self):
        preds = self.model.predict(self.detection_annotations_df)
        targets = self._format_targets(self.detection_annotations_df)

        self.mAP.update(preds=preds, target=targets)

        return self.mAP.compute()

    def classification_accuracy(self):
        self.classification_accuracy.update(self.classification_annotations_df)
        return self.classification_accuracy.compute()

    def confusion_matrix(self):
        return confusion_matrix(self.classification_annotations_df)

    def target_classification_accuracy(self):
        if self.target_classes is not None:
            self.classification_accuracy.update(self.classification_annotations_df, self.target_classes)
            return self.classification_accuracy.compute()
        else:
            return None

    def evaluate_pipeline(self, predictions, ground_truth):
        """
        Evaluate pipeline performance for both detection and classification
        
        Args:
            predictions: List of dictionaries containing predicted boxes and classes
                Each dict should have 'bbox' (x,y,w,h) and 'class_label'
            ground_truth: List of dictionaries containing ground truth annotations
                Each dict should have 'bbox' (x,y,w,h) and 'class_label'
        """
        detection_results = self.evaluate_detection()
        classification_results = self.classification_accuracy()
        
        results = {
            'detection': {
                'precision': detection_results["precision"],
                'recall': detection_results["recall"],
                'f1_score': detection_results["f1_score"],
                'total_predictions': detection_results["total_predictions"],
                'total_ground_truth': detection_results["total_ground_truth"],
                'true_positives': detection_results["true_positives"]
            },
            'classification': {
                'accuracy': classification_results["accuracy"],
                'correct_classifications': classification_results["correct_classifications"],
                'total_correct_detections': classification_results["total_correct_detections"]
            }
        }
        
        return results
