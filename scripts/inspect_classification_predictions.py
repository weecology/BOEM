"""
Script to inspect classification model predictions using different methods.

This script loads a classification model checkpoint, runs detection on a specific image,
and compares predictions from:
1. Direct model forward method
2. Batch predictions from dataloader
3. DeepForest predict_tile with crop_model
"""

import os
import tempfile
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from deepforest.model import CropModel
from deepforest.datasets.cropmodel import BoundingBoxDataset
from src import detection
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Disable TensorBoard logging to avoid disk quota issues
os.environ["TENSORBOARD_LOGDIR"] = tempfile.gettempdir()
# Disable any default logging
os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"


@hydra.main(config_path="boem_conf", config_name="boem_config", version_base=None)
def main(cfg: DictConfig):
    """Main function to inspect classification predictions."""
    
    # Image of interest
    image_name = "C7_L4_F3005_T20241220_120102_924.jpg"
    image_dir = "/blue/ewhite/b.weinstein/BOEM/GulfMexico/JPG_20241220_104800"
    image_path = os.path.join(image_dir, image_name)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image: {image_path}")
    
    # Load classification model from checkpoint
    classification_checkpoint = cfg.classification_model.checkpoint
    print(f"Loading classification model from: {classification_checkpoint}")
    classification_model = CropModel.load_from_checkpoint(classification_checkpoint)
    classification_model.eval()
    
    print(classification_model.label_dict)
    print(classification_model.numeric_to_label_dict)

    # Load detection model
    detection_checkpoint = cfg.detection_model.checkpoint
    print(f"Loading detection model from: {detection_checkpoint}")
    detection_model = detection.load(detection_checkpoint)
    
    # Configure detection model
    detection_model.config["batch_size"] = 2
    detection_model.config["workers"] = cfg.detection_model.trainer.workers
    
    # Run detection model on the image to get bounding boxes
    print(f"\nRunning detection model on {image_name}...")
    detection_predictions = detection_model.predict_tile(
        path=[image_path],
        patch_size=cfg.predict.patch_size,
        patch_overlap=cfg.predict.patch_overlap,
        dataloader_strategy="batch",
        crop_model=None,  # No classification yet
    )
    
    if detection_predictions is None or len(detection_predictions) == 0:
        print("No detections found in image")
        return
    
    # Filter by min_score
    detection_predictions = detection_predictions[detection_predictions["score"] >= cfg.predict.min_score]
    
    if len(detection_predictions) == 0:
        print(f"No detections above min_score {cfg.predict.min_score}")
        return
    
    print(f"Found {len(detection_predictions)} detections")
    
    # Prepare detection results for classification (need image_path as basename)
    detection_results = detection_predictions.copy()
    detection_results["image_path"] = image_name
    
    # Load image to get dimensions for bounding box clamping
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image {image_path}")
    img_height, img_width = original_image.shape[:2]
    
    # Create expanded detection results for methods 1 and 2 (30 pixel expansion like training)
    detection_results_expanded = detection_results.copy()
    detection_results_expanded["xmin"] = (detection_results_expanded["xmin"] - 30).clip(0, img_width)
    detection_results_expanded["ymin"] = (detection_results_expanded["ymin"] - 30).clip(0, img_height)
    detection_results_expanded["xmax"] = (detection_results_expanded["xmax"] + 30).clip(0, img_width)
    detection_results_expanded["ymax"] = (detection_results_expanded["ymax"] + 30).clip(0, img_height)
    
    print(f"Expanded bounding boxes by 30 pixels on all sides for methods 1 and 2")
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Create bounding box dataset using expanded coordinates (for methods 1 and 2)
    bounding_box_dataset = BoundingBoxDataset(
        detection_results_expanded,
        root_dir=image_dir,
        transform=classification_model.get_transform(augmentations=None),
        
    )
    
    # Visualize crops
    print("\n" + "="*80)
    print("VISUALIZING CROPS")
    print("="*80)
    
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create visualization with bounding boxes (original in red, expanded in green)
    img_with_boxes = original_image_rgb.copy()
    for idx, row in detection_results.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Red for original
        cv2.putText(img_with_boxes, f"{idx}", (xmin, ymin - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Also draw expanded boxes in green
    for idx, row in detection_results_expanded.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)  # Green for expanded
    
    # Extract and visualize first few crops (using expanded boxes for methods 1 and 2)
    n_crops_to_show = min(9, len(bounding_box_dataset))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Crops extracted from {image_name} (showing first {n_crops_to_show}) - EXPANDED BY 30px (for methods 1 & 2)', fontsize=14)
    
    # Create directory for individual crop saves
    crops_dir = os.path.join(tempfile.gettempdir(), f"crops_{image_name.replace('.jpg', '')}")
    os.makedirs(crops_dir, exist_ok=True)
    
    for i in range(n_crops_to_show):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Extract crop from original image using expanded bounding box coordinates
        bbox_row = detection_results_expanded.iloc[i]
        xmin, ymin, xmax, ymax = int(bbox_row['xmin']), int(bbox_row['ymin']), int(bbox_row['xmax']), int(bbox_row['ymax'])
        
        # Extract expanded crop from original image
        crop = original_image_rgb[ymin:ymax, xmin:xmax]
        
        # Calculate exact dimensions
        crop_width = xmax - xmin
        crop_height = ymax - ymin
        
        if crop.size > 0:
            ax.imshow(crop, aspect='auto')
            ax.set_title(f"Box {i}\nCoords: ({xmin},{ymin})-({xmax},{ymax})\nSize: {crop_width}x{crop_height}px", fontsize=7)
            ax.axis('off')
            
            # Save individual crop with expanded size
            crop_path = os.path.join(crops_dir, f"crop_{i}_size_{crop_width}x{crop_height}_expanded.png")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        else:
            ax.text(0.5, 0.5, f"Empty crop {i}", ha='center', va='center')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_crops_to_show, 9):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(tempfile.gettempdir(), f"crops_visualization_{image_name.replace('.jpg', '')}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved crop visualization to: {viz_path}")
    print(f"Saved individual crops (expanded by 30px) to: {crops_dir}")
    
    # Also save image with bounding boxes
    bbox_viz_path = os.path.join(tempfile.gettempdir(), f"bboxes_{image_name}")
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_boxes)
    plt.title(f'Detections on {image_name} ({len(detection_results)} boxes)\nRed=original, Green=expanded (+30px)')
    plt.axis('off')
    plt.savefig(bbox_viz_path, dpi=150, bbox_inches='tight')
    print(f"Saved bounding box visualization to: {bbox_viz_path}")
    plt.close('all')
    
    # Method 1: Direct model forward method (using expanded bounding boxes +30px)
    print("\n" + "="*80)
    print("METHOD 1: Direct model forward method (bounding boxes expanded by 30px on all sides)")
    print("="*80)
    
    predictions_method1 = []
    with torch.no_grad():
        for i in range(len(bounding_box_dataset)):
            image_tensor = bounding_box_dataset[i].unsqueeze(0)
            outputs = classification_model.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            
            pred_class_idx = probs.argmax(dim=1).item()
            pred_prob = probs.max(dim=1)[0].item()
            pred_label = classification_model.numeric_to_label_dict[pred_class_idx]
            
            predictions_method1.append({
                "detection_idx": i,
                "predicted_class_idx": pred_class_idx,
                "predicted_label": pred_label,
                "predicted_prob": pred_prob,
                "xmin": detection_results_expanded.iloc[i]["xmin"],
                "ymin": detection_results_expanded.iloc[i]["ymin"],
                "xmax": detection_results_expanded.iloc[i]["xmax"],
                "ymax": detection_results_expanded.iloc[i]["ymax"],
            })
    
    df_method1 = pd.DataFrame(predictions_method1)
    print(f"\nPredictions from direct forward method:")
    print(df_method1[["detection_idx", "predicted_label", "predicted_prob", "xmin", "ymin", "xmax", "ymax"]])
    
    # Method 2: Batch predictions from dataloader (using expanded bounding boxes +30px)
    print("\n" + "="*80)
    print("METHOD 2: Batch predictions from dataloader (bounding boxes expanded by 30px on all sides)")
    print("="*80)
    
    dataloader = DataLoader(
        bounding_box_dataset,
        batch_size=cfg.classification_model.batch_size,
        shuffle=False,
        num_workers=cfg.classification_model.workers
    )
    
    predictions_method2 = []
    with torch.no_grad():
        batch_idx = 0
        for batch_images in dataloader:
            outputs = classification_model.model(batch_images)
            probs = F.softmax(outputs, dim=1)
            
            for i in range(len(batch_images)):
                pred_class_idx = probs[i].argmax().item()
                pred_prob = probs[i].max().item()
                pred_label = classification_model.numeric_to_label_dict[pred_class_idx]
                
                detection_idx = batch_idx * cfg.classification_model.batch_size + i
                if detection_idx < len(detection_results_expanded):
                    predictions_method2.append({
                        "detection_idx": detection_idx,
                        "predicted_class_idx": pred_class_idx,
                        "predicted_label": pred_label,
                        "predicted_prob": pred_prob,
                        "xmin": detection_results_expanded.iloc[detection_idx]["xmin"],
                        "ymin": detection_results_expanded.iloc[detection_idx]["ymin"],
                        "xmax": detection_results_expanded.iloc[detection_idx]["xmax"],
                        "ymax": detection_results_expanded.iloc[detection_idx]["ymax"],
                    })
            batch_idx += 1
    
    df_method2 = pd.DataFrame(predictions_method2)
    print(f"\nPredictions from batch dataloader method:")
    print(df_method2[["detection_idx", "predicted_label", "predicted_prob", "xmin", "ymin", "xmax", "ymax"]])
    
    # Method 3: DeepForest predict_tile with crop_model (using original bounding boxes, no expansion)
    print("\n" + "="*80)
    print("METHOD 3: DeepForest predict_tile with crop_model (original bounding boxes, no expansion)")
    print("="*80)
    
    # Ensure detection model is configured (in case it was reset)
    detection_model.config["batch_size"] = cfg.predict.batch_size
    detection_model.config["workers"] = cfg.detection_model.trainer.workers
    
    predictions_method3 = detection_model.predict_tile(
        path=[image_path],
        patch_size=cfg.predict.patch_size,
        patch_overlap=cfg.predict.patch_overlap,
        dataloader_strategy="batch",
        crop_model=classification_model,
    )
    
    if predictions_method3 is not None and len(predictions_method3) > 0:
        predictions_method3 = predictions_method3[predictions_method3["score"] >= cfg.predict.min_score]
        print(f"\nPredictions from predict_tile method:")
        print(predictions_method3[["xmin", "ymin", "xmax", "ymax", "score", "cropmodel_label", "cropmodel_score"]])
    else:
        print("No predictions from predict_tile method")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    if len(df_method1) > 0 and len(df_method2) > 0:
        # Compare method 1 and method 2
        comparison = pd.merge(
            df_method1[["detection_idx", "predicted_label", "predicted_prob"]],
            df_method2[["detection_idx", "predicted_label", "predicted_prob"]],
            on="detection_idx",
            suffixes=("_method1", "_method2")
        )
        
        # Check for differences
        label_matches = comparison["predicted_label_method1"] == comparison["predicted_label_method2"]
        prob_diffs = (comparison["predicted_prob_method1"] - comparison["predicted_prob_method2"]).abs()
        
        print(f"\nMethod 1 vs Method 2:")
        print(f"  Label matches: {label_matches.sum()}/{len(comparison)}")
        print(f"  Average probability difference: {prob_diffs.mean():.6f}")
        print(f"  Max probability difference: {prob_diffs.max():.6f}")
        
        if not label_matches.all():
            print("\n  Mismatches found:")
            mismatches = comparison[~label_matches]
            print(mismatches[["detection_idx", "predicted_label_method1", "predicted_prob_method1", 
                             "predicted_label_method2", "predicted_prob_method2"]])
    
    if predictions_method3 is not None and len(predictions_method3) > 0:
        print(f"\nMethod 3 (predict_tile) found {len(predictions_method3)} predictions")
        print(f"  Labels: {predictions_method3['cropmodel_label'].value_counts().to_dict()}")
        print(f"  Average score: {predictions_method3['cropmodel_score'].mean():.4f}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Image: {image_name}")
    print(f"Total detections: {len(detection_results)}")
    print(f"Method 1 predictions: {len(df_method1)}")
    print(f"Method 2 predictions: {len(df_method2)}")
    print(f"Method 3 predictions: {len(predictions_method3) if predictions_method3 is not None else 0}")


if __name__ == "__main__":
    main()

