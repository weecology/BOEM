#!/usr/bin/env python3
"""
Generate high-quality detection visualizations with DeepForest's native visualization.
Designed for GPU acceleration via SLURM.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import detection
from deepforest import visualize
import cv2

IMAGE_DIR = "/blue/ewhite/b.weinstein/BOEM/NOAA/Camera B"
ANNOTATIONS_FILE = os.path.join(IMAGE_DIR, "annotations.viame.csv")
OUTPUT_DIR = os.path.join(IMAGE_DIR, "visualizations")

# Model paths
DETECTION_CHECKPOINT = "/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt"

def load_viame_annotations(csv_file):
    """Load VIAME CSV annotations."""
    annotations = []
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 11:
                continue

            filename = parts[1]
            left, top, right, bottom = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            species = parts[9].strip()

            annotations.append({
                'image': filename,
                'xmin': left, 'ymin': top, 'xmax': right, 'ymax': bottom,
                'label': species
            })

    return pd.DataFrame(annotations)

def create_output_dir():
    """Create output directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("="*80)
    print("CAMERA B - HIGH QUALITY DETECTION VISUALIZATION (GPU)")
    print("="*80)

    create_output_dir()

    # Load annotations
    print("\nLoading ground truth annotations...")
    gt = load_viame_annotations(ANNOTATIONS_FILE)
    print(f"  ✓ {len(gt)} annotations loaded")

    # Load detection model
    print("\nLoading detection model on GPU...")
    det_model = detection.load(DETECTION_CHECKPOINT)
    det_model.config["batch_size"] = 64  # GPU can handle larger batches
    det_model.config["workers"] = 5
    print(f"  ✓ Detection model loaded")

    # Get images
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\nProcessing {len(image_files)} high-resolution images...")

    stats = {
        'images_processed': 0,
        'total_detections': 0,
        'images_with_detections': 0
    }

    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(IMAGE_DIR, img_file)
        print(f"  [{idx:2d}/{len(image_files)}] {img_file}", end=" ... ", flush=True)

        try:
            # Run detection with GPU
            predictions = det_model.predict_tile(
                path=[img_path],
                patch_size=1000,
                patch_overlap=0,
                dataloader_strategy="batch",
                crop_model=None
            )

            # Get ground truth for this image
            gt_img = gt[gt['image'] == img_file].copy()

            # Create visualization using DeepForest's native function
            fig = visualize.plot_results(
                predictions=predictions,
                ground_truth=gt_img,
                image_path=img_path,
                show=False,
                thickness=1  # Thin lines
            )

            # Save high quality
            output_path = os.path.join(OUTPUT_DIR, f"viz_{img_file.replace('.JPG', '.png')}")
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            det_count = len(predictions) if predictions is not None else 0
            stats['total_detections'] += det_count
            if det_count > 0:
                stats['images_with_detections'] += 1
            stats['images_processed'] += 1

            print(f"✓ ({det_count} detections)")

        except Exception as e:
            print(f"✗ Error: {str(e)[:60]}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Images processed: {stats['images_processed']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
