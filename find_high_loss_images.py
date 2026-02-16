#!/usr/bin/env python3
"""Find images with highest loss and upload to Label Studio for review."""

import os
import pandas as pd
import numpy as np
import torch
from deepforest import main
from omegaconf import DictConfig, OmegaConf
import yaml
from torchvision.ops import box_iou

# Import Label Studio utilities
from src import label_studio as ls_mod

def compute_loss_for_image(model, annotations_df, image_path, image_dir, iou_threshold=0.5):
    """Compute average detection loss for all annotations in an image.
    
    For detection models, we compute loss based on:
    - IoU between predictions and ground truth
    - False positives and false negatives
    
    Args:
        model: main.deepforest detection model instance
        annotations_df: DataFrame with annotations for this image (must have xmin, ymin, xmax, ymax, label)
        image_path: Path to the image file (relative to image_dir or absolute)
        image_dir: Directory containing images
        iou_threshold: IoU threshold for matching predictions to ground truth
        
    Returns:
        Average loss for the image (1 - mean IoU), or None if no valid annotations
    """
    # Get full path to image
    full_image_path = os.path.join(image_dir, image_path) if not os.path.isabs(image_path) else image_path
    
    if not os.path.exists(full_image_path):
        return None
    
    # Filter out invalid annotations
    valid_annos = annotations_df[
        (annotations_df['xmin'] >= 0) & (annotations_df['ymin'] >= 0) &
        (annotations_df['xmax'] > annotations_df['xmin']) &
        (annotations_df['ymax'] > annotations_df['ymin'])
    ].copy()
    
    if len(valid_annos) == 0:
        return None
    
    # Run detection predictions on the image
    predictions = model.predict_image(path=full_image_path)
    
    if predictions is None or len(predictions) == 0:
        # No predictions - high loss (all ground truth boxes are false negatives)
        return 1.0
    
    # Filter predictions for this image
    # The predictions DataFrame may have image_path as basename or full path
    image_basename = os.path.basename(image_path)
    preds = predictions[predictions['image_path'].str.endswith(image_basename, na=False)].copy()
    
    if len(preds) == 0:
        # No predictions for this image - high loss
        return 1.0
    
    # Convert to tensors for IoU computation
    gt_boxes = torch.tensor(valid_annos[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
    pred_boxes = torch.tensor(preds[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return None
    
    # Compute IoU matrix
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    
    # Match predictions to ground truth (greedy matching)
    matched_gt = set()
    matched_pred = set()
    ious = []
    
    # Sort by IoU descending
    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            iou = iou_matrix[gt_idx, pred_idx].item()
            if iou >= iou_threshold:
                ious.append(iou)
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
    
    # Compute loss components
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    num_matched = len(ious)
    
    # False negatives (unmatched ground truth)
    false_negatives = num_gt - num_matched
    # False positives (unmatched predictions)
    false_positives = num_pred - num_matched
    
    # Loss = 1 - mean IoU (if any matches), plus penalty for unmatched boxes
    if len(ious) > 0:
        mean_iou = np.mean(ious)
        # Penalty for unmatched boxes (normalized by total boxes)
        unmatched_penalty = (false_negatives + false_positives) / max(num_gt, num_pred, 1)
        loss = (1.0 - mean_iou) + unmatched_penalty
    else:
        # No matches - maximum loss
        loss = 1.0 + (false_negatives + false_positives) / max(num_gt, num_pred, 1)
    
    return loss


def find_high_loss_images(csv_dir, checkpoint_path, image_dir, n=10, top_n=100):
    """Find images with highest loss from train and test CSVs.
    
    Args:
        csv_dir: Directory containing train.csv and test.csv
        checkpoint_path: Path to model checkpoint
        image_dir: Directory containing images
        n: Number of images to process
        top_n: Number of top images to return
        
    Returns:
        DataFrame with top N images and their losses
    """
    # Load CSVs
    train_csv = os.path.join(csv_dir, 'train.csv')
    test_csv = os.path.join(csv_dir, 'test.csv')
    
    print(f"Loading CSVs from {csv_dir}")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Combine and get unique images
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    unique_images = all_df['image_path'].unique()

    # Limit to n images
    if n is not None:
        # Sample randomly
        unique_images = np.random.choice(unique_images, size=n, replace=False)
    print(f"Found {len(unique_images)} unique images")
    
    # Load detection model
    print(f"Loading detection model from: {checkpoint_path}")
    model = main.deepforest.load_from_checkpoint(checkpoint_path)
    model.eval()
    print(f"Model has {len(model.label_dict)} classes: {model.label_dict}")
    
    # Compute loss for each image
    image_losses = []
    for i, image_path in enumerate(unique_images):
        if (i + 1) % 100 == 0:
            print(f"Processing image {i+1}/{len(unique_images)}")
        
        # Get annotations for this image
        image_annos = all_df[all_df['image_path'] == image_path].copy()
        
        # Compute loss
        loss = compute_loss_for_image(model, image_annos, image_path, image_dir)
        if loss is not None:
            image_losses.append({
                'image_path': image_path,
                'avg_loss': loss,
                'n_annotations': len(image_annos)
            })
    
    # Create DataFrame and sort by loss
    loss_df = pd.DataFrame(image_losses)
    loss_df = loss_df.sort_values('avg_loss', ascending=False)
    
    # Return top N
    top_images = loss_df.head(top_n)
    print(f"\nTop {top_n} images with highest loss:")
    print(top_images[['image_path', 'avg_loss', 'n_annotations']].head(10))
    
    return top_images


def download_annotations(url, project_name, csv_dir, output_csv=None, delete_tasks=True):
    """Download completed annotations from Label Studio and save to CSV.
    
    Args:
        url: Label Studio server URL
        project_name: Label Studio project name
        csv_dir: Directory where Label Studio saves annotation CSVs (per-flight subdirectories)
        output_csv: Path to save consolidated CSV (default: fixed_annotations.csv in csv_dir)
        delete_tasks: Whether to delete completed tasks from Label Studio after downloading
        
    Returns:
        DataFrame with all downloaded annotations, or None if no annotations found
    """
    # Get API key and set environment variable
    api_key = ls_mod.get_api_key()
    if api_key is None:
        raise ValueError("Could not find Label Studio API key in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    
    # Connect to Label Studio
    label_studio_project = ls_mod.connect_to_label_studio(url=url, project_name=project_name)
    
    # Download completed tasks
    print(f"Downloading annotations from Label Studio project: {project_name}")
    annotations = ls_mod.download_completed_tasks(
        label_studio_project=label_studio_project,
        csv_dir=csv_dir
    )
    
    if annotations is None or len(annotations) == 0:
        print("No new annotations found")
        return None
    
    # Convert label 0 to "Bird" for empty images
    if 'label' in annotations.columns:
        # Handle both numeric 0 and string "0"
        annotations.loc[annotations['label'].isin([0, "0"]), 'label'] = "Bird"
    
    # Save consolidated CSV
    if output_csv is None:
        output_csv = os.path.join(csv_dir, 'fixed_annotations.csv')
    
    # Save annotations to CSV and verify it was written successfully
    annotations_saved = False
    try:
        # If file exists, append to it
        # Note: We keep all annotation rows (images can have multiple bounding boxes)
        # Duplicate checking happens when filtering images to upload
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            # Convert any 0 labels to "Bird" in existing file too
            if 'label' in existing_df.columns:
                existing_df.loc[existing_df['label'].isin([0, "0"]), 'label'] = "Bird"
            # Combine all annotations
            combined = pd.concat([existing_df, annotations], ignore_index=True)
            # Drop exact duplicate rows (same annotation data)
            combined = combined.drop_duplicates(keep='last')
            combined.to_csv(output_csv, index=False)
            print(f"Updated {output_csv} with {len(annotations)} new annotation rows")
            print(f"Total unique images in fixed annotations: {len(combined['image_path'].unique())}")
            print(f"Total annotation rows: {len(combined)}")
        else:
            annotations.to_csv(output_csv, index=False)
            print(f"Saved {len(annotations)} annotations to {output_csv}")
            print(f"Unique images: {len(annotations['image_path'].unique())}")
        
        # Verify the file was written successfully
        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            # Double-check by reading it back
            verify_df = pd.read_csv(output_csv)
            if len(verify_df) > 0:
                annotations_saved = True
                print(f"✓ Verified: Annotations successfully saved to {output_csv}")
            else:
                print(f"⚠ Warning: CSV file is empty, annotations may not have been saved correctly")
        else:
            print(f"⚠ Warning: CSV file was not created or is empty")
            
    except Exception as e:
        print(f"✗ Error saving annotations to CSV: {e}")
        print("Annotations will NOT be deleted from Label Studio to prevent data loss")
        return annotations
    
    # Only delete completed tasks if annotations were successfully saved
    if delete_tasks and annotations_saved:
        print("\nDeleting completed tasks from Label Studio...")
        try:
            ls_mod.delete_completed_tasks(label_studio_project)
            print("✓ Completed tasks deleted successfully")
        except Exception as e:
            print(f"✗ Error deleting completed tasks: {e}")
            print("Tasks remain in Label Studio - you can retry deletion later")
    elif delete_tasks and not annotations_saved:
        print("\n⚠ Skipping deletion: Annotations were not verified to be saved successfully")
        print("Tasks remain in Label Studio to prevent data loss")
    
    return annotations


def load_fixed_annotations(fixed_annotations_csv):
    """Load set of image paths that have already been fixed/annotated.
    
    Args:
        fixed_annotations_csv: Path to CSV file with fixed annotations
        
    Returns:
        Set of image_path values (basenames) that have been fixed
    """
    if not os.path.exists(fixed_annotations_csv):
        return set()
    
    df = pd.read_csv(fixed_annotations_csv)
    if 'image_path' not in df.columns:
        return set()
    
    # Return set of basenames (in case paths differ)
    return {os.path.basename(path) for path in df['image_path'].unique()}


def upload_to_label_studio(image_paths, image_dir, csv_dir, project_name, cfg, fixed_annotations_csv=None):
    """Upload images to Label Studio with their annotations as preannotations.
    
    Args:
        image_paths: List of image filenames (not full paths)
        image_dir: Directory containing images
        csv_dir: Directory containing train.csv and test.csv
        project_name: Label Studio project name
        cfg: Configuration dict for Label Studio
        fixed_annotations_csv: Path to CSV with already-fixed annotations (optional)
    """
    # Filter out already-processed images
    if fixed_annotations_csv:
        fixed_images = load_fixed_annotations(fixed_annotations_csv)
        original_count = len(image_paths)
        image_paths = [
            img for img in image_paths 
            if os.path.basename(img) not in fixed_images
        ]
        if len(image_paths) < original_count:
            print(f"Filtered out {original_count - len(image_paths)} already-processed images")
    
    if len(image_paths) == 0:
        print("No new images to upload (all have been processed)")
        return
    # Load server config
    server_cfg_path = os.path.join('boem_conf', 'server', 'serenity.yaml')
    if os.path.exists(server_cfg_path):
        with open(server_cfg_path) as f:
            server_cfg = yaml.safe_load(f)
    else:
        server_cfg = cfg.get('server', {
            'user': os.getenv('LABEL_STUDIO_USER', 'b.weinstein'),
            'host': os.getenv('LABEL_STUDIO_HOST', 'serenity.ifas.ufl.edu'),
            'key_filename': os.path.expanduser('~/.ssh/id_ed25519')
        })
    
    # Create SFTP client
    sftp_client = ls_mod.create_sftp_client(**server_cfg)
    
    # Prepare full image paths
    full_image_paths = [os.path.join(image_dir, img) for img in image_paths]
    
    # Load annotations for these images
    train_csv = os.path.join(csv_dir, 'train.csv')
    test_csv = os.path.join(csv_dir, 'test.csv')
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Prepare preannotations dict (keyed by basename)
    preannotations = {}
    for img_path in image_paths:
        img_annos = all_df[all_df['image_path'] == img_path].copy()
        if len(img_annos) > 0:
            # Format for Label Studio (needs label column, and optionally cropmodel_label)
            img_annos = img_annos.copy()
            if 'cropmodel_label' not in img_annos.columns:
                img_annos['cropmodel_label'] = img_annos['label']
            if 'score' not in img_annos.columns:
                img_annos['score'] = 0.5  # Placeholder score
            if 'comet_id' not in img_annos.columns:
                img_annos['comet_id'] = 'high_loss_review'
            preannotations[img_path] = img_annos
    
    # Get API key from .label_studio.config and set environment variable
    api_key = ls_mod.get_api_key()
    if api_key is None:
        raise ValueError("Could not find Label Studio API key in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    
    # Upload directly using label_studio module
    print(f"\nUploading {len(full_image_paths)} images to Label Studio project: {project_name}")
    ls_mod.upload_to_label_studio(
        images=full_image_paths,
        sftp_client=sftp_client,
        url=cfg['url'],
        project_name=project_name,
        images_to_annotate_dir=image_dir,
        folder_name=cfg['folder_name'],
        preannotations=preannotations if preannotations else None
    )
    print("Upload complete!")


def run():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find high loss images and upload to Label Studio')
    parser.add_argument('--csv_dir', type=str, 
                       default='/blue/ewhite/b.weinstein/bird_detector_retrain/2022paper_expanded',
                       help='Directory containing train.csv and test.csv')
    parser.add_argument('--checkpoint', type=str,
                       default='/blue/ewhite/b.weinstein/bird_detector_retrain/2022paper_expanded/checkpoints/51f3b6bccc314bd383bd792b54b18177.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str,
                       default='/blue/ewhite/b.weinstein/bird_detector_retrain/2022paper_expanded',
                       help='Directory containing images')
    parser.add_argument('--top_n', type=int, default=1,
                       help='Number of top images to select')
    parser.add_argument('--project_name', type=str,
                       default='Global Bird Detector',
                       help='Label Studio project name')
    parser.add_argument('--label_studio_config', type=str,
                       help='Path to Label Studio config YAML (optional, uses default if not provided)')
    parser.add_argument('--download_annotations', action='store_false',
                       help='Download completed annotations from Label Studio and save to CSV before uploading new ones')
    parser.add_argument('--no_delete_tasks', action='store_true',
                       help='Do not delete completed tasks from Label Studio after downloading (safer, keeps tasks as backup)')
    parser.add_argument('--fixed_annotations_csv', type=str,
                       default=None,
                       help='Path to CSV file with already-fixed annotations (default: csv_dir/fixed_annotations.csv)')
    
    args = parser.parse_args()
    
    # Load Label Studio config
    if args.label_studio_config and os.path.exists(args.label_studio_config):
        with open(args.label_studio_config) as f:
            ls_config = yaml.safe_load(f)['label_studio']
    else:
        # Use default config from boem_conf
        ls_config_path = 'boem_conf/annotation/label_studio.yaml'
        if os.path.exists(ls_config_path):
            with open(ls_config_path) as f:
                ls_config = yaml.safe_load(f)['label_studio']
        else:
            # Hardcoded defaults
            ls_config = {
                'url': 'https://labelstudio.naturecast.org/',
                'folder_name': '/media/T/lab-white-ernest/label_studio_data/BOEM',
                'server': {
                    'user': os.getenv('LABEL_STUDIO_USER', 'b.weinstein'),
                    'host': os.getenv('LABEL_STUDIO_HOST', 'serenity'),
                    'key_filename': os.getenv('LABEL_STUDIO_KEY', '~/.ssh/id_rsa')
                }
            }
    
    # Download annotations if requested
    if args.download_annotations:
        # Determine output CSV path
        if args.fixed_annotations_csv:
            output_csv = args.fixed_annotations_csv
        else:
            output_csv = os.path.join(args.csv_dir, 'fixed_annotations.csv')
        
        # Use csv_dir as the base directory for Label Studio annotation downloads
        # Label Studio saves to per-flight subdirectories, so we use csv_dir as the parent
        download_annotations(
            url=ls_config['url'],
            project_name=args.project_name,
            csv_dir=args.csv_dir,
            output_csv=output_csv,
            delete_tasks=not args.no_delete_tasks  # Delete unless --no_delete_tasks is set
        )
        print()  # Empty line for readability
    
    # Set default fixed annotations CSV path if not provided
    if args.fixed_annotations_csv is None:
        args.fixed_annotations_csv = os.path.join(args.csv_dir, 'fixed_annotations.csv')
    
    # Find high loss images
    top_images = find_high_loss_images(
        csv_dir=args.csv_dir,
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        top_n=args.top_n
    )
    
    # Get image paths
    image_paths = top_images['image_path'].tolist()
    
    # Upload to Label Studio (will filter out already-processed images)
    upload_to_label_studio(
        image_paths=image_paths,
        image_dir=args.image_dir,
        csv_dir=args.csv_dir,
        project_name=args.project_name,
        cfg=ls_config,
        fixed_annotations_csv=args.fixed_annotations_csv
    )
    
    # Save results to CSV
    output_csv = os.path.join(args.csv_dir, f'high_loss_images_top{args.top_n}.csv')
    top_images.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    run()
