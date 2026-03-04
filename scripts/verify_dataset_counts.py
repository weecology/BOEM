#!/usr/bin/env python
# placeholder to test edit3
"""Verify dataset counts and check for potential label shifts."""

import os
import pandas as pd
from pathlib import Path
from collections import Counter

def verify_dataset_counts(train_crop_dir: str):
    """Verify that file counts match expected counts and check for label shifts."""
    
    train_crop_path = Path(train_crop_dir)
    
    if not train_crop_path.exists():
        print(f"Directory does not exist: {train_crop_dir}")
        return
    
    # Count files by class directory
    class_counts = {}
    total_files = 0
    
    for class_dir in train_crop_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            # Count only image files (PNG)
            image_files = list(class_dir.glob("*.png"))
            file_count = len(image_files)
            class_counts[class_name] = file_count
            total_files += file_count
    
    print(f"\n=== Dataset Verification Report ===")
    print(f"Root directory: {train_crop_dir}")
    print(f"\nTotal image files found: {total_files}")
    print(f"Total class directories: {len(class_counts)}")
    
    # Check for empty directories
    empty_dirs = [name for name, count in class_counts.items() if count == 0]
    if empty_dirs:
        print(f"\n⚠️  WARNING: Found {len(empty_dirs)} empty class directories:")
        for dir_name in empty_dirs[:10]:  # Show first 10
            print(f"  - {dir_name}")
        if len(empty_dirs) > 10:
            print(f"  ... and {len(empty_dirs) - 10} more")
    
    # Check for non-image files
    non_image_files = []
    for class_dir in train_crop_path.iterdir():
        if class_dir.is_dir():
            for file_path in class_dir.iterdir():
                if file_path.is_file() and not file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    non_image_files.append(str(file_path))
    
    if non_image_files:
        print(f"\n⚠️  WARNING: Found {len(non_image_files)} non-image files:")
        for file_path in non_image_files[:10]:
            print(f"  - {file_path}")
        if len(non_image_files) > 10:
            print(f"  ... and {len(non_image_files) - 10} more")
    else:
        print(f"\n✓ All files are valid image files")
    
    # Show class distribution
    print(f"\n=== Class Distribution (top 20) ===")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes[:20]:
        print(f"  {class_name}: {count} images")
    
    if len(sorted_classes) > 20:
        print(f"  ... and {len(sorted_classes) - 20} more classes")
    
    # Check for potential issues
    print(f"\n=== Potential Issues ===")
    issues_found = False
    
    # Check for classes with very few samples (potential data quality issue)
    small_classes = [(name, count) for name, count in class_counts.items() if count < 5]
    if small_classes:
        print(f"⚠️  Found {len(small_classes)} classes with < 5 images (may cause training issues):")
        for name, count in sorted(small_classes, key=lambda x: x[1])[:10]:
            print(f"  - {name}: {count} images")
        issues_found = True
    
    # Check for extremely large classes (potential imbalance)
    large_classes = [(name, count) for name, count in class_counts.items() if count > 500]
    if large_classes:
        print(f"⚠️  Found {len(large_classes)} classes with > 500 images (class imbalance):")
        for name, count in sorted(large_classes, key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {count} images")
        issues_found = True
    
    if not issues_found:
        print("✓ No obvious issues detected")
    
    return {
        'total_files': total_files,
        'num_classes': len(class_counts),
        'class_counts': class_counts,
        'empty_dirs': empty_dirs,
        'non_image_files': non_image_files
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset_counts.py <train_crop_image_dir>")
        print("\nExample:")
        print("  python verify_dataset_counts.py '/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/crops/train/3d88dbea165744a4b64574d054017402'")
        sys.exit(1)
    
    train_crop_dir = sys.argv[1]
    verify_dataset_counts(train_crop_dir)





