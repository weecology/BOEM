#!/usr/bin/env python3
"""DeepForest precision/recall for Camera B.

Uses deepforest.evaluate.evaluate_boxes on the already-saved tiled predictions
(predictions.csv) versus the VIAME ground truth. Box metrics are class-agnostic,
so both sides are labeled 'Object' to match the detector's single class.
"""

import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepforest.evaluate import evaluate_boxes

IMAGE_DIR = "/blue/ewhite/b.weinstein/BOEM/NOAA/Camera B"
ANNOTATIONS_FILE = os.path.join(IMAGE_DIR, "annotations.viame.csv")
PRED_CSV = os.path.join(IMAGE_DIR, "predictions.csv")
IOU_THRESHOLD = 0.4  # DeepForest default


def load_ground_truth():
    rows = []
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 11:
                continue
            rows.append({
                'image_path': parts[1],
                'xmin': int(parts[3]), 'ymin': int(parts[4]),
                'xmax': int(parts[5]), 'ymax': int(parts[6]),
                'label': 'Object',
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print(f"DEEPFOREST BOX EVALUATION  (IoU >= {IOU_THRESHOLD})")
    print("=" * 70)

    gt = load_ground_truth()

    preds = pd.read_csv(PRED_CSV)
    # Group on the same key as GT: the JPG filename (saved in the 'image' column).
    preds['image_path'] = preds['image']
    preds['label'] = 'Object'
    preds = preds[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score']]

    print(f"\nGround-truth boxes : {len(gt)}  across {gt.image_path.nunique()} images")
    print(f"Predicted boxes    : {len(preds)}  across {preds.image_path.nunique()} images")

    results = evaluate_boxes(preds, gt, iou_threshold=IOU_THRESHOLD)

    precision = results['box_precision']
    recall = results['box_recall']
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else 0.0)

    res = results['results']
    tp = int(res['match'].sum()) if res is not None else 0

    print("\n" + "-" * 70)
    print(f"  Box recall     : {recall:.3f}   (fraction of GT dolphins detected)")
    print(f"  Box precision  : {precision:.3f}   (fraction of predictions that are correct)")
    print(f"  F1 score       : {f1:.3f}")
    print(f"  True positives : {tp}  /  {len(gt)} GT  |  {len(preds)} predictions")
    print("-" * 70)

    # Per-image breakdown
    if res is not None:
        print("\nPer-image matches (TP by image):")
        per_img = res.groupby('image_path')['match'].agg(['sum', 'count'])
        gt_counts = gt.groupby('image_path').size()
        for img in sorted(gt_counts.index):
            n_gt = gt_counts[img]
            n_pred = per_img.loc[img, 'count'] if img in per_img.index else 0
            n_tp = int(per_img.loc[img, 'sum']) if img in per_img.index else 0
            print(f"  {img:<45} GT={n_gt:>2}  Pred={n_pred:>2}  TP={n_tp:>2}")

    # Save a summary
    out = os.path.join(IMAGE_DIR, "deepforest_metrics.txt")
    with open(out, 'w') as f:
        f.write(f"DeepForest box evaluation (IoU>={IOU_THRESHOLD})\n")
        f.write(f"box_recall={recall:.4f}\nbox_precision={precision:.4f}\nf1={f1:.4f}\n")
        f.write(f"true_positives={tp}\nground_truth={len(gt)}\npredictions={len(preds)}\n")
    print(f"\n✓ Summary written to {out}")


if __name__ == "__main__":
    main()
