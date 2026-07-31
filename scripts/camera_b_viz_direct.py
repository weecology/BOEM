#!/usr/bin/env python3
"""Full-resolution detection overlays for Camera B.

Draws boxes directly onto the original pixel array with OpenCV (no matplotlib
re-render), so output keeps the native 6464x4852 resolution for zooming.
Also writes predictions.csv so overlays can be regenerated without re-running
the model.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import detection

IMAGE_DIR = "/blue/ewhite/b.weinstein/BOEM/NOAA/Camera B"
ANNOTATIONS_FILE = os.path.join(IMAGE_DIR, "annotations.viame.csv")
OUTPUT_DIR = os.path.join(IMAGE_DIR, "visualizations")
PRED_CSV = os.path.join(IMAGE_DIR, "predictions.csv")

DETECTION_CHECKPOINT = "/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt"

# BGR colors (OpenCV). Thin lines relative to a ~6500px-wide image.
GT_COLOR = (0, 255, 0)       # green
PRED_COLOR = (255, 255, 0)   # cyan
LINE_THICKNESS = 3
JPG_QUALITY = 95


def load_annotations():
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
                'image': parts[1],
                'xmin': int(parts[3]), 'ymin': int(parts[4]),
                'xmax': int(parts[5]), 'ymax': int(parts[6]),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("CAMERA B FULL-RESOLUTION DETECTION OVERLAYS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt = load_annotations()

    print("\nLoading detection model...")
    model = detection.load(DETECTION_CHECKPOINT)
    model.config["batch_size"] = 64
    model.config["workers"] = 5
    print("✓ Model loaded\n")

    image_files = sorted(f for f in os.listdir(IMAGE_DIR)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    all_preds = []
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(IMAGE_DIR, img_file)
        print(f"[{idx:2d}/{len(image_files)}] {img_file}", end=" ... ", flush=True)

        try:
            predictions = model.predict_tile(
                path=[img_path],
                patch_size=1000,
                patch_overlap=0,
                dataloader_strategy="batch",
                crop_model=None,
            )

            # Draw on the FULL-RESOLUTION array (BGR, as read).
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            gt_img = gt[gt['image'] == img_file]
            for _, r in gt_img.iterrows():
                cv2.rectangle(img, (r['xmin'], r['ymin']), (r['xmax'], r['ymax']),
                              GT_COLOR, LINE_THICKNESS)

            det_count = 0
            if predictions is not None and len(predictions) > 0:
                for _, p in predictions.iterrows():
                    cv2.rectangle(
                        img,
                        (int(p['xmin']), int(p['ymin'])),
                        (int(p['xmax']), int(p['ymax'])),
                        PRED_COLOR, LINE_THICKNESS,
                    )
                    det_count += 1
                    rec = p.to_dict()
                    rec['image'] = img_file
                    all_preds.append(rec)

            # Full-res JPG (keeps native WxH; quality 95 keeps files ~5-10MB).
            out_file = os.path.join(OUTPUT_DIR, f"viz_{os.path.splitext(img_file)[0]}.jpg")
            cv2.imwrite(out_file, img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

            print(f"✓ {w}x{h}  (GT {len(gt_img)}, Det {det_count})")
        except Exception as e:
            print(f"✗ {str(e)[:60]}")

    if all_preds:
        pd.DataFrame(all_preds).to_csv(PRED_CSV, index=False)
        print(f"\n✓ Predictions written to {PRED_CSV}")

    print("\n" + "=" * 80)
    print(f"✓ COMPLETE — full-resolution overlays in {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
