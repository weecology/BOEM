"""Evaluate the pre-cleanup detection checkpoint against the current cleaned
test.csv and zero_shot.csv.

The May 21 commit "cleaning detection model" both fixed a label bug AND
switched the DeepForest pin. To separate "metric got honest" from "model got
worse", we evaluate the last pre-cleanup checkpoint
(0ce13c9d5a1448bea1a8e5c74b207413.pl, May 21 03:19) against the current
test/zero_shot data.

Run:
  uv run python scripts/eval_pre_cleanup_checkpoint.py

Requires GPU. Forces val_accuracy_interval=1 so the one-shot validate() call
at epoch 0 actually computes box metrics (otherwise DeepForest gates them).
"""

import os
import torch
from deepforest import main

CHECKPOINT = (
    "/blue/ewhite/b.weinstein/BOEM/training/"
    "checkpoints/0ce13c9d5a1448bea1a8e5c74b207413.pl"
)
CROPS = "/blue/ewhite/b.weinstein/BOEM/training/crops"


def evaluate(m, csv_name):
    csv_path = os.path.join(CROPS, csv_name)
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}, skipping")
        return
    m.config["validation"]["csv_file"] = csv_path
    m.config["validation"]["root_dir"] = CROPS
    m.config["validation"]["val_accuracy_interval"] = 1
    m.create_trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count() or 1,
        num_nodes=1,
        strategy="auto",
    )
    results = m.trainer.validate(m)
    metrics = results[0] if results else {}
    print(f"\n=== {csv_name} ===")
    for key in ("box_precision", "box_recall", "empty_frame_accuracy"):
        print(f"  {key}: {metrics.get(key, 'MISSING')}")
    return metrics


if __name__ == "__main__":
    print(f"Loading checkpoint: {CHECKPOINT}")
    m = main.deepforest.load_from_checkpoint(CHECKPOINT)
    m.config["workers"] = 4
    m.config["batch_size"] = 8
    print(f"Model label_dict: {m.label_dict}")
    print(f"Model num_classes: {m.config.get('num_classes')}")
    print(f"score_thresh: {m.config.get('score_thresh')}, "
          f"nms_thresh: {m.config.get('nms_thresh')}, "
          f"iou_threshold: {m.config['validation'].get('iou_threshold')}")

    evaluate(m, "test.csv")
    evaluate(m, "zero_shot.csv")
