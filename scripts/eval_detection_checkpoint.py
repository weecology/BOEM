"""In-distribution (validation) box metrics for one detection checkpoint.

Scores the checkpoint on the validation CSV its training run monitored
(test_max_empty_0_25.csv from the same prepare_USGS split), so the number is
validation recall, NOT zero-shot -- zero-shot lives in threshold_sweep.py on the
pinned holdout flights. Reports DeepForest's own validate() metrics (at the
model's baked-in score_thresh) plus box recall/precision at the review thresholds
from one predict pass.

    uv run python scripts/eval_detection_checkpoint.py <checkpoint.ckpt> [val_csv]
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from deepforest import evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import detection  # noqa: E402

CROPS = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def is_empty(df: pd.DataFrame) -> pd.Series:
    return (df.xmin == 0) & (df.ymin == 0) & (df.xmax == 0) & (df.ymax == 0)


def main_(ckpt: str, val_csv: Path) -> None:
    m = detection.load(ckpt)  # re-applies a recorded --input-size
    m.config["workers"] = 8
    m.config["batch_size"] = 32
    m.config["validation"]["csv_file"] = str(val_csv)
    m.config["validation"]["root_dir"] = str(CROPS)
    m.config["validation"]["val_accuracy_interval"] = 1  # else box metrics are gated off at epoch 0
    m.config["validation"]["validate_coordinates"] = False
    m.create_trainer(accelerator="gpu", devices=1, strategy="auto", logger=False)

    gt = pd.read_csv(val_csv, low_memory=False)
    boxes = gt[~is_empty(gt)].copy()
    print(f"Validation set {val_csv.name}: {gt.image_path.nunique():,} tiles, "
          f"{boxes.image_path.nunique():,} with boxes, {len(boxes):,} boxes, "
          f"score_thresh baked into model = {m.model.score_thresh}, "
          f"input resize {m.model.transform.min_size}/{m.model.transform.max_size}", flush=True)

    metrics = m.trainer.validate(m)[0]
    print("\n=== DeepForest validate() (training-time definition) ===")
    for k in ("box_recall", "box_precision", "empty_frame_accuracy", "val_classification", "val_bbox_regression"):
        print(f"  {k}: {metrics.get(k, 'MISSING')}")

    preds = m.predict_file(csv_file=str(val_csv), root_dir=str(CROPS))
    preds["image_path"] = preds["image_path"].map(lambda p: Path(p).name)
    boxes["image_path"] = boxes["image_path"].map(lambda p: Path(p).name)
    # Single-class detector: GT carries species labels, which breaks evaluate_boxes'
    # per-class recall table (mixed int/str index). Box matching ignores label anyway.
    preds["label"] = boxes["label"] = "Object"
    box_tiles = set(boxes.image_path)
    empty_tiles = set(gt.image_path.map(lambda p: Path(p).name)) - box_tiles

    out = Path(ckpt).with_suffix(f".{val_csv.stem}_predictions.csv")
    preds.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # POOLED box recall = matched GT boxes / all GT boxes, the definition validate() and
    # threshold_sweep.py use. evaluate_boxes' own box_recall is a MEAN OF PER-TILE recall,
    # which lets one-bird tiles outvote dense flocks -- not comparable, so not reported.
    print("\n=== Pooled box recall / precision by score threshold (IoU 0.4) ===")
    print(" thresh   recall    prec      TP      FP   empty-tile FP rate")
    for t in THRESHOLDS:
        p = preds[preds.score >= t]
        res = evaluate.evaluate_boxes(p[p.image_path.isin(box_tiles)], boxes, iou_threshold=0.4)["results"]
        tp = int(res.match.sum())
        fp = len(p) - tp
        fp_empty = p[p.image_path.isin(empty_tiles)].image_path.nunique() / max(len(empty_tiles), 1)
        print(f"  {t:.2f}   {tp / len(boxes):6.1%}  {tp / max(len(p), 1):6.1%}  {tp:6,}  {fp:6,}   {fp_empty:6.1%}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main_(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else CROPS / "test_max_empty_0_25.csv")
