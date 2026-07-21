"""Sweep NMS / score / patch_size on dense flock images and measure recall vs ground truth.

Motivation: config.nms_thresh defaults to 0.05, which suppresses any box overlapping
another by >5% IoU. On dense flocks (birds packed shoulder to shoulder) this can
suppress most true detections regardless of score threshold.

Two distinct NMS stages exist and are set differently:
  1. per-patch, inside the torchvision RetinaNet -> config.nms_thresh
     (predict_tile re-assigns model.nms_thresh from config at main.py:607, so
      setting model.nms_thresh directly does NOT stick)
  2. cross-patch, in predict.mosaic -> predict_tile(iou_threshold=...), default 0.15
     (only matters at patch seams when patch_overlap > 0)

score_thresh / detections_per_img / topk_candidates are read from the live model at
postprocess time and are NOT re-synced from config, so they must be set on m.model.
"""
import argparse
import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from deepforest import main as df_main

# Annotation labels that are not real birds.
NON_TARGET_LABELS = {"FalsePositive"}


def load_gt(flock_dir):
    frames = []
    for name in ("train_annotations.csv", "test_annotations.csv"):
        p = os.path.join(flock_dir, name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    gt = pd.concat(frames, ignore_index=True)
    gt = gt[~gt["label"].isin(NON_TARGET_LABELS)]
    gt = gt.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    return gt


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_recall(pred_boxes, gt_boxes, iou_thresh=0.4):
    """Greedy one-to-one matching; returns (n_matched, recall, precision)."""
    if len(gt_boxes) == 0:
        return 0, float("nan"), float("nan")
    if len(pred_boxes) == 0:
        return 0, 0.0, float("nan")
    ious = iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    matched_pred = set()
    # greedy by descending IoU
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for pi, gi in order:
        if ious[pi, gi] < iou_thresh:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(int(pi))
        matched_gt.add(int(gi))
    n = len(matched_gt)
    return n, n / len(gt_boxes), n / len(pred_boxes)


def gt_crowding(gt_boxes):
    """Fraction of GT boxes whose max IoU with another GT box exceeds each threshold.

    This is the key diagnostic: if GT boxes routinely overlap each other by more
    than nms_thresh, NMS *must* delete true positives no matter how good the model is.
    """
    if len(gt_boxes) < 2:
        return {}
    ious = iou_matrix(gt_boxes, gt_boxes)
    np.fill_diagonal(ious, 0.0)
    nn = ious.max(axis=1)
    return {f"gt_frac_maxiou_gt_{t}": float((nn > t).mean()) for t in (0.05, 0.15, 0.3, 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--flock-dir", default="/blue/ewhite/b.weinstein/src/BOEM/flock_dataset")
    ap.add_argument("--n-images", type=int, default=8, help="densest N images to test")
    ap.add_argument("--images", nargs="*", default=None, help="explicit image basenames")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nms", nargs="*", type=float, default=[0.05, 0.15, 0.3, 0.5])
    ap.add_argument("--score", nargs="*", type=float, default=[0.1, 0.3, 0.85])
    ap.add_argument("--patch", nargs="*", type=int, default=[1000, 500])
    ap.add_argument("--patch-overlap", type=float, default=0.0)
    ap.add_argument("--detections-per-img", type=int, default=2000)
    ap.add_argument("--topk-candidates", type=int, default=4000)
    ap.add_argument("--match-iou", type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_dir = os.path.join(args.flock_dir, "images")

    gt = load_gt(args.flock_dir)
    counts = gt.groupby("image_path").size().sort_values(ascending=False)

    if args.images:
        names = args.images
    else:
        names = [n for n in counts.index if os.path.exists(os.path.join(image_dir, n))]
        names = names[: args.n_images]

    print(f"Testing {len(names)} images", flush=True)

    # GT crowding report -- independent of the model
    crowd_rows = []
    for n in names:
        g = gt[gt.image_path == n][["xmin", "ymin", "xmax", "ymax"]].values
        row = {"image": n, "n_gt": len(g)}
        row.update(gt_crowding(g))
        crowd_rows.append(row)
        print(f"  {n}: n_gt={len(g)} " +
              " ".join(f"{k.replace('gt_frac_maxiou_gt_','iou>')}={v:.2%}"
                       for k, v in list(row.items())[2:]), flush=True)
    pd.DataFrame(crowd_rows).to_csv(os.path.join(args.out_dir, "gt_crowding.csv"), index=False)

    m = df_main.deepforest.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        m.to("cuda")
    m.eval()

    # Raise the per-patch caps so they are never the binding constraint.
    m.model.detections_per_img = args.detections_per_img
    m.model.topk_candidates = args.topk_candidates
    m.config["detections_per_img"] = args.detections_per_img
    m.config["topk_candidates"] = args.topk_candidates

    rows = []
    for nms, score, patch in itertools.product(args.nms, args.score, args.patch):
        # nms_thresh MUST go through config: predict_tile re-assigns model.nms_thresh
        # from config every call (main.py:607).
        m.config["nms_thresh"] = nms
        # score_thresh is read from the live model and never re-synced from config.
        m.model.score_thresh = score
        m.config["score_thresh"] = score

        for n in names:
            p = os.path.join(image_dir, n)
            try:
                preds = m.predict_tile(
                    path=p,
                    patch_size=patch,
                    patch_overlap=args.patch_overlap,
                    iou_threshold=nms,  # cross-patch mosaic NMS, keep consistent
                )
            except Exception as e:
                print(f"  FAIL nms={nms} score={score} patch={patch} {n}: {type(e).__name__} {e}", flush=True)
                continue
            npred = 0 if preds is None else len(preds)
            g = gt[gt.image_path == n][["xmin", "ymin", "xmax", "ymax"]].values
            if npred:
                pb = preds[["xmin", "ymin", "xmax", "ymax"]].values
            else:
                pb = np.zeros((0, 4))
            nmatch, rec, prec = match_recall(pb, g, iou_thresh=args.match_iou)
            rows.append({
                "image": n, "nms_thresh": nms, "score_thresh": score,
                "patch_size": patch, "n_pred": npred, "n_gt": len(g),
                "n_matched": nmatch, "recall": rec, "precision": prec,
            })
            print(f"nms={nms} score={score} patch={patch} {n}: "
                  f"pred={npred} gt={len(g)} recall={rec:.2%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "sweep_results.csv"), index=False)

    if len(df):
        agg = (df.groupby(["nms_thresh", "score_thresh", "patch_size"])
                 .agg(n_pred=("n_pred", "sum"), n_gt=("n_gt", "sum"),
                      n_matched=("n_matched", "sum"))
                 .reset_index())
        agg["recall"] = agg.n_matched / agg.n_gt
        agg["precision"] = agg.n_matched / agg.n_pred.replace(0, np.nan)
        agg = agg.sort_values("recall", ascending=False)
        agg.to_csv(os.path.join(args.out_dir, "sweep_summary.csv"), index=False)
        print("\n=== SUMMARY (pooled over images, sorted by recall) ===", flush=True)
        print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
