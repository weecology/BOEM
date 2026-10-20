"""Probe whether a hard detection cap binds on dense-flock imagery.

There are four candidate ceilings, at two different levels:

INFERENCE (torchvision RetinaNet.postprocess_detections, per FORWARD-PASS IMAGE --
which under predict_tile means per PATCH, not per full frame):
  1. score_thresh        -- drops candidates before anything else
  2. topk_candidates=1000-- per FEATURE LEVEL, applied after score_thresh
  3. nms_thresh=0.05     -- IoU suppression
  4. detections_per_img=300 -- hard truncation of the post-NMS list

VALIDATION (torchmetrics MeanAveragePrecision):
  5. max_detection_thresholds=[1,10,100] -- mAP/mAR only score the top 100 boxes
     per image. Handled separately in the mAP section below.

This script hooks postprocess_detections to record, per patch, how many candidates
survived each stage, so cap saturation is measured rather than assumed.
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from deepforest import main as df_main

NON_TARGET_LABELS = {"FalsePositive"}


def load_gt(flock_dir):
    frames = []
    for name in ("train_annotations.csv", "test_annotations.csv"):
        p = os.path.join(flock_dir, name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    gt = pd.concat(frames, ignore_index=True)
    gt = gt[~gt["label"].isin(NON_TARGET_LABELS)]
    return gt.dropna(subset=["xmin", "ymin", "xmax", "ymax"])


def install_probe(model, records):
    """Wrap postprocess_detections to count candidates at every stage, per patch."""
    from torchvision.models.detection import _utils as det_utils
    from torchvision.ops import boxes as box_ops

    def probed(head_outputs, anchors, image_shapes):
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        detections = []
        for index in range(len(image_shapes)):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes, image_scores, image_labels = [], [], []
            n_above_thresh = 0
            n_after_topk = 0
            levels_saturating_topk = 0
            for box_reg_lvl, logits_lvl, anchors_lvl in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_lvl.shape[-1]
                scores_lvl = torch.sigmoid(logits_lvl).flatten()
                keep_idxs = scores_lvl > model.score_thresh
                scores_lvl = scores_lvl[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]
                n_above_thresh += int(topk_idxs.numel())
                if int(topk_idxs.numel()) >= model.topk_candidates:
                    levels_saturating_topk += 1

                num_topk = det_utils._topk_min(topk_idxs, model.topk_candidates, 0)
                scores_lvl, idxs = scores_lvl.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                n_after_topk += int(num_topk)

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_lvl = topk_idxs % num_classes
                boxes_lvl = model.box_coder.decode_single(
                    box_reg_lvl[anchor_idxs], anchors_lvl[anchor_idxs]
                )
                boxes_lvl = box_ops.clip_boxes_to_image(boxes_lvl, image_shape)
                image_boxes.append(boxes_lvl)
                image_scores.append(scores_lvl)
                image_labels.append(labels_lvl)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, model.nms_thresh
            )
            n_after_nms = int(keep.numel())
            keep = keep[: model.detections_per_img]
            n_final = int(keep.numel())

            records.append({
                "n_above_score_thresh": n_above_thresh,
                "n_after_topk": n_after_topk,
                "levels_saturating_topk": levels_saturating_topk,
                "n_after_nms": n_after_nms,
                "n_final": n_final,
                "nms_dropped": n_after_topk - n_after_nms,
                "cap_dropped": n_after_nms - n_final,
                "hit_detections_per_img": n_after_nms > model.detections_per_img,
            })
            detections.append({
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            })
        return detections

    model.postprocess_detections = probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--flock-dir", default="/blue/ewhite/b.weinstein/src/BOEM/flock_dataset")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch", type=int, default=1000)
    ap.add_argument("--patch-overlap", type=float, default=0.0)
    ap.add_argument("--score", type=float, default=0.05,
                    help="score floor; low values stress the caps hardest")
    ap.add_argument("--nms", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_dir = os.path.join(args.flock_dir, "images")
    gt = load_gt(args.flock_dir)
    counts = gt.groupby("image_path").size().sort_values(ascending=False)
    names = [n for n in counts.index if os.path.exists(os.path.join(image_dir, n))][: args.n_images]

    m = df_main.deepforest.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        m.to("cuda")
    m.eval()

    print("=== caps as loaded from the checkpoint ===")
    for k in ("score_thresh", "nms_thresh", "detections_per_img", "topk_candidates"):
        print(f"  live model.{k:22s} = {getattr(m.model, k, 'MISSING')}")
        print(f"  config[{k!r}]".ljust(34) + f" = {m.config.get(k, 'MISSING')}")
    print(flush=True)

    all_rows = []
    for label, det_cap, topk_cap in [
        ("default_caps", 300, 1000),
        ("raised_caps", 20000, 20000),
    ]:
        m.model.detections_per_img = det_cap
        m.model.topk_candidates = topk_cap
        m.model.score_thresh = args.score
        m.config["score_thresh"] = args.score
        m.config["nms_thresh"] = args.nms
        m.config["detections_per_img"] = det_cap
        m.config["topk_candidates"] = topk_cap

        for name in names:
            recs = []
            install_probe(m.model, recs)
            preds = m.predict_tile(
                path=os.path.join(image_dir, name),
                patch_size=args.patch,
                patch_overlap=args.patch_overlap,
                iou_threshold=args.nms,
            )
            n_pred = 0 if preds is None else len(preds)
            n_gt = int(counts[name])
            r = pd.DataFrame(recs)
            row = {
                "setting": label, "image": name, "n_gt": n_gt,
                "n_pred_final_image": n_pred,
                "n_patches": len(r),
                "patch_max_above_thresh": int(r.n_above_score_thresh.max()),
                "patch_max_after_nms": int(r.n_after_nms.max()),
                "patch_max_final": int(r.n_final.max()),
                "patches_hitting_det_cap": int(r.hit_detections_per_img.sum()),
                "patches_saturating_topk": int((r.levels_saturating_topk > 0).sum()),
                "total_dropped_by_det_cap": int(r.cap_dropped.sum()),
                "total_dropped_by_nms": int(r.nms_dropped.sum()),
            }
            all_rows.append(row)
            print(f"[{label}] {name}: gt={n_gt} final={n_pred} "
                  f"patches={row['n_patches']} max/patch={row['patch_max_final']} "
                  f"hit_det_cap={row['patches_hitting_det_cap']} "
                  f"topk_saturated={row['patches_saturating_topk']} "
                  f"nms_dropped={row['total_dropped_by_nms']}", flush=True)
            r.to_csv(os.path.join(args.out_dir, f"patches_{label}_{name}.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "cap_probe.csv"), index=False)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False), flush=True)

    piv = df.pivot(index="image", columns="setting", values="n_pred_final_image")
    piv["delta_from_raising_caps"] = piv.get("raised_caps", 0) - piv.get("default_caps", 0)
    print("\n=== boxes gained by raising detections_per_img/topk_candidates ===")
    print(piv.to_string(), flush=True)
    piv.to_csv(os.path.join(args.out_dir, "cap_delta.csv"))


if __name__ == "__main__":
    main()
