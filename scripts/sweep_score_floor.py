"""Sweep the detection score floor on specific frames and report matched recall/precision.

Predictions are run ONCE at a low floor and cached to CSV, then thresholded offline, so
the curve can be re-analysed without a GPU. Matching is greedy, highest-score-first,
one-to-one, at a configurable IoU.

Also reports the score distribution of matched (TP) vs unmatched (FP) boxes, which is
what determines whether ANY floor can separate birds from foam.
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    u = aa[:, None] + bb[None, :] - inter
    return np.where(u > 0, inter / u, 0.0)


def match(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    """Greedy highest-score-first one-to-one matching. Returns bool array of TP flags."""
    tp = np.zeros(len(pred_boxes), bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_boxes, gt_boxes)
    taken = set()
    for pi in order:
        row = ious[pi].copy()
        for gi in taken:
            row[gi] = -1
        gi = int(np.argmax(row))
        if row[gi] >= iou_thresh:
            tp[pi] = True
            taken.add(gi)
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch", type=int, default=1000)
    ap.add_argument("--base-score", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--pred-cache", default=None,
                    help="reuse a predictions CSV instead of re-running the model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt_all = pd.read_csv(args.gt_csv)
    cache = args.pred_cache or os.path.join(args.out_dir, "predictions.csv")

    if os.path.exists(cache):
        preds_all = pd.read_csv(cache)
        print(f"reusing predictions from {cache}", flush=True)
    else:
        from deepforest import main as df_main
        m = df_main.deepforest.load_from_checkpoint(args.checkpoint, map_location="cpu")
        if torch.cuda.is_available():
            m.to("cuda")
        m.eval()
        m.model.score_thresh = args.base_score
        m.config["score_thresh"] = args.base_score
        m.config["nms_thresh"] = args.nms
        frames = []
        for name in args.images:
            p = os.path.join(args.image_dir, name)
            r = m.predict_tile(path=p, patch_size=args.patch, patch_overlap=0.0,
                               iou_threshold=args.nms)
            if r is None or len(r) == 0:
                continue
            r = r[["xmin", "ymin", "xmax", "ymax", "score"]].copy()
            r["image_path"] = name
            frames.append(r)
            print(f"{name}: {len(r)} boxes at floor {args.base_score}", flush=True)
        preds_all = pd.concat(frames, ignore_index=True)
        preds_all.to_csv(cache, index=False)
        print("wrote", cache, flush=True)

    # ---- match once at the base floor, then threshold offline ----
    tagged = []
    for name in args.images:
        g = gt_all[gt_all.image_path == name][["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
        p = preds_all[preds_all.image_path == name]
        pb = p[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
        ps = p["score"].values.astype(float)
        tp = match(pb, ps, g, args.match_iou)
        d = p.copy(); d["tp"] = tp; d["n_gt"] = len(g)
        tagged.append(d)
    tagged = pd.concat(tagged, ignore_index=True)
    tagged.to_csv(os.path.join(args.out_dir, "predictions_matched.csv"), index=False)

    print(f"\n=== TP vs FP score distribution (IoU {args.match_iou}) ===")
    for name, d in tagged.groupby("image_path"):
        t, f = d[d.tp].score, d[~d.tp].score
        print(f"{name}")
        print(f"   TP n={len(t):4d}  median={t.median():.3f}  p90={t.quantile(.9):.3f}  max={t.max():.3f}")
        print(f"   FP n={len(f):4d}  median={f.median():.3f}  p90={f.quantile(.9):.3f}  max={f.max():.3f}")

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for name, d in tagged.groupby("image_path"):
        n_gt = int(d.n_gt.iloc[0])
        for t in grid:
            s = d[d.score >= t]
            tp = int(s.tp.sum()); npred = len(s)
            rec = tp / n_gt if n_gt else np.nan
            prec = tp / npred if npred else np.nan
            f1 = 2 * rec * prec / (rec + prec) if npred and tp else 0.0
            rows.append({"image": name, "floor": t, "n_gt": n_gt, "n_pred": npred,
                         "tp": tp, "recall": rec, "precision": prec, "f1": f1})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "floor_sweep.csv"), index=False)

    for name, d in df.groupby("image"):
        print(f"\n=== {name} (n_gt={int(d.n_gt.iloc[0])}) ===")
        print(d[["floor", "n_pred", "tp", "recall", "precision", "f1"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        b = d.loc[d.f1.idxmax()]
        print(f"  best F1 at floor {b.floor:.2f}: recall {b.recall:.1%}, precision {b.precision:.1%}, F1 {b.f1:.3f}")

    pooled = (df.groupby("floor")
                .agg(n_gt=("n_gt", "sum"), n_pred=("n_pred", "sum"), tp=("tp", "sum"))
                .reset_index())
    pooled["recall"] = pooled.tp / pooled.n_gt
    pooled["precision"] = pooled.tp / pooled.n_pred.replace(0, np.nan)
    pooled["f1"] = 2 * pooled.recall * pooled.precision / (pooled.recall + pooled.precision)
    pooled.to_csv(os.path.join(args.out_dir, "floor_sweep_pooled.csv"), index=False)
    print("\n=== POOLED ===")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    b = pooled.loc[pooled.f1.idxmax()]
    print(f"\nbest pooled F1 at floor {b.floor:.2f}: "
          f"recall {b.recall:.1%}, precision {b.precision:.1%}, F1 {b.f1:.3f}")


if __name__ == "__main__":
    main()
