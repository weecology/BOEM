"""Find predict.min_score for a detection checkpoint.

min_score is a percentile of one model's score distribution, not a portable
constant. Job 39271997-39272145 inherited 0.85 from a differently-calibrated
checkpoint and returned 100 boxes from 533,876 images; the same checkpoint
never exceeds 0.798 on a flight where the previous one scored 406 boxes. This
script re-derives the threshold per checkpoint from labelled data.

Two measurements, because the operating point is a trade:

  RECALL  runs the detector over human-reviewed full frames from the pinned
          zero-shot holdout flights and matches boxes to ground truth by IoU.
          Answers "what fraction of real objects clears the threshold".

  QUEUE   runs the detector over a random sample of unscreened frames from the
          same flights and counts what clears the threshold. Answers "how many
          images per 1,000 land in Label Studio", i.e. the annotator cost.

The reviewed frames were originally SELECTED because some detector flagged
them, so they under-represent empty ocean. Recall is unaffected (a human marked
every object in each frame, including ones the detector missed), but precision
and FP-per-image measured on them are optimistic. That is exactly why QUEUE is
sampled separately from unscreened imagery.

Writes one JSON per checkpoint; summarise with --report.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detection  # noqa: E402

IMAGERY = "/blue/ewhite/b.weinstein/BOEM/imagery"
ANNOTATIONS = "/blue/ewhite/b.weinstein/BOEM/annotations"

# Pinned zero-shot holdout (submit_prepare_annotations.sh). Detection runs
# 38834235 and 39211658 never trained on these two flights.
HOLDOUT = ["JPG_20260202_141900", "JPG_20260201_134000"]

# Human-marked non-objects. They are not ground-truth positives, but they are
# not ordinary background either -- a human looked and said "no". Held out of
# both the positive set and the FP count so they bias neither metric.
NEGATIVE_LABELS = {"FalsePositive"}

THRESHOLDS = [round(float(x), 3) for x in np.arange(0.05, 1.0, 0.05)] + [0.975, 0.99]


def load_ground_truth(flights):
    """Human annotations for the holdout flights, keyed by image basename."""
    frames = []
    for flight in flights:
        paths = glob.glob(os.path.join(ANNOTATIONS, "*", flight, "*.csv"))
        if not paths:
            continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        df = df[df["flight_name"] == flight]
        df["flight"] = flight
        frames.append(df)
    if not frames:
        raise SystemExit(f"no annotations found for {flights}")
    gt = pd.concat(frames, ignore_index=True)

    gt["basename"] = gt["image_path"].map(lambda p: os.path.basename(str(p)))
    # Degenerate rows carry an empty-image marker, not a box.
    gt = gt[(gt["xmax"] - gt["xmin"] > 0) & (gt["ymax"] - gt["ymin"] > 0)]
    gt["is_negative"] = gt["label"].isin(NEGATIVE_LABELS)
    return gt


def resolve(gt):
    """Absolute image paths for annotated frames that exist on disk."""
    paths, missing = [], 0
    for (flight, basename) in gt[["flight", "basename"]].drop_duplicates().itertuples(index=False):
        p = os.path.join(IMAGERY, flight, basename)
        if os.path.exists(p):
            paths.append(p)
        else:
            missing += 1
    return sorted(paths), missing


def sample_unscreened(flights, annotated_basenames, n, seed=0):
    """Random frames from the holdout flights that were never human-reviewed."""
    rng = random.Random(seed)
    pool = []
    for flight in flights:
        for p in glob.glob(os.path.join(IMAGERY, flight, "*.jpg")):
            if os.path.basename(p) not in annotated_basenames:
                pool.append(p)
    rng.shuffle(pool)
    return sorted(pool[:n])


def predict(checkpoint, image_paths, patch_size, patch_overlap, batch_size, workers, floor):
    m = detection.load(checkpoint)
    # Let every box through; the sweep does the thresholding offline. Without
    # this the model's own score_thresh silently truncates the low end of the
    # curve and every recall number below it is wrong.
    try:
        m.model.score_thresh = floor
    except AttributeError:
        pass
    if hasattr(m, "config") and isinstance(m.config, dict):
        m.config["score_thresh"] = floor

    preds = detection.predict(
        m, image_paths,
        patch_size=patch_size, patch_overlap=patch_overlap,
        crop_model=None, batch_size=batch_size, workers=workers,
    )
    if preds is None or len(preds) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "image_path"])
    preds["basename"] = preds["image_path"].map(lambda p: os.path.basename(str(p)))
    return preds


def match(pred_img, gt_img, iou_thresh):
    """Greedy highest-score-first IoU matching. Returns (n_matched_gt, n_fp)."""
    if len(gt_img) == 0:
        return 0, len(pred_img)
    if len(pred_img) == 0:
        return 0, 0
    p = torch.tensor(pred_img[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
    g = torch.tensor(gt_img[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
    ious = box_iou(p, g).numpy()

    order = np.argsort(-pred_img["score"].values)
    taken_gt, matched, fp = set(), 0, 0
    for pi in order:
        cand = [(ious[pi, gi], gi) for gi in range(len(gt_img))
                if gi not in taken_gt and ious[pi, gi] >= iou_thresh]
        if cand:
            taken_gt.add(max(cand)[1])
            matched += 1
        else:
            fp += 1
    return matched, fp


def sweep_recall(preds, gt, image_paths, iou_thresh):
    """Recall / precision / FP-per-image across thresholds on labelled frames."""
    pos = gt[~gt["is_negative"]]
    neg_basenames = set(gt[gt["is_negative"]]["basename"])
    gt_by_img = {b: g for b, g in pos.groupby("basename")}
    all_basenames = [os.path.basename(p) for p in image_paths]
    n_gt = len(pos)

    rows = []
    for t in THRESHOLDS:
        kept = preds[preds["score"] >= t]
        by_img = {b: g for b, g in kept.groupby("basename")}
        tp = fp = 0
        imgs_surfaced = 0
        imgs_with_gt_surfaced = 0
        for b in all_basenames:
            pi = by_img.get(b, preds.iloc[0:0])
            gi = gt_by_img.get(b, pos.iloc[0:0])
            m, f = match(pi, gi, iou_thresh)
            tp += m
            # A box on a human-marked FalsePositive frame is not counted either way.
            if b not in neg_basenames:
                fp += f
            if len(pi) > 0:
                imgs_surfaced += 1
                if len(gi) > 0:
                    imgs_with_gt_surfaced += 1
        n_img_with_gt = sum(1 for b in all_basenames if b in gt_by_img)
        rows.append({
            "threshold": t,
            "tp": tp, "fp": fp, "n_gt": n_gt,
            "box_recall": tp / n_gt if n_gt else None,
            "box_precision": tp / (tp + fp) if (tp + fp) else None,
            "fp_per_image": fp / len(all_basenames),
            "image_recall": imgs_with_gt_surfaced / n_img_with_gt if n_img_with_gt else None,
            "images_surfaced": imgs_surfaced,
        })
    return rows


def sweep_queue(preds, image_paths):
    """Annotator queue size across thresholds on unscreened frames."""
    n = len(image_paths)
    rows = []
    for t in THRESHOLDS:
        kept = preds[preds["score"] >= t]
        n_img = kept["basename"].nunique()
        rows.append({
            "threshold": t,
            "boxes": int(len(kept)),
            "images": int(n_img),
            "images_per_1000": 1000 * n_img / n if n else None,
            "boxes_per_1000": 1000 * len(kept) / n if n else None,
        })
    return rows


def report(paths):
    for path in paths:
        d = json.load(open(path))
        print(f"\n{'=' * 108}")
        print(f"CHECKPOINT  {d['checkpoint']}")
        print(f"  labelled frames {d['n_labelled_images']}  ground-truth objects {d['n_gt_objects']}"
              f"  |  unscreened sample {d['n_queue_images']}  |  IoU {d['iou_thresh']}")
        print(f"  raw score range on labelled frames: {d['score_min']:.4f} - {d['score_max']:.4f}"
              f"   ({d['n_raw_boxes']} boxes above floor {d['score_floor']})")
        print(f"{'=' * 108}")
        q = {r["threshold"]: r for r in d["queue"]}
        print(f"{'thresh':>7}{'recall':>9}{'prec':>8}{'TP':>7}{'FP':>7}"
              f"{'FP/img':>9}{'img recall':>12}{'queue img/1k':>14}{'queue box/1k':>14}")
        print("-" * 108)
        for r in d["recall"]:
            qq = q.get(r["threshold"], {})
            rec = "  n/a" if r["box_recall"] is None else f"{100 * r['box_recall']:5.1f}%"
            pre = "  n/a" if r["box_precision"] is None else f"{100 * r['box_precision']:4.1f}%"
            imr = "  n/a" if r["image_recall"] is None else f"{100 * r['image_recall']:5.1f}%"
            print(f"{r['threshold']:>7.3f}{rec:>9}{pre:>8}{r['tp']:>7}{r['fp']:>7}"
                  f"{r['fp_per_image']:>9.2f}{imr:>12}"
                  f"{qq.get('images_per_1000', float('nan')):>14.1f}"
                  f"{qq.get('boxes_per_1000', float('nan')):>14.1f}")

        print("\n  recall-target operating points:")
        for target in (0.99, 0.95, 0.90, 0.80):
            hit = [r for r in d["recall"] if r["box_recall"] is not None and r["box_recall"] >= target]
            if not hit:
                print(f"    >={100 * target:.0f}% recall : NOT REACHABLE at any threshold")
                continue
            best = max(hit, key=lambda r: r["threshold"])
            qq = q.get(best["threshold"], {})
            print(f"    >={100 * target:.0f}% recall : min_score {best['threshold']:.3f}"
                  f"  ({100 * best['box_recall']:.1f}% recall, {best['fp_per_image']:.2f} FP/img,"
                  f" {qq.get('images_per_1000', float('nan')):.1f} images per 1,000 to review)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="detection checkpoint to sweep")
    ap.add_argument("--label", default=None, help="short name for the output file")
    ap.add_argument("--flights", nargs="+", default=HOLDOUT)
    ap.add_argument("--queue-sample", type=int, default=2000,
                    help="unscreened frames sampled for the queue-size curve")
    ap.add_argument("--patch-size", type=int, default=1000)
    ap.add_argument("--patch-overlap", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="IMAGES per forward pass; job 39225777 showed 1 is fastest")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.4, help="deepforest evaluate_boxes convention")
    ap.add_argument("--score-floor", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="+", default=None, help="summarise existing JSON(s) and exit")
    args = ap.parse_args()

    if args.report:
        report(args.report)
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required unless --report is given")

    gt = load_ground_truth(args.flights)
    labelled_paths, missing = resolve(gt)
    n_pos = int((~gt["is_negative"]).sum())
    print(f"Ground truth: {len(gt)} boxes ({n_pos} objects, {int(gt['is_negative'].sum())} human-marked "
          f"FalsePositive) over {len(labelled_paths)} frames from {', '.join(args.flights)}", flush=True)
    if missing:
        print(f"  WARNING: {missing} annotated frames not found on disk and skipped", flush=True)

    annotated = {os.path.basename(p) for p in labelled_paths}
    queue_paths = sample_unscreened(args.flights, annotated, args.queue_sample)
    print(f"Queue sample: {len(queue_paths)} unscreened frames", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)

    print(f"\n[1/2] scoring {len(labelled_paths)} labelled frames ...", flush=True)
    p_lab = predict(args.checkpoint, labelled_paths, args.patch_size, args.patch_overlap,
                    args.batch_size, args.workers, args.score_floor)
    print(f"      {len(p_lab)} raw boxes, score {p_lab['score'].min():.4f}-{p_lab['score'].max():.4f}"
          if len(p_lab) else "      0 raw boxes", flush=True)

    print(f"\n[2/2] scoring {len(queue_paths)} unscreened frames ...", flush=True)
    p_q = predict(args.checkpoint, queue_paths, args.patch_size, args.patch_overlap,
                  args.batch_size, args.workers, args.score_floor)
    print(f"      {len(p_q)} raw boxes", flush=True)

    out = {
        "checkpoint": args.checkpoint,
        "flights": args.flights,
        "iou_thresh": args.iou,
        "score_floor": args.score_floor,
        "n_labelled_images": len(labelled_paths),
        "n_gt_objects": n_pos,
        "n_queue_images": len(queue_paths),
        "n_raw_boxes": int(len(p_lab)),
        "score_min": float(p_lab["score"].min()) if len(p_lab) else None,
        "score_max": float(p_lab["score"].max()) if len(p_lab) else None,
        "recall": sweep_recall(p_lab, gt, labelled_paths, args.iou),
        "queue": sweep_queue(p_q, queue_paths),
    }
    label = args.label or os.path.basename(os.path.dirname(args.checkpoint))[:8]
    path = args.out or f"/blue/ewhite/b.weinstein/BOEM/threshold_sweep_{label}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    p_lab.to_csv(path.replace(".json", "_labelled_boxes.csv"), index=False)
    print(f"\nWrote {path}", flush=True)
    report([path])


if __name__ == "__main__":
    main()
