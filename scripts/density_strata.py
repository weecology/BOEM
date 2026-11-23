"""Pooled box recall/precision by ground-truth density stratum (birds per tile).

Shared scorer for the dense-flock experiments. Pooled = matched GT boxes / all GT boxes
(not DeepForest's per-tile mean). Precision counts only predictions on tiles with GT, so
empty-tile false positives are reported separately.

    uv run python scripts/density_strata.py <predictions.csv> [gt.csv] [--target-precision 0.979]
"""
import argparse
from pathlib import Path

import pandas as pd
from deepforest import evaluate

CROPS = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
BINS = [0, 1, 2, 5, 10, 20, 50, float("inf")]
LABELS = ["1", "2", "3-5", "6-10", "11-20", "21-50", ">50"]
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
# c6ecec12/epoch27 overall validation precision at 0.70 (97.88%, job 52344877): arms are compared
# at the threshold where they reach the same precision, so a model that just scores
# everything higher does not look better.
DEFAULT_TARGET_PRECISION = 0.978


def is_empty(df: pd.DataFrame) -> pd.Series:
    return (df.xmin == 0) & (df.xmax == 0) & (df.ymin == 0) & (df.ymax == 0)


def load_gt(gt_csv: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_csv, low_memory=False)
    gt["image_path"] = gt.image_path.map(lambda p: Path(p).name)
    return gt


def density(boxes: pd.DataFrame) -> pd.Series:
    n = boxes.groupby("image_path").size()
    return pd.cut(n, BINS, labels=LABELS).rename("dens")


def match(preds: pd.DataFrame, boxes: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-tile TP / GT / prediction counts at one score threshold (IoU 0.4)."""
    p = preds[(preds.score >= threshold) & preds.image_path.isin(boxes.image_path)]
    g = boxes[["xmin", "ymin", "xmax", "ymax", "image_path"]].assign(label="Object")
    n = boxes.groupby("image_path").size()
    if p.empty:  # evaluate_boxes returns results=None when nothing clears the threshold
        return pd.DataFrame({"n_gt": n, "tp": 0, "pred": 0})
    res = evaluate.evaluate_boxes(p.assign(label="Object"), g, iou_threshold=0.4)["results"]
    return pd.DataFrame({
        "n_gt": n,
        "tp": res.groupby("image_path").match.sum().reindex(n.index, fill_value=0),
        "pred": p.groupby("image_path").size().reindex(n.index, fill_value=0),
    })


def strata_table(preds: pd.DataFrame, gt: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Long table: one row per (threshold, density stratum), plus an 'all' stratum."""
    preds = preds.assign(image_path=preds.image_path.map(lambda p: Path(p).name))
    boxes = gt[~is_empty(gt)]
    empty_tiles = set(gt.image_path) - set(boxes.image_path)
    dens = density(boxes)
    rows = []
    for t in thresholds:
        per_tile = match(preds, boxes, t).join(dens)
        fp_empty = preds[(preds.score >= t) & preds.image_path.isin(empty_tiles)]
        strata = [("all", per_tile)] + [(k, v) for k, v in per_tile.groupby("dens", observed=True)]
        for name, s in strata:
            rows.append({"threshold": t, "dens": name, "tiles": len(s), "n_gt": s.n_gt.sum(),
                         "pred": s.pred.sum(), "tp": s.tp.sum(),
                         "empty_tile_fp_boxes": len(fp_empty) if name == "all" else None})
    out = pd.DataFrame(rows)
    out["recall"] = out.tp / out.n_gt
    out["precision"] = out.tp / out.pred.where(out.pred > 0)
    return out


def matched_precision(table: pd.DataFrame, target: float) -> tuple[float, pd.DataFrame]:
    """Lowest grid threshold whose overall precision reaches target, and the strata there."""
    overall = table[table.dens == "all"]
    ok = overall[overall.precision >= target]
    t = ok.threshold.min() if len(ok) else overall.threshold.max()
    return t, table[table.threshold == t]


def report(table: pd.DataFrame, target: float = DEFAULT_TARGET_PRECISION) -> None:
    order = ["all"] + LABELS
    for t in (0.40, 0.70):
        s = table[table.threshold == t].set_index("dens").reindex(order).dropna(how="all")
        print(f"\n== score >= {t:.2f} ==")
        print(s[["tiles", "n_gt", "tp", "pred", "recall", "precision"]].round(3).to_string())
    t, s = matched_precision(table, target)
    print(f"\n== matched overall precision >= {target:.3f}: threshold {t:.2f} ==")
    print(s.set_index("dens").reindex(order).dropna(how="all")[["n_gt", "tp", "recall", "precision"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("gt", type=Path, nargs="?", default=CROPS / "test_max_empty_0_25.csv")
    ap.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    args = ap.parse_args()
    table = strata_table(pd.read_csv(args.predictions), load_gt(args.gt))
    out = args.predictions.with_suffix(".density_strata.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    report(table, args.target_precision)
