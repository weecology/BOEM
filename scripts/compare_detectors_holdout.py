"""Per-object comparison of detectors already swept on the pinned ZERO-SHOT holdout.

CPU only: reads the `*_labelled_boxes.csv` that threshold_sweep.py / compare_yolo26.py
saved (every box >= 0.01 on the 648 labelled holdout frames) plus their JSONs, and
rebuilds, with the same ground truth and the same greedy IoU-0.4 matching:

  curve.csv              pooled box recall / precision / image recall at every 0.01
  matched_precision.csv  each model's max recall at precision >= 70/77/83/87%
  per_object_hits.csv    one row per ground-truth object: hit @0.70 per model, the
                         score of the box that matched it, best overlapping score,
                         size bin, flock-density bin, bird / non-bird
  breakdown.csv          hit rate @0.70 by size bin, density bin, bird
  pr_curve.png           precision vs recall, all models
  examples/*.jpg         disagreement mosaics @0.70

Greedy highest-score-first matching is nested in the threshold (the boxes above t
are a score-ordered prefix), so one pass over all boxes gives the exact
threshold_sweep.sweep_recall matching at every t. main() checks this against the
JSON's TP/FP before writing anything.

All recall here is ZERO-SHOT, pooled: matched GT boxes / all GT boxes.

    uv run python scripts/compare_detectors_holdout.py
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from threshold_sweep import HOLDOUT, IMAGERY, load_ground_truth, resolve  # noqa: E402

SRC = Path("/blue/ewhite/b.weinstein/BOEM/yolo26_comparison")
OUT = SRC / "three_way"
IOU = 0.4
OP = 0.70
# name -> (sweep JSON stem, colour). Colours: validated default categorical slots 1-3.
MODELS = {
    "1968e962": ("threshold_sweep_1968e962_e29", "#2a78d6"),   # production
    "yolo26s": ("threshold_sweep_yolo26s", "#eb6834"),
    "c6ecec12": ("threshold_sweep_c6ecec12_e27", "#1baf7a"),   # 11-20 refresh
}
PRECISION_TARGETS = [0.70, 0.77, 0.83, 0.87]
NON_BIRD = {"Delphinidae", "Tursiops truncatus", "Chondrichthyes", "Elasmobranchii",
            "Myliobatiformes", "Testudines", "Chelonioidea", "Caretta caretta"}
SIZE_BINS = [0, 30, 45, 60, 90, np.inf]
SIZE_LABELS = ["<30px", "30-45", "45-60", "60-90", ">90"]
DENSITY_BINS = [0, 1, 5, 20, np.inf]
DENSITY_LABELS = ["1", "2-5", "6-20", ">20"]


def greedy(pred: pd.DataFrame, gt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """threshold_sweep.match over ALL boxes. Returns (gt index matched per pred or -1,
    best IoU>=0.4 score per gt, best score per gt of any box whose centre is inside it)."""
    assigned = np.full(len(pred), -1)
    if len(pred) == 0 or len(gt) == 0:
        return assigned, np.zeros(len(gt)), np.zeros(len(gt))
    p = torch.tensor(pred[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
    g = torch.tensor(gt[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
    ious = box_iou(p, g).numpy()
    scores = pred["score"].values
    best = np.where(ious >= IOU, scores[:, None], 0).max(axis=0)
    px, py = (pred.xmin.values + pred.xmax.values) / 2, (pred.ymin.values + pred.ymax.values) / 2
    inside = ((px[:, None] >= gt.xmin.values) & (px[:, None] <= gt.xmax.values)
              & (py[:, None] >= gt.ymin.values) & (py[:, None] <= gt.ymax.values))
    centre = np.where(inside, scores[:, None], 0).max(axis=0)
    taken = set()
    for pi in np.argsort(-scores):
        cand = [(ious[pi, gi], gi) for gi in range(len(gt)) if gi not in taken and ious[pi, gi] >= IOU]
        if cand:
            gi = max(cand)[1]
            taken.add(gi)
            assigned[pi] = gi
    return assigned, best, centre


def match_model(preds: pd.DataFrame, pos: pd.DataFrame, basenames: list[str]):
    """Per-pred TP flag and per-GT matched/best scores across all labelled frames."""
    preds = preds[preds["basename"].isin(basenames)].copy()
    preds["tp"] = False
    matched = pd.Series(0.0, index=pos.index)  # score of the box that matched it (0 = never)
    best = pd.Series(0.0, index=pos.index)
    centre = pd.Series(0.0, index=pos.index)
    pred_by = dict(tuple(preds.groupby("basename")))
    for b, g in pos.groupby("basename"):
        p = pred_by.get(b)
        if p is None:
            continue
        assigned, best[g.index], centre[g.index] = greedy(p, g)
        hit = assigned >= 0
        preds.loc[p.index[hit], "tp"] = True
        matched[g.index[assigned[hit]]] = p["score"].values[hit]
    return preds, matched, best, centre


def curve(preds: pd.DataFrame, pos: pd.DataFrame, neg_frames: set, thresholds) -> pd.DataFrame:
    """sweep_recall's numbers at arbitrary thresholds from one greedy pass."""
    n_gt, gt_frames = len(pos), set(pos["basename"])
    counted = preds[~preds["basename"].isin(neg_frames) | preds["tp"]]
    rows = []
    for t in thresholds:
        k = counted[counted["score"] >= t]
        tp, fp = int(k["tp"].sum()), int((~k["tp"]).sum())
        surfaced = set(preds.loc[preds["score"] >= t, "basename"])
        rows.append({"threshold": round(t, 3), "tp": tp, "fp": fp,
                     "box_recall": tp / n_gt, "box_precision": tp / (tp + fp) if tp + fp else np.nan,
                     "image_recall": len(surfaced & gt_frames) / len(gt_frames)})
    return pd.DataFrame(rows)


def queue_rate(sweep: dict, t: float) -> tuple[float, float]:
    """Unscreened-queue images/1k at the highest 0.05-grid threshold <= t (an upper bound
    off-grid: the queue sample's boxes were never saved, only its per-threshold counts)."""
    grid = [r for r in sweep["queue"] if r["threshold"] <= t + 1e-9]
    r = max(grid, key=lambda r: r["threshold"])
    return r["threshold"], r["images_per_1000"]


# ---------------------------------------------------------------- mosaics

FONT = ImageFont.load_default(size=14)
CELL, PANEL = 300, 380


def crop_cell(row, boxes: dict[str, pd.DataFrame], caption: list[str]) -> Image.Image:
    img = Image.open(Path(IMAGERY) / row.flight / row.basename).convert("RGB")
    cx, cy = (row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2
    half = max(150, 1.6 * max(row.xmax - row.xmin, row.ymax - row.ymin))
    x0, y0 = int(cx - half), int(cy - half)
    crop = img.crop((x0, y0, int(cx + half), int(cy + half)))
    s = CELL / crop.width
    crop = crop.resize((CELL, CELL))
    d = ImageDraw.Draw(crop)

    def rect(b, colour, w):
        d.rectangle([(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s], outline=colour, width=w)

    for name, b in boxes.items():
        for r in b.itertuples():
            rect((r.xmin, r.ymin, r.xmax, r.ymax), MODELS[name][1] if name in MODELS else "#ffffff", 2)
    cell = Image.new("RGB", (CELL, PANEL), "white")
    cell.paste(crop, (0, 0))
    dc = ImageDraw.Draw(cell)
    for i, line in enumerate(caption):
        dc.text((4, CELL + 4 + 18 * i), line, fill="#0b0b0b", font=FONT)
    return cell


def mosaic(cells: list[Image.Image], title: str, subtitle: str, path: Path, cols: int = 4):
    if not cells:
        return
    rows = -(-len(cells) // cols)
    head = 50
    canvas = Image.new("RGB", (cols * (CELL + 8) + 8, head + rows * (PANEL + 8)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill="#0b0b0b", font=ImageFont.load_default(size=20))
    d.text((8, 30), subtitle, fill="#555555", font=FONT)
    for i, c in enumerate(cells):
        canvas.paste(c, (8 + (i % cols) * (CELL + 8), head + (i // cols) * (PANEL + 8)))
    canvas.save(path, quality=90)


def preds_near(preds: pd.DataFrame, row, half: float) -> pd.DataFrame:
    cx, cy = (row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2
    p = preds[(preds["basename"] == row.basename) & (preds["score"] >= OP)]
    return p[(p.xmax > cx - half) & (p.xmin < cx + half) & (p.ymax > cy - half) & (p.ymin < cy + half)]


def object_examples(hits, pos, preds, mask, title, path, n=12, seed=0):
    sel = hits[mask]
    # One object per frame where possible, so a single flock doesn't fill the panel.
    sel = sel.sample(frac=1, random_state=seed).drop_duplicates("basename").head(n) if len(sel) else sel
    cells = []
    for r in sel.itertuples():
        g = pos[pos["basename"] == r.basename]
        boxes = {"gt": g}
        boxes.update({m: preds_near(preds[m], r, 400) for m in MODELS})
        cap = [f"{r.label[:30]} | {r.side:.0f}px | {r.n_in_frame} in frame",
               "  ".join(f"{m[:4]} {getattr(r, 'best_' + m):.2f}" for m in MODELS)]
        cells.append(crop_cell(r, boxes, cap))
    mosaic(cells, f"{title}  (n={int(mask.sum())})",
           "white = ground truth   " + "   ".join(f"{m}" for m in MODELS)
           + f" boxes >= {OP}  (blue / orange / aqua);  caption = best overlapping score per model",
           path)


def fp_examples(preds, pos, neg_frames, model, path, n=12):
    fp = preds[model]
    fp = fp[(fp["score"] >= OP) & ~fp["tp"] & ~fp["basename"].isin(neg_frames)]
    fp = fp.sort_values("score", ascending=False).drop_duplicates("basename").head(n)
    flight = pos.drop_duplicates("basename").set_index("basename")["flight"]
    cells = []
    for r in fp.itertuples():
        rr = pd.Series({"xmin": r.xmin, "ymin": r.ymin, "xmax": r.xmax, "ymax": r.ymax,
                        "basename": r.basename, "flight": flight[r.basename]})
        boxes = {"gt": pos[pos["basename"] == r.basename]}
        boxes.update({m: preds_near(preds[m], rr, 400) for m in MODELS})
        others = [m for m in MODELS if m != model and len(preds_near(preds[m], rr, 1))]
        cells.append(crop_cell(rr, boxes, [f"{model} score {r.score:.2f}",
                                           "also fired: " + (", ".join(others) or "none")]))
    mosaic(cells, f"{model} false positives >= {OP} (highest-scoring, one per frame)",
           "white = ground truth   blue 1968e962 / orange yolo26s / aqua c6ecec12 boxes >= 0.70", path)


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    gt = load_ground_truth(HOLDOUT)
    paths, _ = resolve(gt)
    basenames = [os.path.basename(p) for p in paths]
    pos = gt[~gt["is_negative"]].reset_index(drop=True)
    neg_frames = set(gt.loc[gt["is_negative"], "basename"])

    sweeps, preds, hits = {}, {}, pos.copy()
    fine = np.round(np.arange(0.01, 1.0, 0.01), 3)
    curves = []
    for m, (stem, _) in MODELS.items():
        sweeps[m] = json.load(open(SRC / f"{stem}.json"))
        raw = pd.read_csv(SRC / f"{stem}_labelled_boxes.csv")
        raw["basename"] = raw["image_path"].map(lambda p: os.path.basename(str(p)))
        preds[m], hits[f"score_{m}"], hits[f"best_{m}"], hits[f"centre_{m}"] = match_model(raw, pos, basenames)
        hits[f"hit_{m}"] = hits[f"score_{m}"] >= OP
        # Sensitivity only, NOT the metric of record: a box >= OP centred on the object,
        # regardless of IoU. Separates "not detected" from "detected, box fit < IoU 0.4".
        hits[f"centre_hit_{m}"] = hits[f"centre_{m}"] >= OP

        # Reproduce the sweep JSON exactly before trusting anything below.
        grid = [r["threshold"] for r in sweeps[m]["recall"]]
        check = curve(preds[m], pos, neg_frames, grid)
        ref = pd.DataFrame(sweeps[m]["recall"])
        bad = (check["tp"].values != ref["tp"].values) | (check["fp"].values != ref["fp"].values)
        if bad.any():
            raise SystemExit(f"{m}: greedy pass disagrees with {stem}.json at {list(ref.threshold[bad])}")
        c = curve(preds[m], pos, neg_frames, fine)
        c.insert(0, "model", m)
        curves.append(c)
        print(f"{m}: reproduces {stem}.json TP/FP at all {len(grid)} thresholds", flush=True)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / "curve.csv", index=False)

    # Matched precision: lowest threshold (=> max recall) whose precision clears the target.
    mp = []
    for target in PRECISION_TARGETS:
        for m in MODELS:
            c = curves[(curves.model == m) & (curves.box_precision >= target)]
            if c.empty:
                mp.append({"precision_target": target, "model": m})
                continue
            r = c.loc[c.box_recall.idxmax()]
            qt, q = queue_rate(sweeps[m], r.threshold)
            mp.append({"precision_target": target, "model": m, "threshold": r.threshold,
                       "box_recall": r.box_recall, "box_precision": r.box_precision,
                       "image_recall": r.image_recall, "queue_grid_threshold": qt, "queue_img_per_1k": q})
    mp = pd.DataFrame(mp)
    mp.to_csv(OUT / "matched_precision.csv", index=False)

    # Object attributes.
    hits["side"] = np.sqrt((hits.xmax - hits.xmin) * (hits.ymax - hits.ymin))
    hits["size_bin"] = pd.cut(hits.side, SIZE_BINS, labels=SIZE_LABELS, right=False)
    hits["n_in_frame"] = hits.groupby("basename")["basename"].transform("size")
    hits["density"] = pd.cut(hits.n_in_frame, DENSITY_BINS, labels=DENSITY_LABELS)
    hits["bird"] = ~hits.label.isin(NON_BIRD)
    keep = ["xmin", "ymin", "xmax", "ymax", "label", "flight", "basename", "side", "size_bin",
            "n_in_frame", "density", "bird"] + [f"{k}_{m}" for m in MODELS for k in ("hit", "score", "best", "centre_hit", "centre")]
    hits[keep].to_csv(OUT / "per_object_hits.csv", index=False)

    hit_cols = [f"hit_{m}" for m in MODELS]
    rows = []
    for col in ["size_bin", "density", "bird"]:
        g = hits.groupby(col, observed=True)
        t = g[hit_cols].mean().mul(100).round(1)
        t.insert(0, "n", g.size())
        t.insert(0, "value", t.index.astype(str))
        t.insert(0, "factor", col)
        rows.append(t.reset_index(drop=True))
    breakdown = pd.concat(rows, ignore_index=True)
    breakdown.to_csv(OUT / "breakdown.csv", index=False)
    overlap = hits.groupby(hit_cols).size().rename("n").reset_index()

    # PR curve.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for m, (_, colour) in MODELS.items():
        c = curves[curves.model == m].dropna()
        ax.plot(c.box_recall * 100, c.box_precision * 100, color=colour, lw=2, label=m)
        at = c[c.threshold == OP].iloc[0]
        ax.plot(at.box_recall * 100, at.box_precision * 100, "o", color=colour, ms=8,
                mec="white", mew=2, zorder=3)
    ax.set(xlabel="ZERO-SHOT box recall, pooled (%)", ylabel="box precision (%)",
           title="Pinned zero-shot holdout, 1,229 objects, IoU 0.4  (dot = 0.70)", ylim=(50, 100), xlim=(20, 75))
    ax.grid(color="#e1e0d9", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve.png")

    # Mosaics.
    ex = OUT / "examples"
    h = {m: hits[f"hit_{m}"] for m in MODELS}
    specs = [
        ("1_c6ecec12_hits_1968e962_misses", h["c6ecec12"] & ~h["1968e962"], "New c6ecec12 finds, production 1968e962 misses"),
        ("2_1968e962_hits_c6ecec12_misses", h["1968e962"] & ~h["c6ecec12"], "Production 1968e962 finds, new c6ecec12 misses"),
        ("3_c6ecec12_hits_yolo_misses", h["c6ecec12"] & ~h["yolo26s"], "New c6ecec12 finds, YOLO26s misses"),
        ("4_yolo_hits_c6ecec12_misses", h["yolo26s"] & ~h["c6ecec12"], "YOLO26s finds, new c6ecec12 misses"),
        ("5_all_three_miss", ~h["c6ecec12"] & ~h["1968e962"] & ~h["yolo26s"], "Missed by all three"),
        ("6_c6ecec12_hits_dense_flocks", h["c6ecec12"] & (hits.n_in_frame > 20) & ~h["1968e962"], "Dense flocks (>20): c6ecec12 finds, 1968e962 misses"),
    ]
    for name, mask, title in specs:
        object_examples(hits, pos, preds, mask, title, ex / f"{name}.jpg")
    fp_examples(preds, pos, neg_frames, "c6ecec12", ex / "7_c6ecec12_false_positives.jpg")

    pd.set_option("display.width", 200)
    print("\n@0.70 (from sweep JSONs) -- ZERO-SHOT pooled recall:")
    for m in MODELS:
        r = next(r for r in sweeps[m]["recall"] if r["threshold"] == OP)
        q = next(r for r in sweeps[m]["queue"] if r["threshold"] == OP)
        print(f"  {m:9s} recall {100 * r['box_recall']:.1f}  prec {100 * r['box_precision']:.1f}  "
              f"img recall {100 * r['image_recall']:.1f}  queue {q['images_per_1000']:.1f}/1k")
    print("\nMatched precision:\n", mp.round(3).to_string(index=False))
    print("\nHit rate @0.70 (%):\n", breakdown.to_string(index=False))
    print("\nOverlap @0.70:\n", overlap.to_string(index=False))
    print("\nSensitivity (NOT the metric of record): objects with a box >= 0.70 centred inside them")
    for m in MODELS:
        c, h_ = hits[f"centre_hit_{m}"], hits[f"hit_{m}"]
        print(f"  {m:9s} IoU-0.4 hits {h_.sum()}  centre hits {c.sum()} ({100 * c.mean():.1f}%)"
              f"  centred but IoU<0.4: {(c & ~h_).sum()}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
