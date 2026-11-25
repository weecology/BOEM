"""Inference input-size sweep for the production detector (c6ecec12/epoch27), no retraining.

Ben (via boem-d5, 2026-11-25): validation, zero-shot and July review load are co-equal ("it's all real imagery"),
so push the size curve until it bends. Tiles stay 1000 px, so sizes above 1000 are pure upsampling.

  task <size>   GPU, one size: min_size = max_size = size (800 = production 800/1333)
                VALIDATION   predictions on test_max_empty_0_25.csv (floor 0.3) -> density strata; wall time + peak GPU mem
                ZERO-SHOT    threshold_sweep.py on the pinned holdout (same args as 52121265 / 52502538)
                JULY         review_load_by_threshold.py's 5,000-frame sample -> human_review per threshold
  aggregate     CPU, all sizes
                VALIDATION   recall at interpolated overall precision 0.97, by birds/tile AND by GT box size
                ZERO-SHOT    recall at matched precision 85.4% + PAIRED frame bootstrap of recall vs 800
                JULY         review frames/boxes at 0.60 and 0.70; contact sheets of >= 0.60 boxes new vs 800
                SIZES        GT box-size distribution: July survey annotations vs validation vs holdout

    uv run python scripts/input_size_sweep.py task 1400
    uv run python scripts/input_size_sweep.py aggregate
"""
import importlib.util
import json
import random
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deepforest import evaluate
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from density_strata import CROPS, is_empty, load_gt, matched_precision, report, strata_table  # noqa: E402
from src import detection  # noqa: E402

CKPT = ("/blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/"
        "epoch27-val_cls0.0867.ckpt")
VAL = CROPS / "test_max_empty_0_25.csv"
OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/size_sweep")
SIZES = [800, 1000, 1200, 1400, 1600, 2000]
SIZE_BINS, SIZE_LABELS = [0, 16, 32, 64, np.inf], ["<16", "16-32", "32-64", ">64"]
ZS_PRECISION, VAL_PRECISION = 0.854, 0.97
_load = detection.load


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # functions only; __main__ blocks do not run
    return m


def use_size(size: int) -> None:
    lo, hi = (800, 1333) if size == 800 else (size, size)

    def load(*args, **kwargs):
        m = _load(*args, **kwargs)
        m.model.transform.min_size, m.model.transform.max_size = (lo,), hi
        return m
    detection.load = load


# ------------------------------------------------------------------ GPU task, one size
def task(size: int) -> None:
    out = OUT / f"in{size}"
    out.mkdir(parents=True, exist_ok=True)
    use_size(size)
    m = detection.load(checkpoint=CKPT, score_thresh=0.3)
    print(f"in{size}: transform {m.model.transform.min_size}/{m.model.transform.max_size}", flush=True)
    m.config["workers"], m.config["batch_size"] = 8, 32
    m.create_trainer(accelerator="gpu", devices=1, logger=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    p = m.predict_file(csv_file=str(VAL), root_dir=str(CROPS))
    cost = {"size": size, "val_tiles": len(set(p.image_path)) if len(p) else 0, "val_seconds": time.time() - t0,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    p["image_path"] = p.image_path.map(lambda x: Path(x).name)
    p.to_csv(out / "val_predictions.csv", index=False)
    del m
    torch.cuda.empty_cache()

    t0 = time.time()
    sys.argv = ["threshold_sweep.py", "--checkpoint", CKPT, "--label", f"c6ecec12_e27_in{size}",
                "--out", str(out / "zero_shot.json"), "--queue-sample", "2000", "--batch-size", "1", "--workers", "5"]
    runpy.run_path(str(REPO / "scripts/threshold_sweep.py"), run_name="__main__")
    cost["zero_shot_seconds"] = time.time() - t0

    from hydra import compose, initialize_config_dir
    rl = module("review_load_by_threshold")
    with initialize_config_dir(config_dir=str(REPO / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    random.seed(0)  # the 5,000-frame draw of 52348607 / 52502730
    flights = sorted(rl.IMAGERY.glob(rl.FLIGHT_GLOB))
    all_frames = {f: rl.frames(f) for f in flights}
    n_survey = sum(map(len, all_frames.values()))
    sample = {f: random.sample(v, min(rl.PER_FLIGHT, len(v))) for f, v in all_frames.items()}
    n_sampled = sum(map(len, sample.values()))
    crop, hcast = rl.load_models(cfg)
    t0 = time.time()
    preds = rl.predict(CKPT, sample, cfg, crop, hcast)
    cost["july_seconds"], cost["july_frames"] = time.time() - t0, n_sampled
    preds.to_csv(out / "july_predictions.csv", index=False)
    rl.summarise(preds, n_sampled, n_survey, cfg).to_csv(out / "july_summary.csv", index=False)
    json.dump(cost, open(out / "cost.json", "w"), indent=2)
    print(json.dumps(cost, indent=2), flush=True)


# ------------------------------------------------------------------ CPU aggregate
def recall_by_box_size(preds: pd.DataFrame, gt: pd.DataFrame, t: float) -> pd.Series:
    boxes = gt[~is_empty(gt)].reset_index(drop=True)
    boxes["pos"] = boxes.groupby("image_path").cumcount()
    p = preds[(preds.score >= t) & preds.image_path.isin(boxes.image_path)].assign(label="Object")
    res = evaluate.evaluate_boxes(p, boxes[["xmin", "ymin", "xmax", "ymax", "image_path"]].assign(label="Object"),
                                  iou_threshold=0.4)["results"]
    hit = res[res.match][["image_path", "truth_id"]].astype({"truth_id": int}).assign(match=True)
    boxes = boxes.merge(hit, left_on=["image_path", "pos"], right_on=["image_path", "truth_id"], how="left")
    side = np.sqrt((boxes.xmax - boxes.xmin) * (boxes.ymax - boxes.ymin))
    return boxes.match.fillna(False).astype(bool).groupby(pd.cut(side, SIZE_BINS, labels=SIZE_LABELS), observed=False).mean()


def frame_counts(labelled: pd.DataFrame, ts, gt: pd.DataFrame, t: float) -> pd.DataFrame:
    """Per labelled frame: GT objects, matched, FP at threshold t (threshold_sweep's own greedy matcher)."""
    pos = gt[~gt.is_negative]
    neg_frames = set(gt[gt.is_negative].basename)
    frames = sorted(set(gt.basename))
    p = labelled[labelled.score >= t]
    rows = []
    for f in frames:
        mt, fp = ts.match(p[p.basename == f], pos[pos.basename == f], 0.4)
        # As sweep_recall: boxes on a frame with a human FalsePositive mark count neither way.
        rows.append((f, int((pos.basename == f).sum()), mt, 0 if f in neg_frames else fp))
    return pd.DataFrame(rows, columns=["basename", "n_gt", "tp", "fp"]).set_index("basename")


def zero_shot_table(ts) -> pd.DataFrame:
    gt = ts.load_ground_truth(ts.HOLDOUT)
    rows, counts = [], {}
    for size in SIZES:
        d = json.load(open(OUT / f"in{size}/zero_shot.json"))
        sw = pd.DataFrame(d["recall"]).sort_values("threshold")
        ok = sw[sw.box_precision.notna()]
        t = float(np.interp(ZS_PRECISION, ok.box_precision, ok.threshold))
        lab = pd.read_csv(OUT / f"in{size}/zero_shot_labelled_boxes.csv")
        # Check our per-frame matcher reproduces the sweep's own TP/FP at a grid threshold before trusting it.
        chk = frame_counts(lab, ts, gt, 0.70)
        ref = sw.set_index("threshold").loc[0.70]
        assert (chk.tp.sum(), chk.fp.sum()) == (ref.tp, ref.fp), (size, chk.tp.sum(), chk.fp.sum(), ref.tp, ref.fp)
        counts[size] = frame_counts(lab, ts, gt, t)
        c = counts[size]
        q = pd.DataFrame(d["queue"]).set_index("threshold")
        rows.append({"size": size, "thr@P85.4": t, "recall@P85.4": c.tp.sum() / c.n_gt.sum(),
                     "prec@P85.4": c.tp.sum() / (c.tp.sum() + c.fp.sum()),
                     "recall@0.60": sw.set_index("threshold").loc[0.60, "box_recall"],
                     "prec@0.60": sw.set_index("threshold").loc[0.60, "box_precision"],
                     "queue_img_per_1k@0.60": q.loc[0.60, "images_per_1000"]})
    # Paired bootstrap by frame: resample the same frames for both sizes, recall difference vs 800.
    rng = np.random.default_rng(0)
    base = counts[800]
    idx = rng.integers(0, len(base), size=(2000, len(base)))
    for r in rows:
        c = counts[r["size"]].loc[base.index]
        diff = c.tp.values[idx].sum(1) / c.n_gt.values[idx].sum(1) - base.tp.values[idx].sum(1) / base.n_gt.values[idx].sum(1)
        r["diff_vs_800"], r["ci_lo"], r["ci_hi"] = r["recall@P85.4"] - rows[0]["recall@P85.4"], *np.percentile(diff, [2.5, 97.5])
    return pd.DataFrame(rows)


def foam_sheet(size: int, n: int = 48) -> int:
    """>= 0.60 July boxes at this size with no >= 0.40 box at 800 overlapping them (IoU >= 0.3)."""
    from torchvision.ops import box_iou
    rl = module("review_load_by_threshold")
    big = pd.read_csv(OUT / f"in{size}/july_predictions.csv")
    ref = pd.read_csv(OUT / "in800/july_predictions.csv")
    big = big[big.score >= 0.60]
    new = []
    for f, g in big.groupby("image_path"):
        r = ref[ref.image_path == f]
        if r.empty:
            new.append(g)
            continue
        iou = box_iou(torch.tensor(g[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32),
                      torch.tensor(r[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)).max(1).values
        new.append(g[(iou < 0.3).numpy()])
    new = pd.concat(new).sort_values("score", ascending=False) if new else big.iloc[0:0]
    new.to_csv(OUT / f"in{size}/july_new_vs_800.csv", index=False)
    sheet = Image.new("RGB", (8 * 164, 6 * 180), "white")
    for j, b in enumerate(new.head(n).itertuples()):
        img = Image.open(rl.IMAGERY / b.flight / Path(b.image_path).name).convert("RGB")
        cx, cy = (b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2
        x0, y0 = int(np.clip(cx - 80, 0, img.width - 160)), int(np.clip(cy - 80, 0, img.height - 160))
        ImageDraw.Draw(img).rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=(255, 0, 0), width=2)
        X, Y = (j % 8) * 164, (j // 8) * 180
        sheet.paste(img.crop((x0, y0, x0 + 160, y0 + 160)), (X, Y))
        label = getattr(b, "cropmodel_label", "") or ""
        ImageDraw.Draw(sheet).text((X + 2, Y + 162), f"{b.score:.2f} {str(label)[:16]}", fill=(0, 0, 0))
    sheet.save(OUT / f"july_new_vs_800_in{size}.png")
    return len(new)


def box_sizes(ts) -> pd.DataFrame:
    side = lambda d: np.sqrt((d.xmax - d.xmin) * (d.ymax - d.ymin))  # noqa: E731
    train = pd.read_csv(CROPS / "train.csv", low_memory=False, usecols=["flight", "xmin", "ymin", "xmax", "ymax"])
    july = train[train.flight.fillna("").str.startswith("JPG_202607") & ~is_empty(train)]
    val = load_gt(VAL)
    hold = ts.load_ground_truth(ts.HOLDOUT)
    sets = {"July survey (annotated, train.csv)": side(july), "VALIDATION": side(val[~is_empty(val)]),
            "ZERO-SHOT holdout": side(hold[~hold.is_negative])}
    rows = []
    for k, s in sets.items():
        b = pd.cut(s, SIZE_BINS, labels=SIZE_LABELS).value_counts(normalize=True).reindex(SIZE_LABELS)
        rows.append({"set": k, "n": len(s), "median_px": s.median(), "p10": s.quantile(0.1), "p90": s.quantile(0.9),
                     **{f"frac {lab}": b[lab] for lab in SIZE_LABELS}})
    return pd.DataFrame(rows)


def aggregate() -> None:
    pd.set_option("display.width", 200)
    ts = module("threshold_sweep")
    val_gt = load_gt(VAL)
    strata, bysize, cost = {}, {}, []
    for size in SIZES:
        p = pd.read_csv(OUT / f"in{size}/val_predictions.csv")
        t = strata_table(p, val_gt)
        t.to_csv(OUT / f"in{size}/val.density_strata.csv", index=False)
        th, s = matched_precision(t, VAL_PRECISION)
        strata[size] = s.set_index("dens").recall
        bysize[size] = recall_by_box_size(p.assign(image_path=p.image_path.map(lambda x: Path(x).name)), val_gt, th)
        cost.append(json.load(open(OUT / f"in{size}/cost.json")))
    order = ["all", "1", "2", "3-5", "6-10", "11-20", "21-50", ">50"]
    print(f"\n== VALIDATION recall at interpolated overall precision {VAL_PRECISION}, by birds/tile ==")
    print((pd.DataFrame(strata).reindex(order) * 100).round(1).to_string())
    print(f"\n== VALIDATION recall at the same thresholds, by GT box size (sqrt area, tile px) ==")
    print((pd.DataFrame(bysize) * 100).round(1).to_string())
    zs = zero_shot_table(ts)
    zs.to_csv(OUT / "zero_shot_summary.csv", index=False)
    print(f"\n== ZERO-SHOT (pinned holdout) at matched precision {ZS_PRECISION}; paired frame bootstrap vs 800, 95% CI ==")
    print(zs.round(3).to_string(index=False))
    july = []
    for size in SIZES:
        s = pd.read_csv(OUT / f"in{size}/july_summary.csv").set_index("threshold")
        row = {"size": size}
        for thr in (0.6, 0.7):
            row[f"review frames @{thr}"], row[f"review boxes @{thr}"] = s.loc[thr, "review_frames"], s.loc[thr, "review_boxes"]
        row["new >=0.60 boxes vs 800"] = foam_sheet(size) if size != 800 else 0
        july.append(row)
    print("\n== JULY review load (5,000-frame sample; x106.8 to survey); contact sheets july_new_vs_800_in*.png ==")
    print(pd.DataFrame(july).to_string(index=False))
    c = pd.DataFrame(cost)
    c["val_img_per_s"] = c.val_tiles / c.val_seconds
    print("\n== COST (one B200; val = 14,692 tiles, batch 32; zero-shot + July = full frames, predict_tile) ==")
    print(c.round(2).to_string(index=False))
    print("\n== GT box size (sqrt area px) by image set ==")
    print(box_sizes(ts).round(3).to_string(index=False))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    if sys.argv[1] == "task":
        task(int(sys.argv[2]))
    else:
        aggregate()
