"""Dense-flock Phase 0: inference-only probes on a frozen detector (default c6ecec12/epoch27).

Validation (in-distribution) tiles only; nothing here is zero-shot.

  0a paste test   -- does a fixed bird's score fall as bird chips are pasted around it?
                     near / far / background-chip arms separate local bird context from a
                     global "busy tile" effect and from paste artifacts.
  0b label audit  -- confident predictions that match no GT, by density, on VALIDATION tiles
                     (the model has fitted to its own training labels, so training tiles would
                     hide missing labels). Writes contact sheets for eyeballing.
  0d train fit    -- the same density strata on a stratified sample of TRAINING tiles. Low dense
                     recall on tiles the model was fitted to means it cannot fit flocks
                     (capacity or label conflict), not that it lacks dense examples.
  0c re-predict   -- full validation set at 800px (production), 1000px (native), 1200px
                     (upsampled) and NMS 0.2, each scored by density stratum.

    uv run python scripts/dense_flock_phase0.py [checkpoint.ckpt]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

from deepforest import main

sys.path.insert(0, str(Path(__file__).parent))
from density_strata import CROPS, density, is_empty, load_gt, report, strata_table  # noqa: E402

CKPT = ("/blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/"
        "epoch27-val_cls0.0867.ckpt")
VAL = CROPS / "test_max_empty_0_25.csv"
TRAIN = CROPS / "train_max_empty_0_10.csv"
OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0")
KS = [0, 2, 5, 10, 20, 40]
# SMOKE=1: tiny samples on CPU, outputs under phase0/smoke, to check the script end to end.
SMOKE = bool(os.environ.get("SMOKE"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if SMOKE:
    OUT = OUT / "smoke"
rng = np.random.default_rng(0)


def load_model(ckpt: str) -> main.deepforest:
    m = main.deepforest.load_from_checkpoint(ckpt, map_location=DEVICE).eval()
    m.model.score_thresh = 0.05  # read at postprocess time, so this takes effect on a loaded model
    m.config["workers"], m.config["batch_size"] = (0, 4) if SMOKE else (8, 32)
    m.create_trainer(accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1, logger=False)
    return m


def read(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype("float32")


@torch.no_grad()
def detect(m, images: list[np.ndarray]) -> list[dict]:
    x = [torch.from_numpy(i).permute(2, 0, 1).to(DEVICE) / 255 for i in images]
    return [{k: v.cpu() for k, v in p.items()} for p in m.model(x)]


def score_at(pred: dict, box: torch.Tensor) -> float:
    """Best score among predictions overlapping box at IoU >= 0.4 (0 if none)."""
    if len(pred["boxes"]) == 0:
        return 0.0
    hit = box_iou(box[None], pred["boxes"])[0] >= 0.4
    return float(pred["scores"][hit].max()) if hit.any() else 0.0


# ---------------------------------------------------------------- 0a paste test
def feather(h: int, w: int, edge: int = 3) -> np.ndarray:
    """Alpha mask that ramps 0 -> 1 over `edge` px, so chips have no hard rectangle edge."""
    ramp = lambda n: np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / edge, 0, 1)  # noqa: E731
    return np.outer(ramp(h), ramp(w))[..., None]


def place(n: int, sizes: np.ndarray, center: np.ndarray, side: float, mode: str,
          spacing: float, taken: list[np.ndarray]) -> list[np.ndarray]:
    """Rejection-sample non-overlapping chip boxes. near: a window around the target sized to
    keep the real-flock spacing; far: >= 300px from the target centre."""
    win = np.sqrt(n + 1) * spacing * side
    boxes = []
    for s in sizes:
        for _ in range(500):
            if mode == "far":
                c = rng.uniform(s / 2, 1000 - s / 2, 2)
                if np.linalg.norm(c - center) < 300:
                    continue
            else:
                c = np.clip(center + rng.uniform(-win / 2, win / 2, 2), s / 2, 1000 - s / 2)
            b = np.r_[c - s / 2, c + s / 2]
            if all(b[0] > t[2] + 2 or b[2] < t[0] - 2 or b[1] > t[3] + 2 or b[3] < t[1] - 2
                   for t in taken + boxes):
                boxes.append(b)
                break
    return boxes


def paste_test(m, gt: pd.DataFrame, n_targets: int = 300) -> None:
    """Chips come from OTHER tiles of the target's own parent frame (same lighting, sea state,
    survey), so a pasted bird looks like a bird from that frame, not a foreign rectangle.
    The background arm pastes same-size patches cut from those same donor tiles, shifted off
    every GT box: identical paste procedure and provenance, no bird."""
    boxes = gt[~is_empty(gt)].copy()
    boxes["side"] = np.sqrt((boxes.xmax - boxes.xmin) * (boxes.ymax - boxes.ymin))
    n = boxes.groupby("image_path").size()
    # Real flock spacing: median nearest-neighbour centre distance / box side in >=11-bird tiles.
    nn = []
    for _, g in boxes[boxes.image_path.isin(n[n > 10].index)].groupby("image_path"):
        c = torch.tensor(((g[["xmin", "ymin"]].values + g[["xmax", "ymax"]].values) / 2), dtype=torch.float32)
        d = torch.cdist(c, c).fill_diagonal_(np.inf).min(1).values.numpy()
        nn += list(d / g.side.values)
    spacing = float(np.median(nn))
    print(f"0a: real-flock NN spacing = {spacing:.2f} box sides (median over {len(nn):,} birds)")

    donors = boxes[boxes.side.between(15, 60)]
    n_donors = donors.groupby("bname_parent").size()
    singles = boxes[boxes.image_path.isin(n[n == 1].index) & boxes.side.between(20, 60)]
    cx, cy = (singles.xmin + singles.xmax) / 2, (singles.ymin + singles.ymax) / 2
    singles = singles[cx.between(120, 880) & cy.between(120, 880)
                      & singles.bname_parent.map(n_donors).fillna(0).ge(10)]
    n_targets = 2 if SMOKE else n_targets
    # Keep targets the model already scores >= 0.3 so a drop is measurable.
    cand = singles.sample(min(3 * n_targets, len(singles)), random_state=0)
    base = []
    for i in range(0, len(cand), 32):
        chunk = cand.iloc[i:i + 32]
        preds = detect(m, [read(CROPS / p) for p in chunk.image_path])
        base += [score_at(p, torch.tensor([r.xmin, r.ymin, r.xmax, r.ymax], dtype=torch.float32))
                 for p, r in zip(preds, chunk.itertuples())]
    cand["base"] = base
    targets = cand[cand.base >= 0.3].head(n_targets)
    print(f"0a: {len(targets)} single-bird targets (base score >= 0.3 of {len(cand)} sampled; "
          f"{len(singles)} eligible: parent frame has >= 10 other birds)")

    gt_by_tile = {k: g[["xmin", "ymin", "xmax", "ymax"]].values for k, g in boxes.groupby("image_path")}
    cache: dict[str, np.ndarray] = {}

    def tile(p: str) -> np.ndarray:
        if p not in cache:
            if len(cache) > 300:
                cache.clear()
            cache[p] = read(CROPS / p)
        return cache[p]

    def bird_chip(d) -> np.ndarray:
        return tile(d.image_path)[int(d.ymin):int(d.ymax), int(d.xmin):int(d.xmax)]

    def background_chip(d) -> np.ndarray | None:
        h, w = int(d.ymax) - int(d.ymin), int(d.xmax) - int(d.xmin)
        g = gt_by_tile[d.image_path]
        for _ in range(100):
            y, x = rng.integers(0, 1000 - h), rng.integers(0, 1000 - w)
            if not ((x < g[:, 2] + 5) & (x + w > g[:, 0] - 5) & (y < g[:, 3] + 5) & (y + h > g[:, 1] - 5)).any():
                return tile(d.image_path)[y:y + h, x:x + w]
        return None

    rows = []
    for t in targets.itertuples():
        img = read(CROPS / t.image_path)
        tbox = np.array([t.xmin, t.ymin, t.xmax, t.ymax])
        center = (tbox[:2] + tbox[2:]) / 2
        pool = donors[donors.bname_parent == t.bname_parent]
        batch, meta = [], []
        for arm in ("near", "far", "background"):
            for k in KS:
                if k == 0 and arm != "near":
                    continue
                picks = list(pool.iloc[rng.integers(len(pool), size=k)].itertuples())
                chips = [background_chip(d) if arm == "background" else bird_chip(d) for d in picks]
                chips = [c for c in chips if c is not None and min(c.shape[:2]) >= 8]
                sizes = np.array([max(c.shape[:2]) for c in chips])
                placed = place(len(chips), sizes, center, t.side, "far" if arm == "far" else "near", spacing, [tbox])
                out = img.copy()
                pasted = []
                for c, b in zip(chips, placed):
                    x0, y0 = int(b[0]), int(b[1])
                    h, w = min(c.shape[0], 1000 - y0), min(c.shape[1], 1000 - x0)
                    a = feather(h, w)
                    out[y0:y0 + h, x0:x0 + w] = a * c[:h, :w] + (1 - a) * out[y0:y0 + h, x0:x0 + w]
                    pasted.append(np.array([x0, y0, x0 + w, y0 + h], dtype="float32"))
                batch.append(out)
                meta.append((arm, k, len(placed), pasted if arm != "background" else []))
        preds = detect(m, batch)
        tb = torch.tensor(tbox, dtype=torch.float32)
        for p, (arm, k, k_placed, pb) in zip(preds, meta):
            sc = np.array([score_at(p, torch.from_numpy(b)) for b in pb])
            rows.append({"image_path": t.image_path, "bname_parent": t.bname_parent, "side": t.side,
                         "arm": arm, "k": k, "k_placed": k_placed, "target_score": score_at(p, tb),
                         "pasted_mean_score": sc.mean() if len(sc) else np.nan,
                         "pasted_frac_070": (sc >= 0.7).mean() if len(sc) else np.nan})
        if len(rows) <= len(meta):  # first target: save every arm at k=20 for eyeballing
            for (arm, k, _, _), im in zip(meta, batch):
                if k == 20:
                    Image.fromarray(im.astype("uint8")).save(OUT / f"0a_example_{arm}_k20.png")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "0a_paste_test.csv", index=False)
    k0 = r[r.k == 0].set_index("image_path").target_score
    r["delta"] = r.target_score - r.image_path.map(k0)
    print("\n== 0a: target score vs pasted neighbours (VALIDATION single-bird tiles) ==")
    print(r.groupby(["arm", "k"]).agg(n=("delta", "size"), k_placed=("k_placed", "mean"),
                                      target_score=("target_score", "median"), delta=("delta", "median"),
                                      delta_mean=("delta", "mean"),
                                      target_ge_070=("target_score", lambda x: (x >= 0.7).mean()),
                                      pasted_score=("pasted_mean_score", "median"),
                                      pasted_ge_070=("pasted_frac_070", "mean")).round(3).to_string())


# ---------------------------------------------------------------- 0b label audit
def label_audit(m, gt: pd.DataFrame, n_singles: int = 300) -> None:
    boxes = gt[~is_empty(gt)]
    dens = density(boxes)
    dense_tiles = list(dens[dens.isin(["6-10", "11-20", "21-50", ">50"])].index)
    single_tiles = list(dens[dens == "1"].sample(4 if SMOKE else n_singles, random_state=0).index)
    if SMOKE:
        dense_tiles = dense_tiles[:6]
    rows = []
    for tiles in (dense_tiles, single_tiles):
        for i in range(0, len(tiles), 32):
            chunk = tiles[i:i + 32]
            for p, pred in zip(chunk, detect(m, [read(CROPS / t) for t in chunk])):
                keep = pred["scores"] >= 0.5
                pb, ps = pred["boxes"][keep], pred["scores"][keep]
                g = torch.tensor(boxes.loc[boxes.image_path == p, ["xmin", "ymin", "xmax", "ymax"]].values,
                                 dtype=torch.float32)
                best = box_iou(pb, g).max(1).values if len(pb) else torch.zeros(0)
                for b, s, iou in zip(pb.tolist(), ps.tolist(), best.tolist()):
                    rows.append({"image_path": p, "dens": dens[p], "score": s, "best_gt_iou": iou,
                                 "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    r = pd.DataFrame(rows)
    r["unmatched"] = r.best_gt_iou < 0.4
    r["near_gt"] = r.best_gt_iou.between(0.05, 0.4)  # overlaps a GT box but too loosely: box-fit, not a missing label
    r.to_csv(OUT / "0b_label_audit.csv", index=False)
    n_gt = boxes.groupby("image_path").size()
    tiles = pd.Series(dense_tiles + single_tiles)
    per = pd.DataFrame({"dens": dens[tiles].values, "n_gt": n_gt[tiles].values}).groupby("dens", observed=True).sum()
    u = r[r.unmatched]
    per["unmatched_ge_050"] = u.groupby("dens", observed=True).size()
    per["unmatched_ge_070"] = u[u.score >= 0.7].groupby("dens", observed=True).size()
    per["of_which_loose_fit"] = u[u.near_gt].groupby("dens", observed=True).size()
    per["unmatched_per_100_gt"] = (100 * per.unmatched_ge_050 / per.n_gt).round(2)
    print("\n== 0b: confident predictions matching no GT (IoU < 0.4), VALIDATION tiles ==")
    print(per.fillna(0).to_string())

    # Contact sheets: 160px context around each unmatched >= 0.5 box. Red = prediction, green = GT.
    sheets = OUT / "0b_sheets"
    sheets.mkdir(exist_ok=True)
    for group, sub in (("dense", u[u.dens != "1"]), ("single", u[u.dens == "1"])):
        sub = sub.sort_values("score", ascending=False).head(192)
        for page in range(0, len(sub), 48):
            sheet = Image.new("RGB", (8 * 164, 6 * 180), "white")
            for j, row in enumerate(sub.iloc[page:page + 48].itertuples()):
                img = Image.open(CROPS / row.image_path).convert("RGB")
                cx, cy = (row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2
                x0, y0 = int(np.clip(cx - 80, 0, 840)), int(np.clip(cy - 80, 0, 840))
                d = ImageDraw.Draw(img)
                for gb in boxes.loc[boxes.image_path == row.image_path].itertuples():
                    d.rectangle([gb.xmin, gb.ymin, gb.xmax, gb.ymax], outline=(0, 255, 0))
                d.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline=(255, 0, 0), width=2)
                tile = img.crop((x0, y0, x0 + 160, y0 + 160))
                X, Y = (j % 8) * 164, (j // 8) * 180
                sheet.paste(tile, (X, Y))
                ImageDraw.Draw(sheet).text((X + 2, Y + 162), f"{row.score:.2f} {row.dens} iou{row.best_gt_iou:.2f}",
                                           fill=(0, 0, 0))
            sheet.save(sheets / f"{group}_{page // 48:02d}.png")
    print(f"0b: contact sheets in {sheets}")


# ---------------------------------------------------------------- 0d training-set fit
def train_fit(m, per_bin: int = 1000) -> None:
    """Density strata on TRAINING tiles (up to per_bin tiles per stratum, seed 0) at 800px."""
    gt = load_gt(TRAIN)
    dens = density(gt[~is_empty(gt)])
    tiles = pd.concat([s.sample(min(4 if SMOKE else per_bin, len(s)), random_state=0)
                       for _, s in dens.groupby(dens, observed=True)]).index
    sub = gt[gt.image_path.isin(tiles)]
    csv = OUT / "0d_train_sample.csv"
    sub.to_csv(csv, index=False)
    m.model.score_thresh = 0.3
    p = m.predict_file(csv_file=str(csv), root_dir=str(CROPS))
    p["image_path"] = p.image_path.map(lambda x: Path(x).name)
    p.to_csv(OUT / "0d_train_predictions.csv", index=False)
    table = strata_table(p, sub)
    table.to_csv(OUT / "0d_train.density_strata.csv", index=False)
    print(f"\n######## 0d TRAINING-tile fit ({len(tiles):,} tiles, stratified; 'all' is NOT a population number) ########")
    report(table)


# ---------------------------------------------------------------- 0c re-predict
def repredict(m, gt: pd.DataFrame) -> None:
    variants = {"base800": (800, 800, 0.05), "native1000": (1000, 1000, 0.05),
                "up1200": (1200, 1200, 0.05), "nms020": (800, 800, 0.20)}
    val_csv, tiles = VAL, set(gt.image_path)
    if SMOKE:
        tiles = set(pd.Series(sorted(tiles)).sample(12, random_state=0))
        val_csv = OUT / "smoke_val.csv"
        pd.read_csv(VAL, low_memory=False).loc[lambda d: d.image_path.map(lambda x: Path(x).name).isin(tiles)].to_csv(val_csv, index=False)
    summary = []
    for name, (lo, hi, nms) in variants.items():
        m.model.transform.min_size, m.model.transform.max_size, m.model.nms_thresh = (lo,), hi, nms
        m.model.score_thresh = 0.3
        p = m.predict_file(csv_file=str(val_csv), root_dir=str(CROPS))
        p["image_path"] = p.image_path.map(lambda x: Path(x).name)
        p.to_csv(OUT / f"0c_{name}_predictions.csv", index=False)
        table = strata_table(p, gt[gt.image_path.isin(tiles)])
        table.to_csv(OUT / f"0c_{name}.density_strata.csv", index=False)
        print(f"\n######## 0c {name}: resize {lo}/{hi}, nms {nms} ########")
        report(table)
        summary.append(table.assign(variant=name))
    m.model.transform.min_size, m.model.transform.max_size, m.model.nms_thresh = (800,), 1333, 0.05
    s = pd.concat(summary)
    s = s[s.threshold.isin([0.4, 0.7])].pivot_table(index=["threshold", "dens"], columns="variant",
                                                    values="recall", sort=False)
    print("\n== 0c: VALIDATION pooled recall by variant ==")
    print(s.round(3).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_model(sys.argv[1] if len(sys.argv) > 1 else CKPT)
    print(f"checkpoint {sys.argv[1] if len(sys.argv) > 1 else CKPT}")
    print(f"transform {m.model.transform.min_size}/{m.model.transform.max_size}, nms {m.model.nms_thresh}")
    gt = load_gt(VAL)
    paste_test(m, gt)
    label_audit(m, gt)
    train_fit(m)
    repredict(m, gt)
