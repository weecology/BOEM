"""Per-tile detection loss on a detector's own train + validation tiles, and upload of the worst to Label Studio.

Two steps:

    # GPU: score every tile with the training loss (RetinaNet focal classification + box regression),
    # one value per tile, no augmentation, BatchNorm frozen. Writes <out_dir>/tile_losses.csv.
    uv run python scripts/find_high_loss_tiles.py score <checkpoint.ckpt> <train_csv> <val_csv>

    # CPU: upload the top-N tiles of EACH split (ranked within split) to the Review project with their
    # current ground-truth boxes as preannotations, so annotators correct the label rather than redraw it.
    uv run python scripts/find_high_loss_tiles.py upload --per-split 50 [--dry-run]

High loss on a training tile the model has seen for 30 epochs usually means the label is wrong
(missed animal, box on water, loose box); on a validation tile it can also just be a hard tile.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CROPS = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/high_loss_tiles")


def per_image_compute_loss(retinanet, store: list):
    """Replace RetinaNet.compute_loss so it records each image's loss (torchvision averages over the batch)."""
    original = retinanet.compute_loss

    def compute_loss(targets, head_outputs, anchors):
        for i in range(len(targets)):
            one = original([targets[i]], {k: v[i:i + 1] for k, v in head_outputs.items()}, [anchors[i]])
            store.append({k: float(v) for k, v in one.items()})
        return {k: torch.zeros((), device=anchors[0].device) for k in ("classification", "bbox_regression")}

    retinanet.compute_loss = compute_loss


def score(ckpt: str, train_csv: str, val_csv: str) -> None:
    from deepforest import main

    m = main.deepforest.load_from_checkpoint(ckpt, map_location="cuda")
    m.config["workers"] = 16
    m.to("cuda")
    m.model.train()  # loss is only computed in train mode
    for mod in m.model.modules():  # ...but keep BatchNorm on its running stats
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()

    rows, losses = [], []
    per_image_compute_loss(m.model, losses)  # wrap ONCE: re-wrapping chains to the old store
    for split, csv in (("val", val_csv), ("train", train_csv)):
        losses.clear()
        loader = m.load_dataset(csv, root_dir=str(CROPS), shuffle=False, batch_size=32)
        names = []
        with torch.no_grad():
            for b, (images, targets, image_names) in enumerate(loader):
                images = [im.to("cuda") for im in images]
                targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
                m.model(images, targets)
                names.extend(image_names)
                if b % 500 == 0:
                    print(f"{split}: {len(names):,} tiles", flush=True)
        assert len(names) == len(losses) and any(r["classification"] > 0 for r in losses)
        rows.append(pd.DataFrame(losses).assign(image_path=names, split=split, source_csv=csv))

    df = pd.concat(rows, ignore_index=True)
    df["loss"] = df["classification"] + df["bbox_regression"]
    gt = pd.concat([pd.read_csv(c, low_memory=False) for c in (train_csv, val_csv)])
    real = gt[~((gt.xmin == 0) & (gt.ymin == 0) & (gt.xmax == 0) & (gt.ymax == 0))]
    df["n_boxes"] = df.image_path.map(real.groupby("image_path").size()).fillna(0).astype(int)
    df["checkpoint"] = ckpt
    OUT_DIR.mkdir(exist_ok=True)
    df.sort_values("loss", ascending=False).to_csv(OUT_DIR / "tile_losses.csv", index=False)
    print(f"wrote {OUT_DIR / 'tile_losses.csv'}: {len(df):,} tiles")
    for split, g in df.groupby("split"):
        print(f"{split}: {len(g):,} tiles, loss mean {g.loss.mean():.3f}, p99 {g.loss.quantile(.99):.3f}, max {g.loss.max():.3f}")


def upload(per_split: int, dry_run: bool) -> None:
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir

    from src.annotators import get_annotator
    from src.label_studio import get_api_key

    losses = pd.read_csv(OUT_DIR / "tile_losses.csv")
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config", overrides=[f"image_dir={CROPS}"])
    annotator = get_annotator(cfg)

    existing = set()
    for instance in ("train", "validation", "review"):
        df = annotator.gather_data(instance_name=instance, image_dir=str(CROPS))
        if df is not None and not df.empty:
            existing.update(os.path.basename(str(x)) for x in df["image_path"].unique())
    # Ranked within split: val tiles are unseen, so a joint ranking would be val-dominated.
    fresh = losses[~losses.image_path.isin(existing)]
    sel = fresh.groupby("split", group_keys=False).head(per_split)
    print(f"Top {per_split} tiles per split ({losses.image_path.isin(existing).sum()} tiles already in Label Studio skipped):")
    print(sel[["image_path", "split", "loss", "classification", "bbox_regression", "n_boxes"]].to_string(index=False))
    print(sel.split.value_counts().to_string())

    gt = pd.concat([pd.read_csv(c, low_memory=False) for c in sel.source_csv.unique()])
    gt = gt[gt.image_path.isin(sel.image_path)].drop_duplicates(subset=["image_path", "xmin", "ymin", "xmax", "ymax"])
    boxes = gt[~((gt.xmin == 0) & (gt.ymin == 0) & (gt.xmax == 0) & (gt.ymax == 0))].copy()
    # Show the species in the taxonomy dropdown; the box itself is a generic Object.
    boxes["cropmodel_label"] = boxes["original_label"].fillna("Object")
    boxes["label"] = "Object"
    boxes["score"] = float("nan")
    boxes["cropmodel_score"] = float("nan")
    boxes["comet_id"] = "high_loss_" + Path(losses.checkpoint.iloc[0]).parent.name[:8]
    out = OUT_DIR / f"top{per_split}_per_split_uploaded.csv"
    sel.to_csv(out, index=False)
    print(f"{len(boxes)} GT boxes on {boxes.image_path.nunique()} tiles; {len(sel) - boxes.image_path.nunique()} tiles have none")
    if dry_run:
        return

    # Empty tiles go up with no preannotation so annotators can add what the model saw.
    preannotations = {b: g.reset_index(drop=True) for b, g in boxes.groupby("image_path")}
    images = [str(CROPS / b) for b in sel.image_path]
    annotator.upload(images=images, instance_name="review", preannotations=preannotations)
    print(f"Uploaded {len(images)} tiles to {cfg.annotation.label_studio.instances.review.project_name}; list in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("checkpoint")
    s.add_argument("train_csv")
    s.add_argument("val_csv")
    u = sub.add_parser("upload")
    u.add_argument("--per-split", type=int, default=50)
    u.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("high")
    score(a.checkpoint, a.train_csv, a.val_csv) if a.cmd == "score" else upload(a.per_split, a.dry_run)
