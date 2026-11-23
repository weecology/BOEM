"""Dense-flock Phase 1: does matching the training distribution to dense flocks fix them?

Fine-tunes the production detector (c6ecec12/epoch27) under different training
distributions with identical optimisation (same tiles per epoch, epochs, lr, augmentations,
batch), then scores each on the full VALIDATION set by density stratum. Arms:

  none      no fine-tuning: the reference, scored the same way
  random    15,201 random training tiles (natural ~10% empty share)          -- the control
  dense     every training tile with >= 6 birds (13,681) + 1,520 empties (10%)
  weighted  all training tiles, sampled with weight min(birds, 20), empties held at 10%,
            15,201 draws per epoch                                           -- the deployable form

The random arm is the control because fine-tuning itself (lower lr, more steps) shifts
calibration; compare dense/weighted against random, at matched precision, per stratum.

Also scores the non-training remainder of the external dense frames (Feb 2022 cap probe,
flock_dataset test). Those frames come from training flights: VALIDATION-like, not zero-shot.

    uv run python scripts/dense_flock_finetune.py <arm> <seed>
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import kornia.augmentation as K
import torch
from deepforest import augmentations as df_aug
from deepforest import main
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from density_strata import CROPS, is_empty, load_gt, report, strata_table  # noqa: E402

# As in scripts/USGS_backbone.py: RandomAffine is not in DeepForest's registry.
df_aug._SUPPORTED_TRANSFORMS["RandomAffine"] = (K.RandomAffine, {"degrees": 0, "scale": (0.9, 1.1), "p": 0.2})

CKPT = ("/blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/"
        "epoch27-val_cls0.0867.ckpt")
TRAIN = CROPS / "train_max_empty_0_10.csv"
VAL = CROPS / "test_max_empty_0_25.csv"
OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/phase1")
DENSE_MIN = 6
EMPTY_SHARE = 0.10
EPOCHS, LR, BATCH = 8, 1e-4, 64
SMOKE = bool(os.environ.get("SMOKE"))  # CPU, a few steps, outputs under phase1/smoke
EXTERNAL = {
    "feb2022": (Path("/blue/ewhite/b.weinstein/BOEM/detection_cap_probe/feb2022/train_annotations.csv"),
                Path("/blue/ewhite/b.weinstein/BOEM/detection_cap_probe/feb2022/images")),
    "flock_test": (Path("/blue/ewhite/b.weinstein/src/BOEM/flock_dataset/test_annotations.csv"),
                   Path("/blue/ewhite/b.weinstein/src/BOEM/flock_dataset/images")),
}
if SMOKE:
    OUT = OUT / "smoke"
    EPOCHS, BATCH = 1, 2


def build_arms(train: pd.DataFrame) -> dict:
    """Arm tile lists (fixed seed 0, independent of the training seed) + per-tile weights."""
    n = train[~is_empty(train)].groupby("image_path").size()
    empties = pd.Index(sorted(set(train.image_path) - set(n.index)))
    dense = n[n >= DENSE_MIN].index
    n_empty = round(len(dense) * EMPTY_SHARE / (1 - EMPTY_SHARE))
    dense_tiles = dense.append(pd.Index(pd.Series(empties).sample(n_empty, random_state=0).values))
    per_epoch = len(dense_tiles)
    all_tiles = pd.Index(sorted(set(train.image_path)))
    w = pd.Series(np.minimum(n.reindex(all_tiles).fillna(0).values, 20), index=all_tiles, dtype=float)
    w[empties] = w[w > 0].sum() * EMPTY_SHARE / (1 - EMPTY_SHARE) / len(empties)
    dense_share = w[n.index[n >= DENSE_MIN]].sum() / w.sum()
    print(f"arms: per-epoch tiles {per_epoch:,}; dense arm {len(dense):,} >= {DENSE_MIN}-bird tiles + {n_empty:,} empties; "
          f"weighted arm draws >= {DENSE_MIN}-bird tiles {dense_share:.1%} of the time (uniform: "
          f"{len(dense) / len(all_tiles):.1%})")
    return {
        "random": (pd.Series(all_tiles).sample(per_epoch, random_state=0).values, None, per_epoch),
        "dense": (dense_tiles.values, None, per_epoch),
        "weighted": (all_tiles.values, w.values, per_epoch),
    }


def loader(m, train: pd.DataFrame, tiles, weights, per_epoch: int, tag: str, seed: int):
    csv = OUT / f"train_{tag}.csv"
    train[train.image_path.isin(set(tiles))].to_csv(csv, index=False)
    ds = m.load_dataset(csv_file=str(csv), root_dir=str(CROPS), augmentations=m.config.train.augmentations,
                        transforms=m.transforms, batch_size=BATCH).dataset
    g = torch.Generator().manual_seed(seed)
    if weights is not None:
        order = pd.Series(range(len(tiles)), index=tiles)[ds.image_names].values  # align weights to ds order
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights[order]), per_epoch,
                                                         replacement=True, generator=g)
    else:
        sampler = torch.utils.data.RandomSampler(ds, generator=g)
    if SMOKE:
        sampler = list(sampler)[:4]
    workers = 0 if SMOKE else m.config.workers
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, sampler=sampler, collate_fn=ds.collate_fn,
                                       num_workers=workers, pin_memory=not SMOKE,
                                       persistent_workers=workers > 0)


def fine_tune(arm: str, seed: int) -> main.deepforest:
    pl.seed_everything(seed)
    m = main.deepforest.load_from_checkpoint(CKPT, map_location="cpu")
    if arm == "none":
        return m
    train = load_gt(TRAIN)
    tiles, weights, per_epoch = build_arms(train)[arm]
    m.config["workers"] = 16
    m.config["batch_size"] = BATCH
    m.config["train"]["lr"] = LR
    m.config["train"]["epochs"] = EPOCHS
    m.config["train"]["scheduler"]["type"] = "constantLR"
    m.config["validation"]["csv_file"] = None
    m.existing_val_dataloader = None
    # Same augmentations as scripts/USGS_backbone.py (the run that produced c6ecec12).
    m.config = OmegaConf.merge(m.config, {"train": {"augmentations": [
        {"Rotate": {"degrees": 15, "p": 0.5}}, {"HorizontalFlip": {"p": 0.5}},
        {"VerticalFlip": {"p": 0.3}}, {"RandomAffine": {"degrees": 0, "scale": (0.9, 1.1), "p": 0.3}}]}})
    m.existing_train_dataloader = loader(m, train, tiles, weights, per_epoch, f"{arm}_s{seed}", seed)
    m.create_trainer(accelerator="cpu" if SMOKE else "gpu", devices=1, logger=False)
    m.trainer.fit(m)
    m.trainer.save_checkpoint(OUT / f"{arm}_s{seed}.ckpt")
    return m


def score(m, arm: str, seed: int) -> None:
    tag = f"{arm}_s{seed}"
    m.model.score_thresh = 0.3
    m.config["workers"], m.config["batch_size"] = (0, 2) if SMOKE else (8, 32)
    m.create_trainer(accelerator="cpu" if SMOKE else "gpu", devices=1, logger=False)
    gt = load_gt(VAL)
    val_csv = VAL
    if SMOKE:
        keep = set(pd.Series(sorted(set(gt.image_path))).sample(8, random_state=0))
        gt = gt[gt.image_path.isin(keep)]
        val_csv = OUT / "smoke_val.csv"
        gt.to_csv(val_csv, index=False)
    p = m.predict_file(csv_file=str(val_csv), root_dir=str(CROPS))
    p.to_csv(OUT / f"{tag}.val_predictions.csv", index=False)
    table = strata_table(p, gt)
    table.to_csv(OUT / f"{tag}.val.density_strata.csv", index=False)
    print(f"\n######## {tag}: VALIDATION tiles ########")
    report(table)

    train_parents = set(pd.read_csv(TRAIN, usecols=["bname_parent"]).bname_parent.dropna())
    for name, (csv, img_dir) in EXTERNAL.items():
        ext = pd.read_csv(csv)
        ext["image_path"] = ext.image_path.map(lambda x: Path(x).name)
        ext = ext[~ext.image_path.map(lambda x: Path(x).stem).isin(train_parents)]
        frames = sorted(set(ext.image_path))[: 1 if SMOKE else None]
        ext = ext[ext.image_path.isin(frames)]
        preds = [m.predict_tile(path=str(img_dir / f), patch_size=1000, patch_overlap=0) for f in frames]
        preds = [pd.DataFrame(x.drop(columns="geometry", errors="ignore")).assign(image_path=f)
                 for x, f in zip(preds, frames) if x is not None]
        # A frame with nothing above the floor is a real miss, scored as zero recall.
        preds = pd.concat(preds) if preds else pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "image_path"])
        preds.to_csv(OUT / f"{tag}.{name}_predictions.csv", index=False)
        t = strata_table(preds, ext)
        t.to_csv(OUT / f"{tag}.{name}.density_strata.csv", index=False)
        print(f"\n######## {tag}: {name}, {len(frames)} non-training FULL FRAMES "
              f"(validation-like; strata are birds per FRAME) ########")
        report(t)


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    assert arm in ("none", "random", "dense", "weighted"), arm
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    score(fine_tune(arm, seed), arm, seed)
