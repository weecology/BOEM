"""Confusion matrix for the live metadata-free classifier (56e8585) on its own val split.

Asks two things:
  1. Does Morus bassanus get absorbed into Somateria mollissima?
  2. What does a 70-way forced choice with no background class do to precision?

Run under SLURM with one GPU; writes matrix + per-crop predictions next to the imagery.
"""
import numpy as np
import pandas as pd
import torch
from deepforest.model import CropModel
from pytorch_lightning import Trainer
from torchvision.datasets import ImageFolder

CKPT = "/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/56e8585add144d1eabba1f00c411b985.ckpt"
VAL = "/blue/ewhite/b.weinstein/BOEM/training/classification/crops/val/buffer_30/56e8585add144d1eabba1f00c411b985"
OUT = "/blue/ewhite/b.weinstein/BOEM/classifier_confusion_56e8585"


class ImagesOnly(torch.utils.data.Dataset):
    """ImageFolder yields (image, label); predict_step reads a 2-tuple as (images, metadata)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


m = CropModel.load_from_checkpoint(CKPT)
val_ds = ImageFolder(root=VAL, transform=m.get_transform(augmentations=None))

# The checkpoint's own label mapping must agree with the folder ordering, or every
# label below is silently permuted.
assert m.label_dict == val_ds.class_to_idx, (
    f"label_dict mismatch: model has {len(m.label_dict)} classes, "
    f"folders have {len(val_ds.class_to_idx)}"
)

trainer = Trainer(devices=1, accelerator="gpu", logger=False, enable_checkpointing=False)
results = trainer.predict(m, m.predict_dataloader(ImagesOnly(val_ds)))
label, score = m.postprocess_predictions(results)

df = pd.DataFrame({
    "path": [p for p, _ in val_ds.samples],
    "true": [m.numeric_to_label_dict[y] for _, y in val_ds.samples],
    "pred": [m.numeric_to_label_dict[i] for i in label],
    "score": np.asarray(score),
})
df.to_csv(f"{OUT}_predictions.csv", index=False)
pd.crosstab(df.true, df.pred).to_csv(f"{OUT}_matrix.csv")

print(f"val crops: {len(df)}  classes: {len(val_ds.classes)}")
print(f"overall accuracy: {(df.true == df.pred).mean():.4f}")

def _recall(d):
    wrong = d[d.true != d.pred]
    return pd.Series({
        "n": len(d),
        "recall": (d.true == d.pred).mean(),
        "top_confusion": wrong.pred.value_counts().index[0] if len(wrong) else "-",
        "top_conf_n": wrong.pred.value_counts().iloc[0] if len(wrong) else 0,
        "mean_score": d.score.mean(),
    })


per = pd.concat({k: _recall(g) for k, g in df.groupby("true")}, axis=1).T.sort_values("recall")
print("\n=== per-class recall (worst first) ===")
print(per.round(3).to_string())

def _prec(d):
    wrong = d[d.true != d.pred]
    return pd.Series({
        "n_predicted": len(d),
        "precision": (d.true == d.pred).mean(),
        "top_source": wrong.true.value_counts().index[0] if len(wrong) else "-",
        "mean_score": d.score.mean(),
    })


prec = pd.concat({k: _prec(g) for k, g in df.groupby("pred")}, axis=1).T.sort_values(
    "n_predicted", ascending=False)
print("\n=== precision view: where do predictions come from ===")
print(prec.round(3).to_string())

print("\n=== Morus bassanus / Somateria mollissima ===")
sub = df[df.true.isin(["Morus bassanus", "Somateria mollissima"])]
ct = pd.crosstab(sub.true, sub.pred)
print(ct.loc[:, ct.sum() > 0].to_string())

print("\n=== does high confidence mean correct? ===")
for t in [0.0, 0.6, 0.9, 0.95, 0.99]:
    k = df[df.score >= t]
    print(f"  score>={t:.2f}: n={len(k):5d}  accuracy={(k.true == k.pred).mean():.3f}")

print(f"\nwrote {OUT}_matrix.csv and {OUT}_predictions.csv")
