"""Test whether a small pretrained CNN beats the 4-feature logistic regression.

    uv run python scripts/fit_land_cnn.py

The logistic regression in `fit_land_filter.py` runs on 4 hand-engineered texture/
colour statistics -- "water is one material, land is many" -- which cannot tell beach
sand (structured, low-chroma, like land) from a parking lot (flat and grey, like
water); see the JOB_LEDGER entry for the two frames that exposed this. A CNN can look
at shape (roofs, rows of cars, wave texture) instead of only summary statistics, and a
backbone pretrained on ImageNet already knows how to see edges and objects, so it
should need far fewer labelled frames than training one from scratch -- the whole
point, given there are only 62 labels.

Approach, tuned for "62 labels, 1 CPU":
  - MobileNetV3-Small, ImageNet weights. Smallest common torchvision backbone.
  - Freeze the first 10 of 13 feature blocks (generic edge/texture filters transfer
    fine unmodified); fine-tune the last 3 blocks + a fresh linear head. Full transfer
    (freeze everything, fit only the head) is the fallback -- FREEZE_ALL below -- since
    with this little data the extra capacity may not earn its keep.
  - Same 5-fold StratifiedKFold CV as fit_land_filter.py, same frame split, so the
    reported AUC is directly comparable to the 0.971 logistic regression number.
  - Images are decoded and augmented once per fold-epoch from disk; with 1 CPU and
    ~48 train / 14 val frames per fold this is the bottleneck, not the network itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")
IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
EXCLUDE = {"C5_L1_F1034_T20260202_124404_130.jpg"}  # see fit_land_filter.py

FREEZE_ALL = False   # True = linear probe on frozen features; False = fine-tune tail
UNFROZEN_BLOCKS = 3  # last N of 13 feature blocks trained when FREEZE_ALL is False
IMG_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # nadir imagery has no "up"
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(**NORM),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(**NORM),
])


def load_image(path):
    """Fast JPEG-DCT downscale, matching land_filter.land_features."""
    im = Image.open(path).convert("RGB")
    im.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))
    return im


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, train):
        self.images, self.labels, self.train = images, labels, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = load_image(self.images[i])
        tf = TRAIN_TF if self.train else EVAL_TF
        return tf(im), torch.tensor([self.labels[i]], dtype=torch.float32)


def build_model():
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(1024, 1)
    if FREEZE_ALL:
        cut = len(m.features)
    else:
        cut = len(m.features) - UNFROZEN_BLOCKS
    for i, block in enumerate(m.features):
        for p in block.parameters():
            p.requires_grad = i >= cut
    return m


def run_fold(train_idx, val_idx, images, y):
    model = build_model()
    train_ds = FrameDataset([images[i] for i in train_idx], y[train_idx], train=True)
    val_ds = FrameDataset([images[i] for i in val_idx], y[val_idx], train=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, yb in train_loader:
            opt.zero_grad()
            loss = lossf(model(x), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        print(f"    epoch {epoch + 1}/{EPOCHS}  train loss {total / len(train_idx):.3f}")

    if len(val_idx) == 0:
        return np.array([]), model

    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE):
            probs.append(torch.sigmoid(model(x)).squeeze(-1))
    return torch.cat(probs).numpy(), model


def main():
    torch.manual_seed(0)
    labels = pd.read_csv(OUT_DIR / "annotations.csv")
    feats = pd.read_csv(OUT_DIR / "manifest.csv")[["image", "flight"]]
    d = feats.merge(labels, on="image")
    d = d[d.label.isin(["Land", "Water"]) & ~d.image.isin(EXCLUDE)].reset_index(drop=True)

    images = [IMAGERY / r.flight / r.image for r in d.itertuples()]
    y = (d.label == "Land").astype(int).to_numpy()
    print(f"fitting on {len(d)} frames ({y.sum()} land, {len(y) - y.sum()} water)")
    print(f"config: FREEZE_ALL={FREEZE_ALL} UNFROZEN_BLOCKS={UNFROZEN_BLOCKS} "
          f"epochs={EPOCHS} lr={LR}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        print(f"  fold {fold + 1}/5 ({len(train_idx)} train, {len(val_idx)} val)")
        oof[val_idx], _ = run_fold(train_idx, val_idx, images, y)

    print(f"\ncross-validated ROC-AUC {roc_auc_score(y, oof):.3f}  "
          f"(logistic-regression baseline: 0.971)")

    grid = np.arange(0.01, 1.0, 0.005)
    affordable = [t for t in grid if (oof[y == 0] > t).mean() == 0]
    thresh = float(min(affordable)) if affordable else 1.0
    print(f"zero-water-loss operating point {thresh:.3f}: "
          f"land recall {(oof[y == 1] > thresh).mean():.1%}, "
          f"water lost {(oof[y == 0] > thresh).sum()}/{(y == 0).sum()}")

    print("\nfitting final model on all labelled frames...")
    all_idx = np.arange(len(y))
    _, final_model = run_fold(all_idx, all_idx[:0], images, y)  # empty val = train on everything
    torch.save(final_model.state_dict(), OUT_DIR / "land_cnn.pt")
    (OUT_DIR / "land_cnn_meta.json").write_text(json.dumps({
        "img_size": IMG_SIZE, "freeze_all": FREEZE_ALL, "unfrozen_blocks": UNFROZEN_BLOCKS,
        "cv_auc": float(roc_auc_score(y, oof)), "threshold": thresh, "n_train": int(len(d)),
    }, indent=2))
    print(f"wrote {OUT_DIR / 'land_cnn.pt'}")

    # The two frames that exposed the logistic regression's feature limitation.
    probe = {
        "beach (JPG_20241220_104800, logreg p=0.99, should stay IN)":
            IMAGERY / "JPG_20241220_104800/C5_L3_F2991_T20241220_120033_953.jpg",
        "parking lot (JPG_20260710_155800, logreg p=0.67, should be CUT)":
            IMAGERY / "JPG_20260710_155800/C1_L2_F256_T20260710_162548_140.jpg",
    }
    final_model.eval()
    print("\nprobe frames:")
    with torch.no_grad():
        for name, path in probe.items():
            x = EVAL_TF(load_image(path)).unsqueeze(0)
            p = torch.sigmoid(final_model(x)).item()
            print(f"  {name}\n    CNN p(land)={p:.3f}")


if __name__ == "__main__":
    main()
