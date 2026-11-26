"""E1/E2 input: classifier logits for human-labelled boxes in frames held out of training.

Boxes come from the per-frame annotation CSVs (training/crops), restricted to frames
that have coordinates in the E0 frame index and are absent from the checkpoint's train
split. All Gulf boxes are kept; Atlantic is subsampled by frame for a comparable
control. Writes eval_boxes_<id>.csv + eval_logits_<id>.npy (row-aligned) and
eval_classes_<id>.txt under /blue/ewhite/b.weinstein/BOEM/geo_priors/.

Usage (1 GPU): python scripts/geo_priors_eval_logits.py [comet_id]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from deepforest.model import CropModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.USGS_classification import normalize_label  # noqa: E402

ROOT = Path("/blue/ewhite/b.weinstein/BOEM")
OUT = ROOT / "geo_priors"
TILES = ROOT / "training/crops"
CLS = ROOT / "training/classification"
EXPAND, ATLANTIC_BOXES, SEED = 30, 30_000, 0

comet_id = sys.argv[1] if len(sys.argv) > 1 else "0139af8cad264f569519cd06074acd66"
short = comet_id[:8]
stem = lambda s: re.sub(r"_\d+$", "", Path(s).stem)  # tile name -> frame basename

m = CropModel.load_from_checkpoint(CLS / f"checkpoints/buffer_30/{comet_id}.ckpt")
train_frames = set(
    pd.read_csv(CLS / f"checkpoints/buffer_30/{comet_id}/usgs_train_split.csv", usecols=["image_path"])
    .image_path.map(stem)
)

index = pd.read_csv(OUT / "frame_index.csv", parse_dates=["date"]).set_index("basename")
frames = index[index.region.isin(["Gulf", "Atlantic"])].index.difference(train_frames)
rng = np.random.default_rng(SEED)

parts = []
for f in frames:
    path = TILES / f"{f}.csv"
    if path.exists():
        parts.append(pd.read_csv(path, usecols=["xmin", "ymin", "xmax", "ymax", "label", "image_path"]))
boxes = pd.concat(parts, ignore_index=True)
boxes["label"] = boxes.label.map(normalize_label)
# (0,0,0,0) rows are detection-eval markers, not classifier data; USGS_classification drops them too.
boxes = boxes[(boxes.xmax > boxes.xmin) & boxes.label.isin(m.label_dict)].drop_duplicates().reset_index(drop=True)
boxes["basename"] = boxes.image_path.map(stem)
boxes = boxes.join(index[["lat", "lon", "date", "flight", "asset", "region", "subregion"]], on="basename")

atl = boxes[boxes.region == "Atlantic"]
keep_frames = rng.permutation(atl.basename.unique())
cum = atl.groupby("basename").size().reindex(keep_frames).cumsum()
atl_keep = set(cum[cum <= ATLANTIC_BOXES].index)
boxes = boxes[(boxes.region == "Gulf") | boxes.basename.isin(atl_keep)].reset_index(drop=True)
print(boxes.groupby(["region", "subregion"]).agg(boxes=("label", "size"), frames=("basename", "nunique")))


class Boxes(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with rasterio.open(TILES / r.image_path) as src:
            box = [max(0, v) for v in (r.xmin - EXPAND, r.ymin - EXPAND, r.xmax + EXPAND, r.ymax + EXPAND)]
            x0, y0, x1, y1 = (int(v) for v in m.expand_bbox_to_square(box, src.width, src.height))
            img = np.rollaxis(src.read(window=((y0, y1), (x0, x1))), 0, 3)
        return self.transform(Image.fromarray(img))


m = m.eval().cuda()
loader = torch.utils.data.DataLoader(Boxes(boxes, m.get_transform(augmentations=None)), batch_size=128, num_workers=4)
with torch.no_grad():
    logits = torch.cat([m.forward(x.cuda(), metadata=None).float().cpu() for x in loader]).numpy()
assert logits.shape == (len(boxes), len(m.label_dict))

classes = [c for c, _ in sorted(m.label_dict.items(), key=lambda kv: kv[1])]
boxes.to_csv(OUT / f"eval_boxes_{short}.csv", index=False)
np.save(OUT / f"eval_logits_{short}.npy", logits.astype(np.float32))
(OUT / f"eval_classes_{short}.txt").write_text("\n".join(classes))
pred = np.array(classes)[logits.argmax(1)]
print(boxes.assign(ok=pred == boxes.label.values).groupby("region").ok.agg(["size", "mean"]))
