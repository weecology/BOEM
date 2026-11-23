"""Dense-flock annotation set A: validation dense tiles for independent double annotation.

Purpose: measure how often two annotators agree on flock birds (the human ceiling for dense
recall), and produce corrected labels for the dense validation stratum -- the only dense
benchmark available, since every dense USGS frame on disk is already in train/val and BOEM
survey flights hold almost no flocks (checked 2026-11-23).

Selection (seed 0): VALIDATION tiles from test_max_empty_0_25.csv, quotas per birds-per-tile
stratum, spread round-robin across sources so no single survey dominates. Excludes tiles
already in local Label Studio CSVs and boem-9c's high-loss upload (pending in Review).

Writes the tile list, their existing GT (for scoring agreement later) and preview sheets.
Nothing is uploaded: the upload (blind, no preannotations, its own project with 2 annotators per
task so it never enters the routine train/val download) is a separate step, run after Ben's OK.

    uv run python scripts/build_dense_annotation_set.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CROPS = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
VAL = CROPS / "test_max_empty_0_25.csv"
OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/annotation_set_A")
ANNOTATIONS = Path("/blue/ewhite/b.weinstein/BOEM/annotations")
HIGH_LOSS = Path("/blue/ewhite/b.weinstein/BOEM/high_loss_tiles/top50_per_split_uploaded.csv")
QUOTAS = {"6-10": 12, "11-20": 16, "21-50": 10, ">50": 2}
BINS = [5, 10, 20, 50, float("inf")]


def already_annotated() -> set[str]:
    names = set(pd.read_csv(HIGH_LOSS).image_path.map(lambda p: Path(p).name))
    for csv in ANNOTATIONS.rglob("*.csv"):
        col = pd.read_csv(csv, usecols=lambda c: c == "image_path")  # some side files have no image_path
        if "image_path" in col:
            names |= set(col.image_path.map(lambda p: Path(str(p)).name))
    return names


def round_robin(tiles: pd.DataFrame, n: int) -> pd.DataFrame:
    """n tiles, cycling over sources (largest first) so each source contributes before any repeats."""
    shuffled = tiles.sample(frac=1, random_state=0)
    shuffled["rank"] = shuffled.groupby("source").cumcount()
    return shuffled.sort_values(["rank", "source"]).head(n).drop(columns="rank")


def sheet(sel: pd.DataFrame, boxes: pd.DataFrame, path: Path, cols: int = 5, px: int = 320) -> None:
    rows = -(-len(sel) // cols)
    out = Image.new("RGB", (cols * (px + 4), rows * (px + 18)), "white")
    for i, t in enumerate(sel.itertuples()):
        img = Image.open(CROPS / t.image_path).convert("RGB")
        d = ImageDraw.Draw(img)
        for b in boxes[boxes.image_path == t.image_path].itertuples():
            d.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=(0, 255, 0), width=2)
        X, Y = (i % cols) * (px + 4), (i // cols) * (px + 18)
        out.paste(img.resize((px, px)), (X, Y))
        ImageDraw.Draw(out).text((X + 2, Y + px + 2), f"{t.n_gt} GT  {t.dens}  {str(t.source)[-6:]}", fill=(0, 0, 0))
    out.save(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(VAL, low_memory=False)
    gt["image_path"] = gt.image_path.map(lambda p: Path(p).name)
    boxes = gt[~((gt.xmin == 0) & (gt.xmax == 0) & (gt.ymin == 0) & (gt.ymax == 0))]
    tiles = boxes.groupby("image_path").agg(n_gt=("xmin", "size"), source=("source", "first"),
                                            bname_parent=("bname_parent", "first")).reset_index()
    tiles["source"] = tiles.source.fillna("NA")
    tiles["dens"] = pd.cut(tiles.n_gt, BINS, labels=list(QUOTAS)).astype(str)
    seen = already_annotated()
    eligible = tiles[tiles.dens.isin(QUOTAS) & ~tiles.image_path.isin(seen)]
    print(f"eligible dense val tiles (>= 6 birds, not already in LS / high-loss upload): {len(eligible)} "
          f"of {tiles.dens.isin(QUOTAS).sum()}")
    sel = pd.concat([round_robin(eligible[eligible.dens == d], q) for d, q in QUOTAS.items()])
    sel = sel.sort_values("n_gt").reset_index(drop=True)
    sel.to_csv(OUT / "tiles.csv", index=False)
    boxes[boxes.image_path.isin(sel.image_path)].to_csv(OUT / "existing_gt.csv", index=False)
    print(sel.groupby("dens", sort=False).agg(tiles=("n_gt", "size"), birds=("n_gt", "sum"),
                                              sources=("source", "nunique")).to_string())
    print(f"total: {len(sel)} tiles, {sel.n_gt.sum():,} existing GT birds per annotator pass, "
          f"{sel.source.nunique()} sources, {sel.bname_parent.nunique()} parent frames")
    for i in range(0, len(sel), 20):
        sheet(sel.iloc[i:i + 20], boxes, OUT / f"preview_{i // 20:02d}.png")
    print(f"wrote {OUT}/tiles.csv, existing_gt.csv, preview_*.png")
