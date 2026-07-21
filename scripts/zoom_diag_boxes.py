"""Extract zoomed crops around epoch16's top detections on known-seal images.

For each of several seal images, crop a padded window around the top-N boxes at
full resolution and upscale, so a human can judge whether the detection is a seal.
Saves individual zoom JPEGs + a montage.
"""
import os
import pandas as pd
from PIL import Image, ImageDraw

DIAG = "/blue/ewhite/b.weinstein/BOEM/detection_diag"
IMGDIR = "/blue/ewhite/b.weinstein/BOEM/screened_images/JPG_20230426_110600"
OUT = os.path.join(DIAG, "zooms")
os.makedirs(OUT, exist_ok=True)

boxes = pd.read_csv(os.path.join(DIAG, "boxes_balanced.csv"))
maxscore = boxes.groupby("image")["score"].max().sort_values(ascending=False)
picks = list(maxscore.head(12).index)  # 12 highest-confidence seal images

PAD = 90          # px of context around each box at full res
ZOOM = 3          # upscale factor
TOP_PER_IMG = 2

tiles = []
for img in picks:
    p = os.path.join(IMGDIR, img)
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert("RGB")
    W, H = im.size
    sub = boxes[boxes["image"] == img].sort_values("score", ascending=False).head(TOP_PER_IMG)
    for i, (_, r) in enumerate(sub.iterrows()):
        cx = (r["xmin"] + r["xmax"]) / 2
        cy = (r["ymin"] + r["ymax"]) / 2
        l = max(0, int(cx - PAD)); t = max(0, int(cy - PAD))
        rgt = min(W, int(cx + PAD)); b = min(H, int(cy + PAD))
        crop = im.crop((l, t, rgt, b))
        # draw the box within the crop coords
        dr = ImageDraw.Draw(crop)
        dr.rectangle([r["xmin"] - l, r["ymin"] - t, r["xmax"] - l, r["ymax"] - t],
                     outline=(0, 255, 0), width=2)
        crop = crop.resize((crop.width * ZOOM, crop.height * ZOOM), Image.NEAREST)
        out = os.path.join(OUT, f"zoom_{img}_{i}_s{r['score']:.2f}.jpg")
        crop.save(out, "JPEG", quality=85)
        tiles.append((crop, f"{img[:22]} s={r['score']:.2f}"))

# montage grid
if tiles:
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw = max(c.width for c, _ in tiles)
    th = max(c.height for c, _ in tiles) + 16
    grid = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    dr = ImageDraw.Draw(grid)
    for idx, (c, cap) in enumerate(tiles):
        gx = (idx % cols) * tw
        gy = (idx // cols) * th
        grid.paste(c, (gx, gy))
        dr.text((gx + 3, gy + c.height + 2), cap, fill=(255, 255, 255))
    grid.save(os.path.join(OUT, "montage.jpg"), "JPEG", quality=85)
    print("montage:", os.path.join(OUT, "montage.jpg"), "tiles:", len(tiles))
print("done")
