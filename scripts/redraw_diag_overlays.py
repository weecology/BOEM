"""Redraw clean detection overlays from the saved diagnostic boxes (CPU only).

Draws the top-K highest-scoring epoch16 boxes per seal image so we can judge
whether the sub-0.85 detections actually land on seals. Emits downscaled JPEGs.
"""
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DIAG = "/blue/ewhite/b.weinstein/BOEM/detection_diag"
IMGDIR = "/blue/ewhite/b.weinstein/BOEM/screened_images/JPG_20230426_110600"
OUT = os.path.join(DIAG, "clean_overlays")
os.makedirs(OUT, exist_ok=True)

boxes = pd.read_csv(os.path.join(DIAG, "boxes_balanced.csv"))
maxscore = boxes.groupby("image")["score"].max().sort_values(ascending=False)

# pick a spread: 6 highest-confidence images + 4 mid-confidence
picks = list(maxscore.head(6).index) + list(maxscore.iloc[30:34].index)

TOPK = 6

def band(s):
    if s >= 0.85: return (0, 220, 0)       # green: would survive 0.85
    if s >= 0.5:  return (255, 165, 0)      # orange: 0.5-0.85
    return (255, 40, 40)                    # red: 0.3-0.5

manifest = []
for img in picks:
    p = os.path.join(IMGDIR, img)
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im)
    sub = boxes[boxes["image"] == img].sort_values("score", ascending=False).head(TOPK)
    for _, r in sub.iterrows():
        s = float(r["score"])
        c = band(s)
        dr.rectangle([r["xmin"], r["ymin"], r["xmax"], r["ymax"]], outline=c, width=5)
        dr.text((r["xmin"] + 2, max(0, r["ymin"] - 14)), f"{s:.2f}", fill=c)
    im.thumbnail((760, 760))
    out = os.path.join(OUT, f"clean_{img}.jpg")
    im.save(out, "JPEG", quality=80)
    manifest.append((img, round(float(sub["score"].max()), 3), out, f"{W}x{H}"))
    print(f"{img}: top6 max={sub['score'].max():.3f} -> {out}", flush=True)

pd.DataFrame(manifest, columns=["image", "top_score", "overlay", "orig_size"]).to_csv(
    os.path.join(OUT, "manifest.csv"), index=False)
print("done", len(manifest))
