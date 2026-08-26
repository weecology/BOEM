"""Show frames around the land filter's decision boundary for visual sign-off.

    uv run python scripts/viz_land_boundary.py

Rescoring only needs the four cached features (struct, fine_edge, chroma, bg_frac) run
back through the CURRENT `land_model.json` -- no need to re-open any JPEGs to get an
up-to-date probability for frames already scored under an older model version. Frames
are drawn in bands around the threshold, thinned so no single flight/camera/moment
dominates (reusing `mine_land_examples.round_robin`), then rendered into one contact
sheet per band so a human can eyeball whether the threshold is cutting where the
collaborators want: water and mixed/beach frames kept, unambiguous suburban land cut.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mine_land_examples import round_robin
from src.land_filter import load_model

OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")
IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
REPORT_DIR = PROJECT_ROOT / "reports"
FEATURES = ["struct", "fine_edge", "chroma", "bg_frac"]

POOLS = ["manifest.csv", "scored_frames.csv", "new_flight_scores.csv", "new_flight_scores_deep.csv"]
THUMB = 220
N_PER_BAND = 10


def load_pool():
    dfs = [pd.read_csv(OUT_DIR / p)[["image", "flight", "camera", "frame_no", *FEATURES]]
           for p in POOLS]
    return pd.concat(dfs).drop_duplicates("image").reset_index(drop=True)


def score(df, model):
    x = df[model["features"]].to_numpy(np.float64)
    z = (x - model["mean"]) / model["scale"] @ model["coef"] + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def band_sheet(name, rows, thresh, out_path):
    font = ImageFont.load_default()
    tiles = []
    for r in rows:
        path = IMAGERY / r.flight / r.image
        im = Image.open(path).convert("RGB")
        im.draft("RGB", (THUMB * 2, THUMB * 2))
        im = im.resize((THUMB, THUMB))
        draw = ImageDraw.Draw(im)
        decision = "CUT" if r.prob > thresh else "keep"
        draw.rectangle([0, 0, THUMB, 16], fill=(0, 0, 0))
        draw.text((2, 2), f"p={r.prob:.3f} {decision}", fill=(255, 255, 0), font=font)
        draw.rectangle([0, THUMB - 12, THUMB, THUMB], fill=(0, 0, 0))
        draw.text((2, THUMB - 12), r.flight[:20], fill=(200, 200, 200), font=font)
        tiles.append(im)

    if not tiles:
        return
    cols = 5
    rows_n = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (THUMB * cols, THUMB * rows_n + 30), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{name}  (threshold {thresh:.3f})", fill=(255, 255, 255), font=font)
    for i, im in enumerate(tiles):
        x, y = (i % cols) * THUMB, 30 + (i // cols) * THUMB
        sheet.paste(im, (x, y))
    sheet.save(out_path)
    print(f"wrote {out_path} ({len(tiles)} frames)")


def main():
    model = load_model()
    thresh = model["threshold"]
    print(f"current threshold: {thresh:.3f}")

    pool = load_pool()
    pool["prob"] = score(pool, model)
    print(f"scored {len(pool)} frames from {pool.flight.nunique()} flights")

    # Bands relative to the operating point -- the two either side of it are the ones
    # collaborators actually need to see; the outer two are a sanity check that
    # confident water/land still look like water/land.
    bands = [
        ("1_confident_water", 0.0, 0.3),
        ("2_water_side", 0.3, 0.7),
        ("3_boundary_below_KEPT", 0.7, thresh),
        ("4_boundary_above_CUT", thresh, min(thresh + 0.08, 0.999)),
        ("5_confident_land_CUT", min(thresh + 0.08, 0.999), 1.01),
    ]

    REPORT_DIR.mkdir(exist_ok=True)
    taken = {}
    for name, lo, hi in bands:
        band = pool[(pool.prob >= lo) & (pool.prob < hi)].copy()
        band["absm"] = (band.prob - thresh).abs()
        sel = round_robin(band, N_PER_BAND, taken)
        if len(sel) < N_PER_BAND:
            print(f"  {name}: only {len(sel)} available (wanted {N_PER_BAND})")
        band_sheet(name, list(sel.itertuples()), thresh, REPORT_DIR / f"land_boundary_{name}.png")


if __name__ == "__main__":
    main()
