"""Score a flight's frames for land/water and write the result next to the imagery.

    uv run python scripts/screen_land.py /blue/ewhite/b.weinstein/BOEM/imagery/JPG_20260710_155800

Writes `land_screen.csv` (per-frame features and the land flag) into the image
directory. Pass --report to print the feature distribution instead of writing, which
is how you sanity-check the thresholds against a new sea state.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.land_filter import (MAX_BG_FRAC, MAX_EDGE_RATIO, MIN_CHROMA,
                             MIN_STRUCT, land_features)


def score(path):
    return {"image": path.name, **land_features(path)}


image_dir = Path(sys.argv[1])
report = "--report" in sys.argv
images = sorted(image_dir.glob("*.jpg"))

with ProcessPoolExecutor(16) as pool:
    df = pd.DataFrame(pool.map(score, images, chunksize=16))

df["land"] = (
    (df.struct > MIN_STRUCT)
    & (df.fine_edge < MAX_EDGE_RATIO * df.struct)
    & (df.chroma > MIN_CHROMA)
    & (df.bg_frac < MAX_BG_FRAC)
)
print(f"{image_dir.name}: {df.land.sum()}/{len(df)} land ({100 * df.land.mean():.1f}%)")

if report:
    df["ratio"] = df.fine_edge / df.struct
    print(df[["struct", "ratio", "chroma", "bg_frac"]].describe(
        percentiles=[.05, .5, .95, .99]).round(4).to_string())
else:
    out = image_dir / "land_screen.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
