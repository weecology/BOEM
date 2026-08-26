"""Hard-mine land/water frames for annotation, to fit a learned land filter.

`src/land_filter.py` currently uses four hand-tuned thresholds fitted to 88 frames
labelled by eye. That is thin, and the Dec-2024 whitecaps showed the thresholds do not
transfer across sea states. This script selects the frames worth paying an annotator
for: the ones nearest the current decision boundary, where the labels actually move
the fit.

Mining uses the signed log-margin to the current rule's conjunction boundary:

    m1 = log(struct / MIN_STRUCT)              m2 = log(MAX_EDGE_RATIO * struct / fine_edge)
    m3 = log(chroma / MIN_CHROMA)              m4 = log(MAX_BG_FRAC / bg_frac)
    margin = min(m1..m4)

which is >0 exactly when the rule says land, and whose absolute value is the distance
to the boundary. Frames are drawn in bands of |margin| -- densest at the boundary,
with a thinner tail of confident anchors so the fit is pinned at both ends -- and
capped per flight so no single flight or sea state dominates.

    uv run python scripts/mine_land_examples.py

Writes the manifest to /blue/ewhite/b.weinstein/BOEM/annotations/land_screen/.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.land_filter import (MAX_BG_FRAC, MAX_EDGE_RATIO, MIN_CHROMA,
                             MIN_STRUCT, land_features)

IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")

# Spans both seasons and all four survey periods: Dec-2024 (whitecaps, the hard
# negative), Feb-2026, and the Jul-2026 flights including the coastal line that
# actually contains land.
FLIGHTS = [
    "JPG_20241219_120500", "JPG_20241219_150200", "JPG_20241220_145900",
    "JPG_20260201_093500", "JPG_20260202_122400", "JPG_20260202_141900",
    "JPG_20260710_155800", "JPG_20260710_163500", "JPG_20260711_090800",
    "JPG_20260711_141200", "JPG_20260712_083900", "JPG_20260713_121500",
]
FRAMES_PER_FLIGHT = 350          # scored per flight, before selection

# (label, |margin| range, n from the land side, n from the water side)
BANDS = [
    ("boundary", 0.00, 0.15, 60, 60),   # where the labels change the fit
    ("near", 0.15, 0.40, 50, 50),
    ("anchor", 0.40, np.inf, 40, 40),   # pin both ends
]


MIN_FRAME_GAP = 5   # frames ~1s apart overlap heavily; don't pay to label both


def round_robin(pool, n, taken):
    """Take n rows closest to the boundary, cycling flights so none dominates.

    Selecting purely by distance-to-boundary concentrates on whichever flights happen
    to have the most ambiguous frames -- an early run drew 74% of the set from two
    flights and only 19 frames from the Dec-2024 whitecap flights, which are precisely
    the hard negatives the fit needs.

    `taken` maps (flight, camera) -> list of frame numbers already selected, and is
    updated in place so the separation rule holds across bands as well as within one.
    """
    groups = {f: g.sort_values("absm").reset_index(drop=True)
              for f, g in pool.groupby("flight")}
    rows, depth = [], 0
    while len(rows) < n and any(len(g) > depth for g in groups.values()):
        for g in groups.values():
            if len(rows) >= n:
                break
            if len(g) <= depth:
                continue
            row = g.iloc[depth]
            key = (row.flight, row.camera)
            if any(abs(row.frame_no - f) < MIN_FRAME_GAP for f in taken.get(key, ())):
                continue
            taken.setdefault(key, []).append(row.frame_no)
            rows.append(row)
        depth += 1
    return pd.DataFrame(rows)


def score(path):
    return {"image": path.name, "flight": path.parent.name, **land_features(path)}


def sample_flight(flight, rng):
    images = sorted((IMAGERY / flight).glob("*.jpg"))
    if not images:
        return []
    take = min(FRAMES_PER_FLIGHT, len(images))
    return [images[i] for i in rng.choice(len(images), take, replace=False)]


def main():
    # Scoring is I/O bound -- 12 MB per frame off Lustre, ~25 min for 4200 frames --
    # so reuse the cached scores when only the selection is being changed.
    cache = OUT_DIR / "scored_frames.csv"
    if cache.exists():
        print(f"reusing {cache} (delete it to rescore)")
        df = pd.read_csv(cache)
    else:
        rng = np.random.default_rng(0)
        paths = [p for f in FLIGHTS for p in sample_flight(f, rng)]
        print(f"scoring {len(paths)} frames from {len(FLIGHTS)} flights")
        with ProcessPoolExecutor(16) as pool:
            df = pd.DataFrame(pool.map(score, paths, chunksize=16))

    eps = 1e-9
    margins = np.stack([
        np.log(df.struct / MIN_STRUCT + eps),
        np.log((MAX_EDGE_RATIO * df.struct) / (df.fine_edge + eps) + eps),
        np.log(df.chroma / MIN_CHROMA + eps),
        np.log(MAX_BG_FRAC / (df.bg_frac + eps) + eps),
    ])
    df["margin"] = margins.min(0)
    df["rule_says"] = np.where(df.margin > 0, "land", "water")
    parts = df.image.str.extract(r"^(?P<camera>C\d+)_L\d+_F(?P<frame_no>\d+)_")
    df["camera"] = parts.camera
    df["frame_no"] = parts.frame_no.astype(int)
    df.to_csv(OUT_DIR / "scored_frames.csv", index=False)

    picked, taken = [], {}
    for name, lo, hi, n_land, n_water in BANDS:
        band = df[(df.margin.abs() >= lo) & (df.margin.abs() < hi)]
        for side, n in (("land", n_land), ("water", n_water)):
            pool_ = band[band.rule_says == side].assign(absm=lambda d: d.margin.abs())
            take = round_robin(pool_, n, taken)
            picked.append(take.assign(band=name, side=side))
            if len(take) < n:
                print(f"  {name}/{side}: only {len(take)} available (wanted {n})")

    sel = pd.concat(picked).drop(columns="absm").drop_duplicates("image")
    sel = sel.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle so
    # annotators cannot infer the label from task order
    sel.to_csv(OUT_DIR / "manifest.csv", index=False)

    print(f"\nselected {len(sel)} frames -> {OUT_DIR / 'manifest.csv'}")
    print(pd.crosstab(sel.band, sel.rule_says).to_string())
    print("\nper flight:")
    print(sel.flight.value_counts().to_string())


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
