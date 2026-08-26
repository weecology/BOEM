"""Score a random sample of frames from flights the land filter has never seen.

    uv run python scripts/score_flights.py --n 5000 --out /path/scores.csv

Land is rare and clustered -- a flight line clips the coast for a few hundred frames
and is open water for the rest -- so a flight's land content is not something a small
sample estimates well. The point here is not to estimate prevalence but to FIND the
frames near and above the decision boundary, which is where the next round of
annotation has to be spent.

Scoring is ~440 ms of CPU per frame and reads the full 12 MB JPEG off Lustre, so this
belongs in a SLURM job with real cores: an interactive shell on this cluster gets one
CPU and the ProcessPoolExecutor is then inert. See submit_score_flights.sh.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.land_filter import land_features, land_probability, load_model

IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")

# Flights held out of the fit entirely (mine_land_examples.FLIGHTS has the other 12).
# Two per survey period, so a failure that is specific to one sea state or one season
# shows up as a per-flight effect rather than being averaged away.
NEW_FLIGHTS = [
    "JPG_20241219_131500", "JPG_20241220_104800",     # Dec 2024, the whitecap season
    "JPG_20260201_134000", "JPG_20260202_094800",     # Feb 2026
    "JPG_20260711_131000", "JPG_20260712_100400",     # Jul 2026
    "JPG_20260713_101500", "JPG_20260713_160300",
]


def score(path):
    f = land_features(path)
    return {"image": path.name, "flight": path.parent.name, **f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="frames sampled per flight")
    ap.add_argument("--flights", default=",".join(NEW_FLIGHTS))
    ap.add_argument("--out", default=str(OUT_DIR / "new_flight_scores.csv"))
    args = ap.parse_args()

    workers = len(os.sched_getaffinity(0))
    rng = np.random.default_rng(0)
    paths = []
    for flight in args.flights.split(","):
        images = sorted((IMAGERY / flight).glob("*.jpg"))
        if not images:
            print(f"WARNING: no frames for {flight}")
            continue
        take = min(args.n, len(images))
        paths += [images[i] for i in rng.choice(len(images), take, replace=False)]
        print(f"{flight}: sampling {take} of {len(images)}")

    print(f"\nscoring {len(paths)} frames on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        df = pd.DataFrame(pool.map(score, paths, chunksize=8))

    model = load_model()
    df["prob"] = [land_probability(r, model) for r in df.to_dict("records")]
    df["pred_land"] = df.prob > model["threshold"]
    # Frames ~1 s apart overlap heavily; the selector needs these to thin runs of
    # near-identical frames rather than paying to annotate the same stretch of coast.
    parts = df.image.str.extract(r"^(?P<camera>C\d+)_L\d+_F(?P<frame_no>\d+)_")
    df["camera"] = parts.camera
    df["frame_no"] = parts.frame_no.astype(int)

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"threshold {model['threshold']:.3f} -> {df.pred_land.sum()}/{len(df)} "
          f"predicted land ({df.pred_land.mean():.2%})")
    print("\npredicted-land rate per flight:")
    print(df.groupby("flight").pred_land.agg(["sum", "count", "mean"]).to_string())


if __name__ == "__main__":
    main()
