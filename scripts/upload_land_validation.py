"""Upload held-out-flight frames to Label Studio WITH the model's guess pre-filled.

    uv run python scripts/score_flights.py            # writes new_flight_scores.csv
    uv run python scripts/upload_land_validation.py --dry-run
    uv run python scripts/upload_land_validation.py

This is a correction pass, not a labelling pass, and it deliberately inverts the choice
made in `upload_land_project.py`. That script hid the model's guess so annotators could
not anchor on it. Here the guess is the thing under test: each task arrives pre-filled
with the predicted class and its probability, and the annotator's job is to correct the
ones that are wrong. Anchoring is the accepted cost of only paying for corrections.

SAMPLING. Open water is ~all of a flight and ~none of the useful signal, so the sample
is deliberately not representative: it is concentrated above and just below the decision
threshold and pooled across eight flights, because land is rare enough that no single
flight yields enough of it. Two consequences worth remembering when reading the results:

  - The land/water ratio here says NOTHING about the true rate on a flight. Do not read
    "40% of the uploaded frames were land" as a prevalence estimate.
  - Precision on the `land_*` bands is the number that matters. Every predicted-land
    frame that is really water is a frame the filter would have thrown away, and those
    are the errors that can cost an animal. The `water_anchor` band is small on purpose:
    it is a tripwire for catastrophic failure, not a recall estimate.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mine_land_examples import round_robin
from src import label_studio as ls_mod
from src.label_studio import get_api_key
from src.land_filter import load_model

OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")
# The broad 5000-per-flight pass, plus the full-depth pass over the flights that pass
# turned out to be land-poor. Six of the eight yielded under 15 predicted-land frames
# at 5000 samples, which is not enough to tell a per-flight failure from noise -- and
# validating on the two land-rich flights alone would just repeat the single-flight
# weakness the training set already has.
SCORES = [OUT_DIR / "new_flight_scores.csv", OUT_DIR / "new_flight_scores_deep.csv"]
IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
# Label Studio caps project titles at 50 characters and rejects a longer one with a
# bare "400 Bad Request", so keep this short and check it below.
PROJECT_NAME = "BOEM - Land Screen Validation"
LS_MAX_TITLE = 50

# (name, lo, hi, n, order) over predicted probability; `thresh` is substituted at
# runtime. Weighted hard toward the land side -- see the sampling note above.
#
# `order` matters more than it looks. "boundary" takes the frames nearest the threshold
# first, which is right for the two bands that straddle it: that is where the model is
# least trustworthy. It is wrong for the two outer bands -- drawing them boundary-first
# just refills the middle, and a `water_anchor` made entirely of p~=0.30 frames is not a
# confident-water tripwire at all. Those use "spread": a random draw across the band.
# Sizes are set against measured supply (317,763 scored frames): after thinning there
# are 215 confident-land, 350 marginal-land and 1,529 boundary frames available, so the
# two land bands take a little over half of what exists and the outer bands are
# supply-rich. 65% of the budget goes to the land side because that is where the
# safety-critical error lives -- a predicted-land frame that is really water is a frame
# the filter would have discarded.
BANDS = [
    ("land_confident", 0.85, 1.01, 120, "spread"),      # is confident land really land?
    ("land_marginal", "thresh", 0.85, 140, "boundary"),  # what precision turns on
    ("boundary_below", 0.30, "thresh", 90, "boundary"),  # land we would keep (missed)
    ("water_anchor", 0.0, 0.30, 50, "spread"),          # tripwire, not a recall estimate
]

LABEL_CONFIG = """<View>
  <Header value="Correct the model's guess if it is wrong."/>
  <Text name="guide" value="Water = open water only, ANY colour or texture: blue, green, brown, grey glint, waves, whitecaps, foam, wakes, and dark seagrass or bottom patches in turbid shallows. Texture alone is NOT land. Land = any solid ground: beach, dune, marsh, forest, grass, buildings, roads, docks. Mixed = both present, e.g. a shoreline crossing the frame. Unusable = black, truncated, or too blurred to judge."/>
  <Text name="model_guess" value="Model says: $model_label (p(land)=$prob, band=$band, flight=$flight_name)"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" width="100%"/>
  <Choices name="surface" toName="image" choice="single-radio" required="true" showInLine="true">
    <Choice value="Water" hotkey="1" background="#1f77b4"/>
    <Choice value="Land" hotkey="2" background="#8c564b"/>
    <Choice value="Mixed" hotkey="3" background="#ff7f0e"/>
    <Choice value="Unusable" hotkey="4" background="#7f7f7f"/>
  </Choices>
</View>"""


def select(df, thresh):
    """Draw each band, round-robin across flights, thinning overlapping frames."""
    rng = np.random.default_rng(0)
    picked, taken = [], {}
    for name, lo, hi, n, order in BANDS:
        lo = thresh if lo == "thresh" else lo
        hi = thresh if hi == "thresh" else hi
        band = df[(df.prob > lo) & (df.prob <= hi)].copy()
        # round_robin consumes each flight in ascending "absm" order, so absm is just
        # the priority key: distance to the threshold, or a random draw across the band.
        band["absm"] = ((band.prob - thresh).abs() if order == "boundary"
                        else rng.random(len(band)))
        take = round_robin(band, n, taken)
        if len(take) < n:
            print(f"  {name}: only {len(take)} available (wanted {n})")
        picked.append(take.assign(band=name))
    return pd.concat(picked).drop(columns="absm").drop_duplicates("image").reset_index(drop=True)


def build_tasks(sel, model_version):
    tasks = []
    for r in sel.itertuples():
        label = "Land" if r.pred_land else "Water"
        tasks.append({
            "data": {
                "image": os.path.join("/data/local-files/?d=BOEM/input/", r.image),
                "flight_name": r.flight,
                "image_relative_path": f"{r.flight}/{r.image}",
                "model_label": label,
                "prob": round(float(r.prob), 3),
                "band": r.band,
            },
            "predictions": [{
                "model_version": model_version,
                "score": float(r.prob),
                "result": [{"from_name": "surface", "to_name": "image", "type": "choices",
                            "value": {"choices": [label]}}],
            }],
        })
    return tasks


def main():
    if len(PROJECT_NAME) > LS_MAX_TITLE:
        raise ValueError(
            f"PROJECT_NAME is {len(PROJECT_NAME)} chars; Label Studio allows "
            f"{LS_MAX_TITLE} and answers with an opaque 400 if you exceed it.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scores", default=None,
                    help="comma-separated score CSVs (default: both passes)")
    args = ap.parse_args()

    model = load_model()
    thresh = model["threshold"]
    paths = [Path(p) for p in args.scores.split(",")] if args.scores else SCORES
    present = [p for p in paths if p.exists()]
    if not present:
        raise FileNotFoundError(f"no score files found: {[str(p) for p in paths]}")
    for p in paths:
        print(f"{'  ' if p in present else 'MISSING '}{p}")
    # The deep pass re-scores frames the broad pass already covered; keep one row each.
    df = pd.concat([pd.read_csv(p) for p in present]).drop_duplicates("image")
    sel = select(df, thresh)
    sel.to_csv(OUT_DIR / "validation_manifest.csv", index=False)

    images = [str(IMAGERY / r.flight / r.image) for r in sel.itertuples()]
    missing = [p for p in images if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames not on disk, e.g. {missing[0]}")

    print(f"scored pool: {len(df)} frames, {df.pred_land.sum()} predicted land "
          f"({df.pred_land.mean():.2%}) at threshold {thresh:.3f}")
    print(f"\nselected {len(sel)} frames from {sel.flight.nunique()} flights")
    print(pd.crosstab(sel.band, sel.pred_land).to_string())
    print("\nper flight:")
    print(sel.flight.value_counts().to_string())
    print(f"\nwrote {OUT_DIR / 'validation_manifest.csv'}")

    model_version = f"land_filter_n{model['n_train']}_t{thresh:.3f}"
    if args.dry_run:
        print(f"\nDRY-RUN: nothing uploaded. model_version={model_version}")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    ls_cfg = cfg.annotation.label_studio

    project = ls_mod.connect_to_label_studio(
        url=ls_cfg.url, project_name=PROJECT_NAME, label_config=LABEL_CONFIG)
    sftp_client = ls_mod.create_sftp_client(
        user=cfg.server.user, host=cfg.server.host, key_filename=cfg.server.key_filename)
    ls_mod.upload_images(sftp_client=sftp_client, images=images, folder_name=ls_cfg.folder_name)

    tasks = build_tasks(sel, model_version)
    for i in range(0, len(tasks), 100):
        project.import_tasks(tasks[i:i + 100])
    print(f"\nUploaded {len(tasks)} frames to '{PROJECT_NAME}' (model_version={model_version}).")


if __name__ == "__main__":
    main()
