"""Screen out frames that are over land before running the detector.

The detector is trained on marine targets and produces open-set false positives on
land (roofs, cars, dune vegetation), which then flood human review. Hue is not a
usable signal -- Gulf water is often green, and sun glint renders it grey -- so this
filter keys on structure instead:

    water is one material, land is many

Frames are scored on a 64x64 box-averaged thumbnail, which averages waves and glint
away while preserving objects, plus a 256x256 thumbnail that retains them:

    struct     coarse-scale structure, after high-passing out the sun-angle gradient
    fine_edge  fine-scale gradient energy: waves, chop and glint
    chroma     spread of colour, i.e. how many distinct materials are present
    bg_frac    fraction of the frame that is a single background colour

Land must satisfy all four: real coarse structure; that structure *not* explained by
chop (land concentrates energy at coarse scale, water spreads it across scales); more
than one material; and no single dominant background. Any one test alone is weak --
the ratio test alone passes 176/269 water frames -- but the conjunction holds.

Validated on 751 frames sampled from three flights, 88 of them labelled by eye:
24/30 land caught, 0/58 water lost. Thresholds are deliberately conservative because
the errors are asymmetric -- dropping a water frame can lose an animal, while letting
a land frame through only preserves the status quo.

CAVEAT: whitecaps in the Dec-2024 flight are the hard negative that sets these
thresholds; they cost ~20% of land recall. Re-check flag rates on any flight with a
markedly different sea state, and see `scripts/screen_land.py --report`.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MIN_STRUCT = 0.012
MAX_EDGE_RATIO = 0.70
MIN_CHROMA = 0.030
MAX_BG_FRAC = 0.65


def land_features(path, coarse=64, fine=256):
    """Structure/texture/colour statistics used to separate land from water."""
    im = Image.open(path).convert("RGB")
    im.draft("RGB", (coarse * 8, coarse * 8))  # JPEG DCT-domain downscale, ~10x faster
    fine_gray = np.asarray(im.resize((fine, fine), Image.BILINEAR), np.float32).mean(-1) / 255.0
    # BOX averaging is what removes the waves; BILINEAR would alias them through.
    a = np.asarray(im.resize((coarse, coarse), Image.BOX), np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray = a.mean(-1)
    high_pass = gray - ndimage.gaussian_filter(gray, 4)  # drop the illumination gradient

    px = a.reshape(-1, 3)
    median_colour = np.median(px, 0)

    return {
        "struct": float(high_pass.std()),
        "fine_edge": float(np.hypot(*np.gradient(fine_gray)).mean()),
        "chroma": float(np.std(r - g) + np.std(g - b)),
        "bg_frac": float((np.linalg.norm(px - median_colour, axis=1) < 0.05).mean()),
    }


def is_land(path):
    """True when the frame is over land and should be skipped by the detector."""
    f = land_features(path)
    return (
        f["struct"] > MIN_STRUCT
        and f["fine_edge"] < MAX_EDGE_RATIO * f["struct"]
        and f["chroma"] > MIN_CHROMA
        and f["bg_frac"] < MAX_BG_FRAC
    )


def filter_water(paths, workers=8):
    """Split image paths into (water, land). Both keep the input order."""
    from concurrent.futures import ProcessPoolExecutor

    paths = list(paths)
    with ProcessPoolExecutor(workers) as pool:
        flags = list(pool.map(is_land, paths, chunksize=16))
    water = [p for p, land in zip(paths, flags) if not land]
    land = [p for p, land in zip(paths, flags) if land]
    return water, land


# --- learned filter -------------------------------------------------------
# The four thresholds above are hand-tuned and were fitted by eye. The logistic
# regression below is fitted to Label Studio annotations by
# `scripts/fit_land_filter.py`, which writes MODEL_PATH; on the 2026-08-24 labels
# it dominates the rule on both axes (see the JOB_LEDGER entry). The rule is kept
# because `scripts/mine_land_examples.py` mines against its decision boundary.

MODEL_PATH = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen/land_model.json")


@lru_cache(maxsize=4)
def load_model(path=MODEL_PATH):
    """Load the fitted coefficients. Cached: this is called per frame."""
    m = json.loads(Path(path).read_text())
    return {**m,
            "coef": np.asarray(m["coef"], np.float64),
            "mean": np.asarray(m["scaler_mean"], np.float64),
            "scale": np.asarray(m["scaler_scale"], np.float64)}


def land_probability(features, model=None):
    """P(land) for one `land_features` dict, on the fitted model's scale."""
    m = model or load_model()
    x = np.array([features[k] for k in m["features"]], np.float64)
    z = float(np.dot((x - m["mean"]) / m["scale"], m["coef"]) + m["intercept"])
    return float(1.0 / (1.0 + np.exp(-z)))


def is_land_learned(path, model=None):
    """True when the fitted model calls the frame land at its stored threshold."""
    m = model or load_model()
    return land_probability(land_features(path), m) > m["threshold"]
