"""Inference input size for the production detector: what does 1000/1200 px cost outside validation?

Phase 0c showed predicting at 1000 (native) or 1200 (upsampled) instead of torchvision's default
800 raises VALIDATION recall by 7-11 pts at equal precision. Validation empties are capped at 25%
and contain no July foam, so this checks the two places a production change would bite, at each size:

  zero-shot   threshold_sweep.py on the pinned holdout flights (ZERO-SHOT box recall/precision,
              unscreened-queue hits) -- same args as c6ecec12's number of record (52121265).
  July load   review_load_by_threshold.py's 5,000-frame July sample (seed 0) through the pipeline's
              own generate_pool_predictions + human_review -> reviewer frames/boxes per threshold.

Both scripts are reused unchanged: detection.load is wrapped to set the RetinaNet transform size,
and outputs go to their own directory so earlier results are not overwritten.

    uv run python scripts/dense_flock_input_size.py [sizes ...]    # default 800 1000 1200
"""
import importlib.util
import random
import runpy
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import detection  # noqa: E402

CKPT = ("/blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/"
        "epoch27-val_cls0.0867.ckpt")
OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/input_size")
_load = detection.load


def use_size(size: int) -> None:
    """Make every detection.load() return a model that resizes to `size` (production is 800/1333).
    max_size scales with it so non-square edge patches keep production's aspect handling."""
    def load(*args, **kwargs):
        m = _load(*args, **kwargs)
        m.model.transform.min_size, m.model.transform.max_size = (size,), round(1333 * size / 800)
        return m
    detection.load = load


def zero_shot(size: int) -> Path:
    out = OUT / f"threshold_sweep_c6ecec12_e27_in{size}.json"
    sys.argv = ["threshold_sweep.py", "--checkpoint", CKPT, "--label", f"c6ecec12_e27_in{size}",
                "--out", str(out), "--queue-sample", "2000", "--batch-size", "1", "--workers", "5"]
    runpy.run_path(str(REPO / "scripts/threshold_sweep.py"), run_name="__main__")
    return out


def july_load(sizes: list[int]) -> None:
    spec = importlib.util.spec_from_file_location("review_load", REPO / "scripts/review_load_by_threshold.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)  # functions only; its __main__ block does not run
    with initialize_config_dir(config_dir=str(REPO / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    random.seed(0)  # same draw as 52348607
    flights = sorted(rl.IMAGERY.glob(rl.FLIGHT_GLOB))
    all_frames = {f: rl.frames(f) for f in flights}
    n_survey = sum(map(len, all_frames.values()))
    sample = {f: random.sample(v, min(rl.PER_FLIGHT, len(v))) for f, v in all_frames.items()}
    n_sampled = sum(map(len, sample.values()))
    print(f"July: {len(flights)} flights, {n_survey:,} frames; sampled {n_sampled:,}", flush=True)
    crop, hcast = rl.load_models(cfg)
    for size in sizes:
        use_size(size)
        preds = rl.predict(CKPT, sample, cfg, crop, hcast)
        preds.to_csv(OUT / f"july_predictions_in{size}.csv", index=False)
        s = rl.summarise(preds, n_sampled, n_survey, cfg)
        s.to_csv(OUT / f"july_summary_in{size}.csv", index=False)
        print(f"\n=== July reviewer load, c6ecec12 @ input {size}px (x{n_survey / n_sampled:.1f} to survey) ===")
        print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    sizes = [int(s) for s in sys.argv[1:]] or [800, 1000, 1200]
    OUT.mkdir(parents=True, exist_ok=True)
    sweeps = []
    for size in sizes:
        use_size(size)
        print(f"\n######## ZERO-SHOT sweep, input {size}px ########", flush=True)
        sweeps.append(zero_shot(size))
    july_load(sizes)
    sys.argv = ["threshold_sweep.py", "--report", *map(str, sweeps)]
    runpy.run_path(str(REPO / "scripts/threshold_sweep.py"), run_name="__main__")
