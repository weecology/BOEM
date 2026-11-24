"""Score detector checkpoints at a given input size on every dense-flock axis.

For each (checkpoint, size) run:
  VALIDATION  density strata on test_max_empty_0_25.csv          (raw predictions, score_thresh 0.3)
  TRAIN FIT   density strata on the 0d stratified training sample (phase0/0d_train_sample.csv)
  ZERO-SHOT   threshold_sweep.py on the pinned holdout, same args as 52121265 / 52502538
  JULY LOAD   review_load_by_threshold.py's 5,000-frame sample through generate_pool_predictions + human_review

Input size is set as training set it (USGS_backbone.py --input-size -> min_size = max_size = size), because a plain
load_from_checkpoint ignores the checkpoint's stored "input_size" and falls back to torchvision's 800/1333.
size 800 keeps production's 800/1333.

    uv run python scripts/score_detector_at_size.py <label> <checkpoint> <size> [<label> <checkpoint> <size> ...]
Each run can skip axes with a 4th token per triple, e.g. "val,train" -- given as label:axes.
"""
import importlib.util
import random
import runpy
import sys
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from density_strata import CROPS, load_gt, report, strata_table  # noqa: E402
from src import detection  # noqa: E402

OUT = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/score_at_size")
VAL = CROPS / "test_max_empty_0_25.csv"
TRAIN_SAMPLE = Path("/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0d_train_sample.csv")
AXES = ("val", "train", "zeroshot", "july")
_load = detection.load


def sizes(size: int) -> tuple[int, int]:
    return (800, 1333) if size == 800 else (size, size)


def use_size(size: int) -> None:
    lo, hi = sizes(size)

    def load(*args, **kwargs):
        m = _load(*args, **kwargs)
        m.model.transform.min_size, m.model.transform.max_size = (lo,), hi
        return m
    detection.load = load


def tiles(label: str, ckpt: str, size: int) -> None:
    m = detection.load(checkpoint=ckpt, score_thresh=0.3)
    print(f"[{label}] transform {m.model.transform.min_size}/{m.model.transform.max_size}, "
          f"score_thresh {m.model.score_thresh}", flush=True)
    m.config["workers"], m.config["batch_size"] = 8, 32
    m.create_trainer(accelerator="gpu", devices=1, logger=False)
    for name, csv in (("val", VAL), ("train", TRAIN_SAMPLE)):
        if name not in RUN_AXES:
            continue
        p = m.predict_file(csv_file=str(csv), root_dir=str(CROPS))
        p["image_path"] = p.image_path.map(lambda x: Path(x).name)
        p.to_csv(OUT / f"{label}.{name}_predictions.csv", index=False)
        t = strata_table(p, load_gt(csv))
        t.to_csv(OUT / f"{label}.{name}.density_strata.csv", index=False)
        print(f"\n######## {label}: {'VALIDATION' if name == 'val' else 'TRAINING-tile sample (0d)'} ########")
        report(t)


def zero_shot(label: str, ckpt: str) -> None:
    sys.argv = ["threshold_sweep.py", "--checkpoint", ckpt, "--label", label,
                "--out", str(OUT / f"threshold_sweep_{label}.json"),
                "--queue-sample", "2000", "--batch-size", "1", "--workers", "5"]
    runpy.run_path(str(REPO / "scripts/threshold_sweep.py"), run_name="__main__")


def july(runs: list[tuple[str, str, int]]) -> None:
    spec = importlib.util.spec_from_file_location("review_load", REPO / "scripts/review_load_by_threshold.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    with initialize_config_dir(config_dir=str(REPO / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    random.seed(0)  # same 5,000-frame draw as 52348607 / 52502730
    flights = sorted(rl.IMAGERY.glob(rl.FLIGHT_GLOB))
    all_frames = {f: rl.frames(f) for f in flights}
    n_survey = sum(map(len, all_frames.values()))
    sample = {f: random.sample(v, min(rl.PER_FLIGHT, len(v))) for f, v in all_frames.items()}
    n_sampled = sum(map(len, sample.values()))
    crop, hcast = rl.load_models(cfg)
    for label, ckpt, size in runs:
        use_size(size)
        preds = rl.predict(ckpt, sample, cfg, crop, hcast)
        preds.to_csv(OUT / f"{label}.july_predictions.csv", index=False)
        s = rl.summarise(preds, n_sampled, n_survey, cfg)
        s.to_csv(OUT / f"{label}.july_summary.csv", index=False)
        print(f"\n=== July reviewer load, {label} (x{n_survey / n_sampled:.1f} to survey) ===")
        print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    runs = []
    for i in range(0, len(args), 3):
        label, axes = (args[i].split(":") + [",".join(AXES)])[:2]
        runs.append((label, args[i + 1], int(args[i + 2]), set(axes.split(","))))
    july_runs = []
    for label, ckpt, size, RUN_AXES in runs:
        use_size(size)
        if {"val", "train"} & RUN_AXES:
            tiles(label, ckpt, size)
        if "zeroshot" in RUN_AXES:
            print(f"\n######## {label}: ZERO-SHOT sweep ########", flush=True)
            zero_shot(label, ckpt)
        if "july" in RUN_AXES:
            july_runs.append((label, ckpt, size))
    if july_runs:
        july(july_runs)
