"""Reviewer workload vs detection score threshold, on a random sample of survey frames.

Re-predicts a random sample of frames (default: 250 per July 2026 flight) at a 0.40
floor with each detector, through the pipeline's own generate_pool_predictions
(production CropModel + temperature + H-CAST from boem_config), then applies the
pipeline's human_review() split at each threshold. Reports, per detector and threshold,
the frames/boxes that would reach human review (uncertain), be auto-annotated (confident)
or be ignored, scaled to the whole survey.

    uv run python scripts/review_load_by_threshold.py <detector.ckpt> [<detector.ckpt> ...]
"""

import random
import sys
from pathlib import Path

import pandas as pd
from deepforest.model import CropModel
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import active_learning, classification, detection, hierarchical  # noqa: E402

IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
OUT = Path("/blue/ewhite/b.weinstein/BOEM/review_load_by_threshold")
FLIGHT_GLOB = "JPG_202607*"
PER_FLIGHT = 250
FLOOR = 0.40
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def frames(flight: Path) -> list[str]:
    return sorted(str(p) for p in flight.iterdir() if p.suffix.lower() == ".jpg")


def load_models(cfg):
    crop = CropModel.load_from_checkpoint(cfg.classification_model.checkpoint)
    classification.apply_temperature(crop, cfg.classification_model.temperature)
    crop.config["cropmodel"]["expand"] = cfg.classification_model.expand
    h = cfg.hierarchical
    hcast = hierarchical.load_hcast_model(
        checkpoint_path=h.checkpoint, label_csv=h.label_csv, expand_pixels=int(h.expand or 0),
        square=bool(h.square), eval_crop_ratio=h.eval_crop_ratio,
    )
    return crop, hcast


def predict(ckpt: str, sample: dict[Path, list[str]], cfg, crop, hcast) -> pd.DataFrame:
    det = detection.load(checkpoint=ckpt, score_thresh=FLOOR)
    parts = []
    for flight, pool in sample.items():
        p = active_learning.generate_pool_predictions(
            pool=pool, pool_limit=None, patch_size=cfg.active_learning.patch_size,
            patch_overlap=cfg.active_learning.patch_overlap, min_score=FLOOR, model=det,
            batch_size=cfg.predict.batch_size, crop_model=crop, hcast_model=hcast,
            image_dir=str(flight), hcast_batch_size=cfg.hierarchical.batch_size,
            hcast_workers=cfg.hierarchical.workers, workers=cfg.predict.workers,
        )
        if p is not None and not p.empty:
            parts.append(p.assign(flight=flight.name))
        print(f"  {flight.name}: {0 if p is None else len(p)} boxes >= {FLOOR}", flush=True)
    return pd.concat(parts, ignore_index=True)


def summarise(preds: pd.DataFrame, n_sampled: int, n_survey: int, cfg) -> pd.DataFrame:
    hr = cfg.human_review
    rows = []
    for t in THRESHOLDS:
        conf, unc = active_learning.human_review(
            preds, min_detection_score=t, review_low=hr.review_low, review_high=hr.review_high,
            review_on_disagreement=hr.review_on_disagreement, review_all_below=0,
        )
        kept = preds[preds.score >= t]
        scale = n_survey / n_sampled
        rows.append({
            "threshold": t,
            "frames_any_box": kept.image_path.nunique(),
            "boxes_any": len(kept),
            "review_frames": unc.image_path.nunique(),
            "review_boxes": len(unc),
            "auto_boxes": len(conf),
            "ignored_boxes": len(kept) - len(conf) - len(unc),
            "survey_review_frames": round(unc.image_path.nunique() * scale),
            "survey_review_boxes": round(len(unc) * scale),
            "survey_frames_any_box": round(kept.image_path.nunique() * scale),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    random.seed(0)
    flights = sorted(IMAGERY.glob(FLIGHT_GLOB))
    all_frames = {f: frames(f) for f in flights}
    n_survey = sum(map(len, all_frames.values()))
    sample = {f: random.sample(v, min(PER_FLIGHT, len(v))) for f, v in all_frames.items()}
    n_sampled = sum(map(len, sample.values()))
    print(f"{len(flights)} flights, {n_survey:,} frames; sampled {n_sampled:,} ({PER_FLIGHT}/flight, seed 0)", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.Series([p for v in sample.values() for p in v]).to_csv(OUT / "sampled_frames.csv", index=False, header=["image_path"])
    crop, hcast = load_models(cfg)
    for ckpt in sys.argv[1:]:
        tag = f"{Path(ckpt).parent.name[:8]}_{Path(ckpt).name.split('-')[0]}"
        print(f"\n=== {tag}: {ckpt}", flush=True)
        preds = predict(ckpt, sample, cfg, crop, hcast)
        preds.to_csv(OUT / f"predictions_{tag}.csv", index=False)
        s = summarise(preds, n_sampled, n_survey, cfg)
        s.to_csv(OUT / f"summary_{tag}.csv", index=False)
        print(f"\n{tag}: sample {n_sampled:,} frames, scaled x{n_survey / n_sampled:.1f} to the {n_survey:,}-frame survey")
        print(s.to_string(index=False), flush=True)
