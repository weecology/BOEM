"""Run the production detector + classifiers on a directory of images without the rest of the pipeline.

Mirrors Pipeline.run(): same Hydra config, same model loading, same generate_pool_predictions().
No Label Studio, no Comet, no .prediction_cache writes. Used to look at a new camera's output.

    uv run python scripts/predict_new_camera.py IMAGE_DIR OUT_DIR [LOW_SCORE]

LOW_SCORE (default 0.30) is how far down the detector is run so the score distribution is visible;
rows with score >= predict.min_score are the ones the pipeline would keep (column `in_pipeline`).
"""
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw
from deepforest.model import CropModel

from src import classification, detection, hierarchical
from src.active_learning import generate_pool_predictions
from src.pipeline import _collect_images

Image.MAX_IMAGE_PIXELS = None

image_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
low_score = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
out_dir.mkdir(parents=True, exist_ok=True)

with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "boem_conf"), version_base=None):
    cfg = compose(config_name="boem_config", overrides=[f"image_dir={image_dir}", "predict.batch_size=1"])

detector = detection.load(checkpoint=cfg.detection_model.checkpoint, score_thresh=low_score)
classifier = CropModel.load_from_checkpoint(cfg.classification_model.checkpoint)
classification.apply_temperature(classifier, cfg.classification_model.temperature)
classifier.config["cropmodel"]["expand"] = cfg.classification_model.expand
hcast = hierarchical.load_hcast_model(
    checkpoint_path=cfg.hierarchical.checkpoint,
    label_csv=cfg.hierarchical.label_csv,
    expand_pixels=int(cfg.hierarchical.expand),
    square=bool(cfg.hierarchical.square),
    eval_crop_ratio=cfg.hierarchical.eval_crop_ratio,
)

pool = _collect_images(str(image_dir))
print(f"{len(pool)} images; detector {cfg.detection_model.checkpoint}; classifier {cfg.classification_model.checkpoint}")
predictions = generate_pool_predictions(
    pool=pool,
    pool_limit=None,
    patch_size=cfg.active_learning.patch_size,
    patch_overlap=cfg.active_learning.patch_overlap,
    min_score=low_score,
    model=detector,
    batch_size=1,
    crop_model=classifier,
    hcast_model=hcast,
    image_dir=str(image_dir),
    hcast_batch_size=cfg.hierarchical.batch_size,
    hcast_workers=cfg.hierarchical.workers,
    workers=cfg.predict.workers,
)
predictions["in_pipeline"] = predictions["score"] >= cfg.predict.min_score
predictions.to_csv(out_dir / "predictions.csv", index=False)

floor = cfg.predict.min_score
kept = predictions[predictions.in_pipeline]
print(f"\nboxes >= {low_score}: {len(predictions)}   boxes >= pipeline floor {floor}: {len(kept)}")
print("score quantiles (all):", predictions.score.quantile([0.1, 0.5, 0.9, 1.0]).round(3).to_dict())
print(f"\ncropmodel_label counts at >= {floor}:\n{kept.cropmodel_label.value_counts().to_string()}")
print(f"\ncropmodel_label counts at >= {low_score} (not in pipeline):\n{predictions.cropmodel_label.value_counts().head(15).to_string()}")

# Overlay: pipeline boxes in red, sub-floor boxes in yellow, on a downscaled copy.
for path in pool:
    image = Image.open(path)
    scale = 3000 / max(image.size)
    overlay = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(overlay)
    rows = predictions[predictions.image_path.map(lambda p: Path(p).name) == Path(path).name]
    for r in rows.itertuples():
        draw.rectangle([r.xmin * scale, r.ymin * scale, r.xmax * scale, r.ymax * scale],
                       outline="red" if r.in_pipeline else "yellow", width=2)
    overlay.save(out_dir / f"{Path(path).stem}_overlay.jpg")
