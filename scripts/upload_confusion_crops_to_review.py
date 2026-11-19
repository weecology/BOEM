"""Upload the validation crops behind a confusion-matrix cell to Label Studio review.

Reads a ``scripts/classifier_confusion.py`` predictions CSV and selects every val crop
whose true OR predicted label is in ``--labels``, i.e. the correct calls and both sides
of each confusion. Each val crop is named ``<UBFAI crop stem>_<n>.png``; the UBFAI crop
it came from (often a flock tile) is uploaded with ALL of its USGS boxes as
preannotations, so the annotator re-checks the whole tile, not one box.

Usage:
    uv run python scripts/upload_confusion_crops_to_review.py \
        /blue/ewhite/b.weinstein/BOEM/classifier_confusion_a3dc30a0_predictions.csv \
        --labels "Pelecanus occidentalis" "Ardea alba" --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotators import get_annotator
from src.label_studio import get_api_key

UBFAI_CROPS_DIR = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
SUFFIX = r"_\d+\.png$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions_csv")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    preds = pd.read_csv(args.predictions_csv)
    sel = preds[preds["true"].isin(args.labels) | preds["pred"].isin(args.labels)].copy()
    sel["parent"] = sel["base"].str.replace(SUFFIX, ".png", regex=True)
    sel["csv"] = sel["parent"].str.replace(SUFFIX, ".csv", regex=True)
    print(f"{len(sel)} val crops touch {args.labels}; {sel['parent'].nunique()} parent UBFAI crops")
    print(pd.crosstab(sel["true"], sel["pred"]).stack().loc[lambda s: s > 0].to_string())

    boxes = pd.concat(
        [pd.read_csv(UBFAI_CROPS_DIR / c).query("image_path in @parents")
         for c, parents in sel.groupby("csv")["parent"].unique().items()],
        ignore_index=True,
    ).drop_duplicates(subset=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config", overrides=[f"image_dir={UBFAI_CROPS_DIR}"])
    annotator = get_annotator(cfg)

    existing = set()
    for instance in ("train", "validation", "review"):
        df = annotator.gather_data(instance_name=instance, image_dir=str(UBFAI_CROPS_DIR))
        if df is not None and not df.empty:
            existing.update(os.path.basename(str(x)) for x in df["image_path"].unique())
    boxes["image_path"] = boxes["image_path"].map(lambda x: os.path.basename(str(x)))
    skipped = set(boxes["image_path"]) & existing
    boxes = boxes[~boxes["image_path"].isin(existing)].copy()

    # Show the USGS label in the taxonomy dropdown; the box itself is a generic Object.
    boxes["cropmodel_label"] = boxes["label"]
    boxes["label"] = "Object"
    boxes["score"] = float("nan")
    boxes["cropmodel_score"] = float("nan")
    boxes["comet_id"] = "confusion_qc_" + Path(args.predictions_csv).stem.split("_")[2]

    print(f"Skipping {len(skipped)} parents already annotated in Label Studio")
    print(f"To upload: {boxes['image_path'].nunique()} crop images, {len(boxes)} boxes")
    print(boxes["cropmodel_label"].value_counts().to_string())
    if args.dry_run:
        return

    preannotations = {b: g.reset_index(drop=True) for b, g in boxes.groupby("image_path")}
    images = [str(UBFAI_CROPS_DIR / b) for b in sorted(preannotations)]
    annotator.upload(images=images, instance_name="review", preannotations=preannotations)
    print(f"Uploaded {len(images)} images to {cfg.annotation.label_studio.instances.review.project_name}")


if __name__ == "__main__":
    main()
