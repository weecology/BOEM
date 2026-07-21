"""Upload the April classifier's harbor-seal candidates to the Label Studio review project.

Context: on identical cached detection boxes, the April classifier (4c002d6b) calls 279 boxes
Phoca vitulina while the current classifier (d8995) calls zero — labelling most of them brown
pelican instead. These go to review so annotators can adjudicate which model is right.

Boxes come from scripts/reclassify_with_old_model.py output (old_label/new_label per box).
Uploads per flight because the annotator resolves images against cfg.image_dir.

Usage:
    uv run python scripts/upload_old_harbor_seals_to_review.py [--dry-run] [--label Phoca\\ vitulina]
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

RECLASS_CSV = "/blue/ewhite/b.weinstein/BOEM/detection_diag/reclassified_old_vs_new.csv"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reclass-csv", default=RECLASS_CSV)
    p.add_argument("--label", default="Phoca vitulina",
                   help="old_label to upload; use 'ALL_SEALS' for harbor+gray")
    p.add_argument("--instance", default="review")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would upload without touching Label Studio.")
    return p.parse_args()


def main():
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    df = pd.read_csv(args.reclass_csv)
    if args.label == "ALL_SEALS":
        sel = df[df["old_label"].astype(str).str.contains("Phoc|Halichoerus", na=False)].copy()
    else:
        sel = df[df["old_label"].astype(str) == args.label].copy()
    if sel.empty:
        raise SystemExit(f"No boxes with old_label={args.label!r} in {args.reclass_csv}")

    # Label Studio draws the box with this text; use the old model's call (what we're testing).
    sel["label"] = sel["old_label"]
    # label_studio_bbox_format puts comet_id in the task's model_version field, so tag these
    # as coming from the April classifier rather than a pipeline run.
    sel["comet_id"] = "old_classifier_4c002d6b_harbor_seal_qc"

    print(f"{len(sel)} boxes / {sel['image_path'].nunique()} images / "
          f"{sel['flight'].nunique()} flights  (old_label={args.label})")

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "boem_conf")):
        cfg = compose(config_name="boem_config")

    api_key = get_api_key()
    if api_key is None:
        raise SystemExit("No Label Studio API key found (.label_studio.config)")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    project_name = cfg.annotation.label_studio.instances[args.instance].project_name
    print(f"target project: {project_name}")

    total_imgs = 0
    for flight, fg in sel.groupby("flight"):
        # locate the flight's image dir (screened_images or imagery)
        image_dir = None
        for parent in ("screened_images", "imagery"):
            cand = f"/blue/ewhite/b.weinstein/BOEM/{parent}/{flight}"
            if os.path.isdir(cand):
                image_dir = cand
                break
        if image_dir is None:
            print(f"  [skip] {flight}: image dir not found")
            continue

        preannotations = {}
        for basename, g in fg.groupby(fg["image_path"].apply(os.path.basename)):
            if not os.path.exists(os.path.join(image_dir, basename)):
                continue
            preannotations[basename] = g[["image_path", "xmin", "ymin", "xmax", "ymax",
                                          "label", "score", "comet_id"]].reset_index(drop=True)
        if not preannotations:
            print(f"  [skip] {flight}: no images resolved on disk")
            continue

        image_paths = [os.path.join(image_dir, b) for b in sorted(preannotations)]
        n_boxes = sum(len(v) for v in preannotations.values())
        print(f"  {flight}: {len(image_paths)} images, {n_boxes} boxes")
        total_imgs += len(image_paths)

        if args.dry_run:
            continue

        cfg.image_dir = image_dir
        annotator = get_annotator(cfg)
        annotator.upload(images=image_paths, instance_name=args.instance,
                         preannotations=preannotations)

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone. {verb} {total_imgs} images to: {project_name}")


if __name__ == "__main__":
    main()
