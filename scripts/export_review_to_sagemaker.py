"""Export the completed first-pass Review queue to a SageMaker Ground Truth package.

Pulls every annotated task from the Label Studio project
`Bureau of Ocean Energy Management - Review` (the July 2026 Gulf survey human-review
queue) and writes, per flight, the file set the UMESC SageMaker pipeline consumes:

    <stamp>_<flight>_annotation.csv   one row per first-pass box, geometry only
    <stamp>_<flight>_roster.txt       s3_uri / status / views
    <stamp>_<flight>_metadata.txt     frame basenames
    <stamp>_<flight>_jobs.txt         the frames handed out this batch

into annotations/sagemaker/input/ .

The box GEOMETRY is the human first-pass set (model detections as adjusted/added by the
UF reviewers). The box LABEL is left blank on purpose: the second pass is an independent
blind check, so neither the Object/FalsePositive verdict nor the first-pass species is
carried across. Those first-pass verdicts are saved separately to
annotations/sagemaker/second_pass_<stamp>/firstpass_verdicts.csv (NOT uploaded) so the
two-pass agreement report can run once results come back.

Images are not transferred here -- the script writes second_pass_<stamp>/images_to_stage.txt
(local absolute paths under IMAGERY_BASE) for a manual Globus push to the UMESC-UF Pipeline
collection / the s3://umesc-ocs-label/uf-production/images/ prefix.

Usage:
    uv run python scripts/export_review_to_sagemaker.py --dry-run
    uv run python scripts/export_review_to_sagemaker.py
"""
from __future__ import annotations

import argparse
import os
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import sagemaker_gt as sm
from src.label_studio import get_api_key

IMAGERY_BASE = "/blue/ewhite/b.weinstein/BOEM/imagery"
REVIEW_PROJECT_NAME = "Bureau of Ocean Energy Management - Review"
S3_PREFIX = "s3://umesc-ocs-label/uf-production/images/"
DEFAULT_W, DEFAULT_H = 6464, 4852


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--imagery-base", default=IMAGERY_BASE)
    p.add_argument("--s3-prefix", default=S3_PREFIX)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where the per-flight annotation/roster/metadata/jobs files go "
        "(default: <repo>/../BOEM annotations/sagemaker/input, resolved from config).",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    return p.parse_args()


def _species_by_region(result: list) -> dict:
    out = {}
    for r in result or []:
        if r.get("type") == "taxonomy":
            tx = r["value"].get("taxonomy") or []
            if tx:
                out[r.get("id")] = tx[0][-1]
    return out


def main() -> None:
    args = _parse_args()
    stamp = dt.datetime.utcnow().strftime("%Y%m%d")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Label Studio API key in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    ls_url = cfg.annotation.label_studio.url

    if args.output_dir:
        input_dir = args.output_dir
    else:
        review_csv_dir = str(cfg.annotation.label_studio.instances.review.csv_dir)
        annotations_base = os.path.dirname(review_csv_dir)
        input_dir = os.path.join(annotations_base, "sagemaker", "input")
        second_pass_dir = os.path.join(annotations_base, "sagemaker", f"second_pass_{stamp}")

    from label_studio_sdk import Client

    ls = Client(url=ls_url, api_key=api_key)
    ls.check_connection()
    project = next(
        p for p in ls.list_projects() if p.get_params()["title"] == REVIEW_PROJECT_NAME
    )
    tasks = project.get_tasks()
    annotated = [t for t in tasks if t.get("annotations")]
    print(f"{REVIEW_PROJECT_NAME}: {len(annotated)}/{len(tasks)} tasks annotated")

    rows = []
    for t in annotated:
        data = t.get("data") or {}
        flight = data.get("flight_name", "UNKNOWN")
        basename = os.path.basename(
            data.get("image_relative_path") or data.get("image", "")
        )
        ann = t["annotations"][0]
        result = ann.get("result", [])
        species = _species_by_region(result)
        # region id -> taxonomy leaf; rectangles carry the same id
        for r in result or []:
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            ow = r.get("original_width") or DEFAULT_W
            oh = r.get("original_height") or DEFAULT_H
            rows.append(
                {
                    "flight_name": flight,
                    "image_path": basename,
                    "xmin": round(v["x"] / 100.0 * ow, 2),
                    "ymin": round(v["y"] / 100.0 * oh, 2),
                    "xmax": round((v["x"] + v["width"]) / 100.0 * ow, 2),
                    "ymax": round((v["y"] + v["height"]) / 100.0 * oh, 2),
                    "first_pass_label": (v.get("rectanglelabels") or [""])[0],
                    "first_pass_species": species.get(r.get("id"), ""),
                    "annotator": ann.get("created_username", "").strip(),
                }
            )

    boxes = pd.DataFrame(rows)
    if boxes.empty:
        print("No boxes found -- aborting.")
        return

    n_frames = boxes["image_path"].nunique()
    print(
        f"{len(boxes)} first-pass boxes across {n_frames} frames / "
        f"{boxes['flight_name'].nunique()} flights "
        f"({(boxes.first_pass_label == 'Object').sum()} Object, "
        f"{(boxes.first_pass_label == 'FalsePositive').sum()} FalsePositive)"
    )

    # ---- staging list + missing-image check -------------------------------------
    stage_paths, missing = [], []
    for flight, g in boxes.groupby("flight_name"):
        for b in sorted(g["image_path"].unique()):
            fp = os.path.join(args.imagery_base, flight, b)
            (stage_paths if os.path.isfile(fp) else missing).append(fp)
    print(f"images on disk: {len(stage_paths)}   missing: {len(missing)}")
    for m in missing[:10]:
        print(f"  MISSING {m}")

    if args.dry_run:
        print("\n[dry-run] would write per-flight files to", input_dir)
        for flight, g in boxes.groupby("flight_name"):
            print(f"  {stamp}_{flight}_annotation.csv  ({len(g)} boxes, "
                  f"{g['image_path'].nunique()} frames)")
        return

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(second_pass_dir, exist_ok=True)

    # ---- private first-pass verdict sidecar (NOT uploaded) ---------------------
    verdict_path = os.path.join(second_pass_dir, "firstpass_verdicts.csv")
    boxes.to_csv(verdict_path, index=False)

    # ---- per-flight SageMaker package ----------------------------------------
    written = []
    s3_prefix = args.s3_prefix.rstrip("/")
    for flight, g in boxes.groupby("flight_name"):
        basenames = sorted(g["image_path"].unique())
        s3_uris = [f"{s3_prefix}/{b}" for b in basenames]

        # geometry only -- blank label, no model hint
        pre = g[["image_path", "xmin", "ymin", "xmax", "ymax"]].copy()
        pre["label"] = ""

        roster_path = sm.write_daily_roster(
            s3_uris=s3_uris, output_dir=input_dir, stamp=stamp, flight_name=flight
        )
        jobs_path, selected = sm.assign_jobs_from_roster(
            roster_path=roster_path, output_dir=input_dir, num_jobs=len(s3_uris),
            stamp=stamp, flight_name=flight,
        )
        meta_path = sm.write_daily_metadata(
            selected, output_dir=input_dir, stamp=stamp, flight_name=flight
        )
        csv_path = sm.write_daily_annotation_csv(
            s3_uris,
            output_dir=input_dir,
            flight_path=flight,
            instance_type="review",
            stamp=stamp,
            preannotations=pre,
            flight_name=flight,
        )
        written.extend([roster_path, jobs_path, meta_path, csv_path])
        print(f"  {flight}: {len(basenames)} frames, {len(g)} boxes -> {os.path.basename(csv_path)}")

    stage_list = os.path.join(second_pass_dir, "images_to_stage.txt")
    with open(stage_list, "w") as fh:
        fh.write("\n".join(stage_paths) + "\n")

    print(f"\nWrote {len(written)} package files to {input_dir}")
    print(f"First-pass verdicts (private): {verdict_path}")
    print(f"Images to stage ({len(stage_paths)}): {stage_list}")
    if missing:
        miss_list = os.path.join(second_pass_dir, "images_missing.txt")
        with open(miss_list, "w") as fh:
            fh.write("\n".join(missing) + "\n")
        print(f"Missing images ({len(missing)}): {miss_list}")


if __name__ == "__main__":
    main()
