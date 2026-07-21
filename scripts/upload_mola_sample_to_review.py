"""Upload a flight-stratified sample of a suspect class from the NEAQ caches to review.

Motivation: the full-run summary shows ~2,771 "Mola mola" (ocean sunfish) predictions
at very high confidence -- implausible for such a rare animal, i.e. a systematic false
positive. To let annotators characterize and correct it without reviewing all 1,350
frames, this uploads a sample **stratified by source flight**: up to --per-flight frames
randomly drawn from each flight that predicted the class, so every flight/date/camera is
represented.

Reuses the per-flight .prediction_cache/pool_predictions.csv -- runs NO models. Only the
target-class boxes are attached as preannotations, so annotators see exactly the boxes to
fix (mark FalsePositive / reclassify). Uploads to "Bureau of Ocean Energy Management - Review".

Usage:
    uv run python scripts/upload_mola_sample_to_review.py --dry-run
    uv run python scripts/upload_mola_sample_to_review.py
    uv run python scripts/upload_mola_sample_to_review.py --label "Mola mola" --per-flight 5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import label_studio as ls_mod
from src.label_studio import get_api_key

NEAQ_BASE = "/blue/ewhite/b.weinstein/BOEM/neaq"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label", default="Mola mola", help='Class to sample (default: "Mola mola").')
    p.add_argument("--per-flight", type=int, default=5, help="Max frames sampled per flight (default: 5).")
    p.add_argument("--neaq-base", default=NEAQ_BASE, help=f"NEAQ base dir (default: {NEAQ_BASE}).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling (default: 42).")
    p.add_argument("--dry-run", action="store_true", help="Report what would upload; no transfer.")
    return p.parse_args()


def _flight_meta(cache_path: str):
    img_dir = os.path.dirname(os.path.dirname(cache_path))
    date = os.path.basename(os.path.dirname(img_dir)).replace(".", "")
    cam = "belly" if "elly" in os.path.basename(img_dir) else "whale"
    return f"neaq_{date}_{cam}", img_dir


def _build_preannotations(predictions: pd.DataFrame, comet_id_fallback: str) -> dict[str, pd.DataFrame]:
    df = predictions.copy()
    df["image_path"] = df["image_path"].map(lambda p: os.path.basename(str(p)))
    if "comet_id" not in df.columns:
        df["comet_id"] = comet_id_fallback
    if "label" not in df.columns:
        df["label"] = "Object"
    if "score" not in df.columns:
        df["score"] = 1.0
    return {b: g.reset_index(drop=True) for b, g in df.groupby("image_path")}


def main() -> None:
    args = _parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")

    ls_cfg = cfg.annotation.label_studio
    review_project_name = ls_cfg.instances.review.project_name

    caches = sorted(glob.glob(os.path.join(args.neaq_base, "*", "*", ".prediction_cache", "pool_predictions.csv")))
    if not caches:
        print(f"No caches under {args.neaq_base}")
        return

    sftp_client = None if args.dry_run else ls_mod.create_sftp_client(
        user=cfg.server.user, host=cfg.server.host, key_filename=cfg.server.key_filename,
    )

    total_frames = 0
    total_boxes = 0
    n_flights = 0
    for cache_path in caches:
        flight_name, img_dir = _flight_meta(cache_path)
        try:
            preds = pd.read_csv(cache_path)
        except Exception as exc:
            print(f"[{flight_name}] failed to read cache: {exc}")
            continue
        if preds.empty or "cropmodel_label" not in preds.columns:
            continue

        preds["image_path"] = preds["image_path"].map(lambda p: os.path.basename(str(p)))
        target = preds[preds["cropmodel_label"] == args.label].copy()
        if target.empty:
            continue

        # Frames that exist on disk, then stratified random sample within this flight
        frames_on_disk = [b for b in target["image_path"].unique() if os.path.isfile(os.path.join(img_dir, b))]
        if not frames_on_disk:
            print(f"[{flight_name}] {target['image_path'].nunique()} '{args.label}' frames, none on disk -- skipping")
            continue
        s = pd.Series(sorted(frames_on_disk))
        chosen = set(s.sample(n=min(args.per_flight, len(s)), random_state=args.seed))

        frame_preds = target[target["image_path"].isin(chosen)]
        preannotations = _build_preannotations(frame_preds, comet_id_fallback=flight_name)
        image_paths = sorted(os.path.join(img_dir, b) for b in chosen)

        n_flights += 1
        total_frames += len(image_paths)
        total_boxes += len(frame_preds)
        print(f"[{flight_name}] '{args.label}' frames_avail={len(frames_on_disk)} "
              f"sampled={len(image_paths)} boxes={len(frame_preds)}"
              + (" (DRY-RUN)" if args.dry_run else ""))

        if args.dry_run:
            continue

        ls_mod.upload_to_label_studio(
            images=image_paths,
            sftp_client=sftp_client,
            url=ls_cfg.url,
            project_name=review_project_name,
            images_to_annotate_dir=img_dir,
            folder_name=ls_cfg.folder_name,
            preannotations=preannotations,
        )
        print(f"[{flight_name}] uploaded {len(image_paths)} frames to '{review_project_name}'")

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone. {verb} {total_frames} '{args.label}' frames ({total_boxes} boxes) "
          f"stratified across {n_flights} flights (<= {args.per_flight}/flight) to '{review_project_name}'.")


if __name__ == "__main__":
    main()
