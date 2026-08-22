"""Upload cached predictions for unannotated images to the Label Studio review project.

Walks /blue/ewhite/b.weinstein/BOEM/imagery, finds each flight that has a cached
predictions CSV (.full_flight_predictions.csv preferred, else
.prediction_cache/pool_predictions.csv), filters out images already annotated in
train/validation/review CSVs (and USGS train/test crops), and uploads the
remaining images plus preannotations to the
"Bureau of Ocean Energy Management - Review" Label Studio project.

Every box is gated on its own score at --min-score (default predict.min_score), and an
image with nothing left above the gate is not uploaded. This matters because the caches
are built by keeping whole IMAGES: a frame with one good detection carries its
sub-threshold neighbours too, and uploading those puts foam in front of an annotator
under cover of a real box.

Does NOT run any models -- if a flight has no cache, it is skipped. A cache written by an
older checkpoint is therefore uploaded as-is, so pass --flights to restrict the run to
flights whose .prediction_cache provenance files match the current config.

Usage:
    uv run python scripts/upload_cached_predictions_to_review.py
    uv run python scripts/upload_cached_predictions_to_review.py --flights JPG_20241219_120500 JPG_20241220_145900
    uv run python scripts/upload_cached_predictions_to_review.py --dry-run
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

from src import label_studio as ls_mod
from src.annotators import gather_data as annot_gather_data
from src.label_studio import get_api_key
from src.usgs_annotations import load_usgs_annotated_image_paths

IMAGERY_BASE = "/blue/ewhite/b.weinstein/BOEM/imagery"
IMAGE_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF", ".tiff", ".TIFF")

# Gull genera from transformed_taxonomy.json (Aves > Charadriiformes > Laridae).
# Detections classified as one of these species are dropped from preannotations.
GULL_GENERA = ("Larus", "Leucophaeus", "Rissa", "Chroicocephalus", "Hydrocoloeus")


def _is_gull(label) -> bool:
    if not isinstance(label, str):
        return False
    first = label.split()[0] if label else ""
    return first in GULL_GENERA


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--imagery-base",
        default=IMAGERY_BASE,
        help=f"Base imagery directory (default: {IMAGERY_BASE})",
    )
    p.add_argument(
        "--flights",
        nargs="+",
        default=None,
        help="Optional subset of flight directory names to process (default: all flights with a cache).",
    )
    p.add_argument(
        "--exclude-flights",
        nargs="+",
        default=["JPG_20241220_104800"],
        help="Flight directory names to skip entirely.",
    )
    p.add_argument(
        "--include-gulls",
        action="store_true",
        help="Keep gull detections in the upload (default: drop gull detections, drop images with only gulls).",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop every box scoring below this, per box (default: predict.min_score from the config). "
             "An image whose boxes all fall below it is not uploaded at all. Pass 0 to disable the gate.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be uploaded for each flight without contacting Label Studio.",
    )
    return p.parse_args()


def _find_cache(flight_dir: str) -> str | None:
    full = os.path.join(flight_dir, ".full_flight_predictions.csv")
    if os.path.isfile(full):
        return full
    pool = os.path.join(flight_dir, ".prediction_cache", "pool_predictions.csv")
    if os.path.isfile(pool):
        return pool
    return None


def _annotated_basenames(flight_dir: str, cfg) -> set[str]:
    """Collect basenames of images already annotated in train/validation/review and USGS crops."""
    ls_instances = cfg.annotation.label_studio.instances
    flight_name = os.path.basename(flight_dir)
    basenames: set[str] = set()
    for instance in ("train", "validation", "review"):
        ann_dir = os.path.join(ls_instances[instance].csv_dir, flight_name)
        df = annot_gather_data(annotation_dir=ann_dir, image_dir=flight_dir)
        if df is not None and not df.empty and "image_path" in df.columns:
            basenames.update(os.path.basename(str(p)) for p in df["image_path"].unique())

    usgs_train, usgs_test = load_usgs_annotated_image_paths(flight_dir)
    basenames.update(os.path.basename(p) for p in usgs_train)
    basenames.update(os.path.basename(p) for p in usgs_test)
    return basenames


def _build_preannotations(predictions: pd.DataFrame, comet_id_fallback: str) -> dict[str, pd.DataFrame]:
    """Group predictions by image basename, ensuring required columns exist."""
    df = predictions.copy()
    df["image_path"] = df["image_path"].map(lambda p: os.path.basename(str(p)))
    if "comet_id" not in df.columns:
        df["comet_id"] = comet_id_fallback
    else:
        df["comet_id"] = df["comet_id"].fillna(comet_id_fallback)
    if "label" not in df.columns:
        df["label"] = "Object"
    if "score" not in df.columns:
        df["score"] = 1.0

    preannotations: dict[str, pd.DataFrame] = {}
    for basename, group in df.groupby("image_path"):
        preannotations[basename] = group.reset_index(drop=True)
    return preannotations


def _list_flights(imagery_base: str, only: list[str] | None, exclude: list[str] | None) -> list[str]:
    flights = [
        os.path.join(imagery_base, name)
        for name in sorted(os.listdir(imagery_base))
        if os.path.isdir(os.path.join(imagery_base, name))
    ]
    if only:
        only_set = set(only)
        flights = [f for f in flights if os.path.basename(f) in only_set]
    if exclude:
        exclude_set = set(exclude)
        flights = [f for f in flights if os.path.basename(f) not in exclude_set]
    return flights


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

    min_score = args.min_score if args.min_score is not None else float(cfg.predict.min_score)
    print(f"Box score gate: {min_score} ({'--min-score' if args.min_score is not None else 'predict.min_score'})")

    flights = _list_flights(args.imagery_base, args.flights, args.exclude_flights)
    if not flights:
        print(f"No flight directories found under {args.imagery_base}")
        return

    sftp_client = None if args.dry_run else ls_mod.create_sftp_client(
        user=cfg.server.user,
        host=cfg.server.host,
        key_filename=cfg.server.key_filename,
    )

    total_uploaded = 0
    for flight_dir in flights:
        flight_name = os.path.basename(flight_dir)
        cache_path = _find_cache(flight_dir)
        if cache_path is None:
            print(f"[{flight_name}] no cache, skipping")
            continue

        try:
            predictions = pd.read_csv(cache_path)
        except Exception as exc:
            print(f"[{flight_name}] failed to read {cache_path}: {exc}")
            continue
        if predictions.empty or "image_path" not in predictions.columns:
            print(f"[{flight_name}] cache empty or missing image_path column, skipping")
            continue

        predictions["image_path"] = predictions["image_path"].map(lambda p: os.path.basename(str(p)))
        n_cached_boxes = len(predictions)

        # Gate every box on its own score, BEFORE any image is selected. A cache can hold
        # sub-threshold boxes for an image that also has a good one (the pool is built by
        # keeping whole images, not whole boxes), and uploading those drags foam into the
        # queue attached to a real detection. Dropping them here also drops any image left
        # with nothing above the gate, which is the point: one good box justifies that box,
        # not its neighbours.
        if min_score > 0 and "score" in predictions.columns:
            predictions = predictions[predictions["score"] >= min_score].copy()
        n_below_score = n_cached_boxes - len(predictions)
        if predictions.empty:
            print(f"[{flight_name}] all {n_cached_boxes} cached boxes scored < {min_score} -- skipping")
            continue

        n_predicted_images = predictions["image_path"].nunique()
        annotated = _annotated_basenames(flight_dir, cfg)
        unannotated_predictions = predictions[~predictions["image_path"].isin(annotated)].copy()
        if unannotated_predictions.empty:
            print(
                f"[{flight_name}] {n_predicted_images} predicted images, "
                f"all already annotated -- skipping"
            )
            continue

        n_gull_rows = 0
        if not args.include_gulls and "cropmodel_label" in unannotated_predictions.columns:
            gull_mask = unannotated_predictions["cropmodel_label"].map(_is_gull)
            n_gull_rows = int(gull_mask.sum())
            unannotated_predictions = unannotated_predictions[~gull_mask].copy()
            if unannotated_predictions.empty:
                print(
                    f"[{flight_name}] {n_predicted_images} predicted images, "
                    f"dropped {n_gull_rows} gull detections, nothing left -- skipping"
                )
                continue

        # Only upload images that exist on disk in this flight directory
        candidate_basenames = unannotated_predictions["image_path"].unique()
        existing_on_disk = []
        for b in candidate_basenames:
            full = os.path.join(flight_dir, b)
            if os.path.isfile(full):
                existing_on_disk.append(full)
        if not existing_on_disk:
            print(f"[{flight_name}] no predicted images present on disk -- skipping")
            continue

        existing_basenames = {os.path.basename(p) for p in existing_on_disk}
        unannotated_predictions = unannotated_predictions[
            unannotated_predictions["image_path"].isin(existing_basenames)
        ]

        preannotations = _build_preannotations(
            unannotated_predictions, comet_id_fallback=f"cached_{flight_name}"
        )
        image_paths = sorted(existing_on_disk)

        print(
            f"[{flight_name}] cache={os.path.relpath(cache_path, flight_dir)} "
            f"cached_boxes={n_cached_boxes} dropped_below_{min_score}={n_below_score} "
            f"predicted_imgs={n_predicted_images} "
            f"already_annotated={len(annotated & set(predictions['image_path']))} "
            f"dropped_gull_dets={n_gull_rows} "
            f"to_upload={len(image_paths)} boxes={len(unannotated_predictions)}"
        )

        if args.dry_run:
            total_uploaded += len(image_paths)
            continue

        ls_mod.upload_to_label_studio(
            images=image_paths,
            sftp_client=sftp_client,
            url=ls_cfg.url,
            project_name=review_project_name,
            images_to_annotate_dir=flight_dir,
            folder_name=ls_cfg.folder_name,
            preannotations=preannotations,
        )
        total_uploaded += len(image_paths)
        print(f"[{flight_name}] uploaded {len(image_paths)} images to '{review_project_name}'")

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone. {verb} {total_uploaded} images total across {len(flights)} flight(s).")


if __name__ == "__main__":
    main()
