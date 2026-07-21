"""Upload every cached NEAQ frame containing a cetacean to Label Studio review.

Reuses the per-flight prediction caches written by the pipeline
(.prediction_cache/pool_predictions.csv) -- it runs NO models. For each cached
flight it selects frames whose classifier predicted a cetacean at any taxonomic
level (Cetacea/Mysticeti/Odontoceti/family/genus/species), then uploads those
frames plus their cetacean bounding-box preannotations to the
"Bureau of Ocean Energy Management - Review" project so annotators can see the
whales and dolphins -- independent of the seal-targeted active-learning run.

By default only the cetacean boxes are attached as preannotations (keeps each
import payload small, avoiding the nginx 413 that dense all-box batches hit);
pass --all-boxes to attach every detection in the selected frames.

Usage:
    uv run python scripts/upload_cetaceans_to_review.py --dry-run
    uv run python scripts/upload_cetaceans_to_review.py
    uv run python scripts/upload_cetaceans_to_review.py --camera whale
"""
from __future__ import annotations

import argparse
import glob
import os
import re
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

# Matches cetacean taxa at any taxonomic level (same set used by
# scripts/upload_whales_to_review.py). "higher up in the tree trickles down".
CETACEAN_PATTERN = re.compile(
    "Tursiops|Cetacea|Delphinidae|Delphin|Mysticeti|Odontoceti|"
    "Stenella|Megaptera|Balaen|Grampus|Phocoena|Kogia|Ziphius|"
    "Mesoplodon|Physeter|Orcinus|Globicephala|Pseudorca|Feresa|"
    "Peponocephala|Steno|Sotalia|Sousa|Cephalorhynchus|Lissodelphis|"
    "Lagenorhynchus|Lagenodelphis|Inia|Platanista|Pontoporia|Lipotes"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--neaq-base", default=NEAQ_BASE, help=f"NEAQ base dir (default: {NEAQ_BASE})")
    p.add_argument(
        "--camera",
        choices=("whale", "belly", "both"),
        default="both",
        help="Which camera's frames to include (default: both).",
    )
    p.add_argument(
        "--all-boxes",
        action="store_true",
        help="Attach all detections in the selected frames, not just cetacean boxes.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap total frames uploaded (debug).")
    p.add_argument(
        "--exclude-flights",
        nargs="*",
        default=None,
        help="Flight names (e.g. neaq_20220616_whale) to skip -- used to resume after a failure.",
    )
    p.add_argument(
        "--import-chunk-size",
        type=int,
        default=5,
        help="Frames per Label Studio import POST (default: 5, keeps each request under the nginx body limit).",
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would upload; no transfer.")
    return p.parse_args()


def _flight_name(cache_path: str) -> str:
    # .../neaq/2022.08.19/Whale camera edited/.prediction_cache/pool_predictions.csv
    img_dir = os.path.dirname(os.path.dirname(cache_path))
    date = os.path.basename(os.path.dirname(img_dir)).replace(".", "")
    cam = "belly" if "elly" in os.path.basename(img_dir) else "whale"
    return f"neaq_{date}_{cam}", img_dir, cam


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
    for cache_path in caches:
        flight_name, img_dir, cam = _flight_name(cache_path)
        if args.camera != "both" and cam != args.camera:
            continue
        if args.exclude_flights and flight_name in set(args.exclude_flights):
            print(f"[{flight_name}] excluded -- skipping")
            continue
        try:
            preds = pd.read_csv(cache_path)
        except Exception as exc:
            print(f"[{flight_name}] failed to read cache: {exc}")
            continue
        if preds.empty or "cropmodel_label" not in preds.columns:
            continue

        cet_mask = preds["cropmodel_label"].astype(str).str.contains(CETACEAN_PATTERN)
        if not cet_mask.any():
            continue
        cet_frames = set(preds.loc[cet_mask, "image_path"].map(lambda p: os.path.basename(str(p))).unique())

        preds["image_path"] = preds["image_path"].map(lambda p: os.path.basename(str(p)))
        frame_preds = preds[preds["image_path"].isin(cet_frames)].copy()
        if not args.all_boxes:
            frame_preds = frame_preds[frame_preds["cropmodel_label"].astype(str).str.contains(CETACEAN_PATTERN)].copy()

        # Only frames that exist on disk
        on_disk, missing = [], 0
        for b in sorted(cet_frames):
            full = os.path.join(img_dir, b)
            if os.path.isfile(full):
                on_disk.append(full)
            else:
                missing += 1
        if not on_disk:
            print(f"[{flight_name}] {len(cet_frames)} cetacean frames but none on disk -- skipping")
            continue

        if args.limit is not None:
            room = args.limit - total_frames
            if room <= 0:
                break
            on_disk = on_disk[:room]
        keep = {os.path.basename(p) for p in on_disk}
        frame_preds = frame_preds[frame_preds["image_path"].isin(keep)]

        preannotations = _build_preannotations(frame_preds, comet_id_fallback=flight_name)
        n_frames, n_boxes = len(on_disk), len(frame_preds)
        total_frames += n_frames
        total_boxes += n_boxes
        print(f"[{flight_name}] cetacean_frames={len(cet_frames)} on_disk={n_frames} "
              f"missing={missing} preann_boxes={n_boxes}"
              + (" (DRY-RUN)" if args.dry_run else ""))

        if args.dry_run:
            continue

        ls_mod.upload_to_label_studio(
            images=on_disk,
            sftp_client=sftp_client,
            url=ls_cfg.url,
            project_name=review_project_name,
            images_to_annotate_dir=img_dir,
            folder_name=ls_cfg.folder_name,
            preannotations=preannotations,
            import_chunk_size=args.import_chunk_size,
        )
        print(f"[{flight_name}] uploaded {n_frames} frames to '{review_project_name}'")

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone. {verb} {total_frames} cetacean frames ({total_boxes} preannotation boxes) "
          f"to '{review_project_name}'.")


if __name__ == "__main__":
    main()
