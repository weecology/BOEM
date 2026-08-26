"""Upload genus/family-rank Dolphinidae crops to a dedicated species-review project.

Annotators frequently box a dolphin but stop at "Delphinidae" (or "Dolphin", "Dolphin sp")
rather than picking a species -- src/classification.py's map_dolphin_family_labels() collapses
those onto the indeterminate "Delphinidae sp" class so the flat classifier's two-word filter
doesn't silently drop them. That indeterminate class is a stopgap, not a design goal: nearly
every one of these boxes is almost certainly a bottlenose dolphin (Tursiops truncatus), the
species classes just haven't caught up to what annotators can now tell apart.

This script re-uploads every crop still labeled at genus/family rank to a fresh Label Studio
project ("dolphin_review", see boem_conf/annotation/label_studio.yaml) with the species field
pre-filled -- annotators only need to correct the rare non-bottlenose case, not re-box from
scratch. It is meant to be re-run as more Dolphinidae-rank crops accumulate; already-reviewed
crops (found in the train/validation/review/dolphin_review annotation trees) are skipped
automatically.

PRE-LABEL. As of the a3dc30a0 checkpoint wired into boem_conf/classification_model/finetune.yaml
(2026-08-26), the flat classifier has no white-sided dolphin (Lagenorhynchus) class -- only
Tursiops truncatus, Delphinus delphis, and Stenella frontalis. Every crop is therefore
pre-labeled "Tursiops truncatus"; the Label Studio taxonomy (transformed_taxonomy.json) already
lists "Lagenorhynchus acutus" (Atlantic white-sided) as a selectable correction, so annotators
can flip the minority that are actually white-sided without the tool blocking them. Revisit this
default if a white-sided class ever gets trained in.

Usage:
    uv run python scripts/upload_dolphinidae_to_review.py --dry-run
    uv run python scripts/upload_dolphinidae_to_review.py
    uv run python scripts/upload_dolphinidae_to_review.py --limit 50 --dry-run
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
from src.classification import DOLPHIN_FAMILY_LABELS
from src.label_studio import get_api_key

UBFAI_CROPS_DIR = "/blue/ewhite/b.weinstein/BOEM/training/crops"
DOLPHIN_PRELABEL = "Tursiops truncatus"
EXCLUDE_CSV = frozenset({"train.csv", "test.csv", "zero_shot.csv"})
EXCLUDE_PREFIX = "train_max_empty_"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--crops-dir", default=UBFAI_CROPS_DIR, help=f"UBFAI crops directory (default: {UBFAI_CROPS_DIR})"
    )
    p.add_argument(
        "--include-annotated",
        action="store_true",
        help="Include crops already present in the train/validation/review/dolphin_review annotation trees.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap total crops uploaded (debug).")
    p.add_argument(
        "--import-chunk-size",
        type=int,
        default=5,
        help="Crops per Label Studio import POST (default: 5, keeps each request under the nginx body limit).",
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would upload; no transfer.")
    return p.parse_args()


def _load_dolphinidae_pool(crops_dir: str) -> pd.DataFrame:
    all_csvs = list(Path(crops_dir).glob("*.csv"))
    per_image_csvs = [
        str(x) for x in all_csvs
        if x.name not in EXCLUDE_CSV and not x.name.startswith(EXCLUDE_PREFIX)
    ]
    if not per_image_csvs:
        raise ValueError(f"No per-image CSVs found in {crops_dir}")

    frames = []
    for csv_path in per_image_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: failed to parse {csv_path}: {exc}")
            continue
        if "label" not in df.columns:
            continue
        hit = df["label"].astype(str).str.strip().isin(DOLPHIN_FAMILY_LABELS)
        if hit.any():
            frames.append(df[hit])
    if not frames:
        raise ValueError(f"No Dolphinidae-rank rows found under {crops_dir}")

    df = pd.concat(frames, ignore_index=True)
    dup_subset = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    df = df.drop_duplicates(subset=[c for c in dup_subset if c in df.columns])
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df


def _existing_basenames(csv_bases: list[str]) -> set[str]:
    basenames: set[str] = set()
    for base in csv_bases:
        for csv_path in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "image_path" in df.columns:
                basenames.update(os.path.basename(str(p)) for p in df["image_path"].dropna().unique())
    return basenames


def main() -> None:
    args = _parse_args()
    crops_dir = args.crops_dir
    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"UBFAI crops dir not found: {crops_dir}")

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    ls_cfg = cfg.annotation.label_studio
    project_name = ls_cfg.instances.dolphin_review.project_name

    pool = _load_dolphinidae_pool(crops_dir)
    pool["image_basename"] = pool["image_path"].map(lambda p: os.path.basename(str(p)))

    if not args.include_annotated:
        csv_bases = [
            ls_cfg.instances.train.csv_dir,
            ls_cfg.instances.validation.csv_dir,
            ls_cfg.instances.review.csv_dir,
            ls_cfg.instances.dolphin_review.csv_dir,
        ]
        existing = _existing_basenames(csv_bases)
        before = pool["image_basename"].nunique()
        pool = pool[~pool["image_basename"].isin(existing)].copy()
        skipped = before - pool["image_basename"].nunique()
        if skipped:
            print(f"Skipping {skipped} crops already present in prior annotation trees "
                  "(use --include-annotated to force re-upload).")
    if pool.empty:
        print("Nothing left to upload.")
        return

    basenames = sorted(pool["image_basename"].unique())
    if args.limit is not None:
        basenames = basenames[: args.limit]
    keep = set(basenames)
    pool = pool[pool["image_basename"].isin(keep)]

    on_disk, missing = [], []
    for b in basenames:
        full = os.path.join(crops_dir, b)
        (on_disk if os.path.isfile(full) else missing).append(full if os.path.isfile(full) else b)
    if missing:
        print(f"Warning: {len(missing)} crop images missing on disk, skipping (e.g. {missing[:3]}).")
    if not on_disk:
        print("No crop images found on disk -- nothing to upload.")
        return
    keep_on_disk = {os.path.basename(p) for p in on_disk}
    pool = pool[pool["image_basename"].isin(keep_on_disk)]

    pool["cropmodel_label"] = DOLPHIN_PRELABEL
    pool["cropmodel_score"] = 2.0
    pool["score"] = 2.0
    pool["label"] = "Object"
    pool["comet_id"] = "dolphinidae_species_review"

    preannotations: dict[str, pd.DataFrame] = {}
    for basename, group in pool.groupby("image_basename"):
        group = group.copy()
        group["image_path"] = basename
        preannotations[basename] = group.drop(columns=["image_basename"])

    n_crops, n_boxes = len(on_disk), len(pool)
    print(f"Dolphinidae-rank crops to upload: {n_crops} images, {n_boxes} boxes, "
          f"pre-labeled '{DOLPHIN_PRELABEL}' -> project '{project_name}'"
          + (" (DRY-RUN)" if args.dry_run else ""))
    if args.dry_run:
        return

    sftp_client = ls_mod.create_sftp_client(
        user=cfg.server.user, host=cfg.server.host, key_filename=cfg.server.key_filename,
    )
    ls_mod.upload_to_label_studio(
        images=on_disk,
        sftp_client=sftp_client,
        url=ls_cfg.url,
        project_name=project_name,
        images_to_annotate_dir=crops_dir,
        folder_name=ls_cfg.folder_name,
        preannotations=preannotations,
        import_chunk_size=args.import_chunk_size,
    )
    print(f"Uploaded {n_crops} crops ({n_boxes} boxes) to '{project_name}'.")


if __name__ == "__main__":
    main()
