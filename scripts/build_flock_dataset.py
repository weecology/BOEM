"""
Build a BOEM flock-detector dataset: images with many individuals (often same species).

Selects images ranked by individual count and same-species concentration, creates
a train/test split by image, copies images and annotations to an output directory,
and optionally zips the result for sharing.
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.annotators import gather_data

# Generic labels we exclude when counting "species" for flock tendency
EXCLUDE_LABELS = {
    "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial",
}


def _is_species_label(label: str) -> bool:
    if pd.isna(label) or str(label).strip() == "":
        return False
    s = str(label).strip()
    if s in EXCLUDE_LABELS:
        return False
    return True


def discover_flights(image_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (flight_dir, flight_name).
    If image_dir contains JPG files directly, single flight; else each subdir is a flight.
    """
    image_dir = os.path.abspath(image_dir)
    jpg_here = (
        list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.JPG"))
    )
    if jpg_here:
        return [(image_dir, os.path.basename(image_dir))]
    flights = []
    for sub in sorted(Path(image_dir).iterdir()):
        if not sub.is_dir():
            continue
        jpg_in_sub = list(sub.glob("*.jpg")) + list(sub.glob("*.JPG"))
        if jpg_in_sub:
            flights.append((str(sub), sub.name))
    return flights


def load_all_annotations(
    annotation_base: str,
    flights: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Gather train + validation + review from annotation_base for each flight."""
    parts = []
    for flight_dir, flight_name in flights:
        for subset in ("train", "validation", "review"):
            ann_dir = os.path.join(annotation_base, subset, flight_name)
            df = gather_data(annotation_dir=ann_dir, image_dir=flight_dir)
            if df is not None and not df.empty:
                df = df.copy()
                df["flight_dir"] = flight_dir
                df["flight_name"] = flight_name
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["xmax"] > out["xmin"]) & (out["ymax"] > out["ymin"])]
    return out


def image_stats(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image stats: n_individuals, n_species, dominant_label, dominant_fraction.
    """
    rows = []
    for (img, flight_dir), grp in annotations.groupby(
        ["image_path", "flight_dir"], dropna=False
    ):
        n = len(grp)
        valid = grp[grp["label"].apply(_is_species_label)]
        n_species = valid["label"].nunique() if not valid.empty else 0
        if valid.empty:
            dominant_label = grp["label"].mode().iloc[0] if not grp.empty else None
            dominant_frac = 0.0
        else:
            vc = valid["label"].value_counts()
            dominant_label = vc.index[0]
            dominant_frac = vc.iloc[0] / len(valid)
        rows.append({
            "image_path": img,
            "flight_dir": flight_dir,
            "n_individuals": n,
            "n_species": n_species,
            "dominant_label": dominant_label,
            "dominant_fraction": dominant_frac,
        })
    return pd.DataFrame(rows)


def select_and_split(
    stats: pd.DataFrame,
    n_images: int,
    test_frac: float,
    min_individuals: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Filter, rank, take top n_images, split into train/test by image.
    Returns (stats for selected), train_keys, test_keys.
    """
    stats = stats[stats["n_individuals"] >= min_individuals].copy()
    if stats.empty:
        return stats, [], []
    stats = stats.sort_values(
        by=["n_individuals", "dominant_fraction"],
        ascending=[False, False],
    ).reset_index(drop=True)
    stats = stats.head(n_images)
    n_test = max(1, int(len(stats) * test_frac))
    n_train = len(stats) - n_test
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(stats))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    train_rows = stats.iloc[train_idx]
    test_rows = stats.iloc[test_idx]
    train_keys = [
        (row["flight_dir"], row["image_path"]) for _, row in train_rows.iterrows()
    ]
    test_keys = [
        (row["flight_dir"], row["image_path"]) for _, row in test_rows.iterrows()
    ]
    return stats, train_keys, test_keys


def copy_dataset(
    annotations: pd.DataFrame,
    train_keys: List[Tuple[str, str]],
    test_keys: List[Tuple[str, str]],
    output_dir: str,
) -> None:
    """Copy images and write train/test annotation CSVs. image_path in CSVs is basename."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    def copy_one(flight_dir: str, image_path: str) -> str:
        if os.path.isabs(image_path):
            src = image_path
        else:
            src = os.path.join(flight_dir, image_path)
        basename = os.path.basename(image_path)
        if not os.path.exists(src):
            alt = os.path.join(flight_dir, basename)
            src = alt if os.path.exists(alt) else src
        dst = os.path.join(images_dir, basename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        return basename

    train_basenames = set()
    test_basenames = set()
    for flight_dir, image_path in train_keys:
        train_basenames.add(copy_one(flight_dir, image_path))
    for flight_dir, image_path in test_keys:
        test_basenames.add(copy_one(flight_dir, image_path))

    def subset_and_normalize(keys: List[Tuple[str, str]]) -> pd.DataFrame:
        seen = set(keys)
        rows = []
        for _, row in annotations.iterrows():
            key = (row["flight_dir"], row["image_path"])
            if key not in seen:
                continue
            r = row.drop(labels=["flight_dir", "flight_name"], errors="ignore")
            r = r.copy()
            r["image_path"] = os.path.basename(row["image_path"])
            rows.append(r)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        return out.reset_index(drop=True)

    train_ann = subset_and_normalize(train_keys)
    test_ann = subset_and_normalize(test_keys)
    if not train_ann.empty:
        train_ann.to_csv(os.path.join(output_dir, "train_annotations.csv"), index=False)
    if not test_ann.empty:
        test_ann.to_csv(os.path.join(output_dir, "test_annotations.csv"), index=False)
    with open(os.path.join(output_dir, "train_images.txt"), "w") as f:
        for b in sorted(train_basenames):
            f.write(b + "\n")
    with open(os.path.join(output_dir, "test_images.txt"), "w") as f:
        for b in sorted(test_basenames):
            f.write(b + "\n")

    readme = (
        "BOEM flock-detector dataset: images with many individuals (often same species).\n"
        "Train/test split is by image (no image in both). Use for post-hoc flock/smoothing models.\n"
        "train_annotations.csv / test_annotations.csv: image_path is basename; xmin,ymin,xmax,ymax,label.\n"
        "train_images.txt / test_images.txt: list of image basenames in each split.\n"
    )
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(readme)


def create_zip(output_dir: str, zip_path: Optional[str] = None) -> str:
    if zip_path is None:
        zip_path = output_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                path = os.path.join(root, f)
                arc = os.path.relpath(path, os.path.dirname(output_dir))
                zf.write(path, arc)
    return zip_path


def main():
    p = argparse.ArgumentParser(
        description="Build BOEM flock-detector dataset (images with most individuals, train/test split, optional zip)."
    )
    p.add_argument("--image_dir", required=True, help="Path to imagery (one flight dir or parent of flight dirs)")
    p.add_argument("--annotation_base", required=True, help="Annotations dir with train/, validation/, review/")
    p.add_argument("--output_dir", default="./flock_dataset", help="Output directory (default: ./flock_dataset)")
    p.add_argument("--n_images", type=int, default=400, help="Target number of images (default: 400)")
    p.add_argument("--test_frac", type=float, default=0.2, help="Test fraction (default: 0.2)")
    p.add_argument("--min_individuals", type=int, default=2, help="Min individuals per image (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42)")
    p.add_argument("--zip", action="store_true", help="Create zip archive of output_dir")
    args = p.parse_args()

    flights = discover_flights(args.image_dir)
    if not flights:
        raise SystemExit("No flight directories found under %s" % args.image_dir)
    print("Found %d flight(s)" % len(flights))

    annotations = load_all_annotations(args.annotation_base, flights)
    if annotations.empty:
        raise SystemExit("No annotations found. Check annotation_base and flight subdirs.")
    print("Loaded %d annotations from %d images" % (len(annotations), annotations["image_path"].nunique()))

    stats = image_stats(annotations)
    selected, train_keys, test_keys = select_and_split(
        stats,
        n_images=args.n_images,
        test_frac=args.test_frac,
        min_individuals=args.min_individuals,
        seed=args.seed,
    )
    if selected.empty:
        raise SystemExit("No images with >= %d individuals." % args.min_individuals)
    print("Selected %d images (%d train, %d test)" % (len(selected), len(train_keys), len(test_keys)))

    copy_dataset(annotations, train_keys, test_keys, args.output_dir)
    print("Wrote %s/ (images/, train_annotations.csv, test_annotations.csv, train_images.txt, test_images.txt, README.txt)" % args.output_dir)

    if args.zip:
        zip_path = create_zip(args.output_dir)
        print("Created %s" % zip_path)


if __name__ == "__main__":
    main()
