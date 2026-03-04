#!/usr/bin/env python3
"""
Collect images that have at least one detection above min_score from all
.prediction_cache runs, and report (dry run) or copy them to a screened_images dir.

Usage:
  uv run python collect_screened_images.py [ROOT_DIR]
  ROOT_DIR: root to search for .prediction_cache dirs (default: BOEM imagery root)

Dry run (default): report total number of images and total size in GB.
Copying to /blue/ewhite/b.weinstein/BOEM/screened_images is commented out.
"""

import os
import sys
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = "/blue/ewhite/b.weinstein/BOEM/imagery"
SCREENED_ROOT = "/blue/ewhite/b.weinstein/BOEM/screened_images"
MIN_SCORE = 0.5
CACHE_DIRNAME = ".prediction_cache"
PREDICTIONS_CSV = "pool_predictions.csv"


def find_cache_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Return all .prediction_cache directories under root, up to max_depth levels."""
    caches = []

    def scan(at: Path, depth: int) -> None:
        if depth <= 0:
            return
        try:
            for entry in at.iterdir():
                if entry.is_dir():
                    cache = entry / CACHE_DIRNAME
                    if cache.is_dir():
                        caches.append(cache)
                    else:
                        scan(entry, depth - 1)
        except OSError:
            pass

    scan(root, max_depth)
    return sorted(caches)


def resolve_image_path(path_str: str, flight_dir: Path) -> Path | None:
    """Resolve image_path from CSV to absolute Path; return None if missing."""
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    # Relative to flight (image_dir)
    for candidate in (flight_dir / p, flight_dir / p.name):
        if candidate.exists():
            return candidate
    return None


def collect_screened_paths(root: Path, min_score: float = MIN_SCORE) -> list[tuple[Path, Path]]:
    """
    Find all cache dirs under root; for each, load pool_predictions.csv and collect
    unique image paths with at least one detection score > min_score.
    Returns list of (absolute_image_path, flight_dir) for each image.
    """
    cache_dirs = find_cache_dirs(root)
    if not cache_dirs:
        print(f"No {CACHE_DIRNAME} dirs found under {root}")
        return []

    results = []
    for cache_dir in cache_dirs:
        flight_dir = cache_dir.parent
        csv_path = cache_dir / PREDICTIONS_CSV
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "score" not in df.columns or "image_path" not in df.columns:
            continue
        above = df[df["score"] > min_score]
        for path_str in above["image_path"].astype(str).unique():
            resolved = resolve_image_path(path_str.strip(), flight_dir)
            if resolved is not None:
                results.append((resolved, flight_dir))
    return results


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    if not root.is_dir():
        print(f"Root is not a directory: {root}")
        sys.exit(1)

    print(f"Searching for {CACHE_DIRNAME} under {root}")
    items = collect_screened_paths(root)
    # De-duplicate by resolved path (same file might appear in multiple caches)
    seen = set()
    unique = []
    for path, flight_dir in items:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append((path, flight_dir))

    n = len(unique)
    total_bytes = sum(p.stat().st_size for p, _ in unique)
    size_gb = total_bytes / (1024**3)
    print(f"Images with at least one detection > {MIN_SCORE}: {n}")
    print(f"Total size: {size_gb:.2f} GB")

    # Dry run only; copying commented out.
    for img_path, flight_dir in unique:
        dest_dir = Path(SCREENED_ROOT) / flight_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / img_path.name
        if not dest.exists() or dest.stat().st_mtime < img_path.stat().st_mtime:
            shutil.copy2(img_path, dest)
    print(f"Copied {n} images to {SCREENED_ROOT}")


if __name__ == "__main__":
    main()
