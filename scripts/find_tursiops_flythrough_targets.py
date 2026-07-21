#!/usr/bin/env python3
"""
List flights and cameras that have Tursiops predictions in the prediction cache,
to help choose a good flythrough target.

Usage (from repo root):
  uv run python scripts/find_tursiops_flythrough_targets.py
  uv run python scripts/find_tursiops_flythrough_targets.py /path/to/imagery/root
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IMAGERY_ROOT = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
CACHE_DIRNAME = ".prediction_cache"
PREDICTIONS_CSV = "pool_predictions.csv"
TARGET_SUBSTR = "Tursiops"
_FNAME_RE = re.compile(r"(C\d+)_L\d+_F(\d+)_T\d{8}_\d{6}_\d{3}", re.IGNORECASE)


@dataclass
class CameraHit:
    flight: Path
    camera: str
    n_images: int
    n_boxes: int


def _iter_flights(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def _camera_from_image_path(path_str: str) -> str | None:
    name = Path(path_str).name
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return m.group(1)


def _flight_camera_hits(flight_dir: Path) -> list[CameraHit]:
    csv_path = flight_dir / CACHE_DIRNAME / PREDICTIONS_CSV
    if not csv_path.is_file():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if "image_path" not in df.columns:
        return []

    label_cols = [c for c in ("cropmodel_label", "hcast_species") if c in df.columns]
    if not label_cols:
        return []

    mask = False
    for col in label_cols:
        mask = mask | df[col].astype(str).str.contains(TARGET_SUBSTR, case=False, na=False)
    hits = df[mask].copy()
    if hits.empty:
        return []

    hits["camera"] = hits["image_path"].astype(str).map(_camera_from_image_path)
    hits = hits[hits["camera"].notna()]
    if hits.empty:
        return []

    grouped = hits.groupby("camera")
    out: list[CameraHit] = []
    for cam, g in grouped:
        n_images = g["image_path"].astype(str).nunique()
        n_boxes = len(g)
        out.append(CameraHit(flight=flight_dir, camera=str(cam), n_images=n_images, n_boxes=n_boxes))
    return out


def find_tursiops_targets(root: Path) -> list[CameraHit]:
    flights = _iter_flights(root)
    results: list[CameraHit] = []
    for flight in flights:
        results.extend(_flight_camera_hits(flight))
    return sorted(results, key=lambda h: (-h.n_images, -h.n_boxes, h.flight.name, h.camera))


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGERY_ROOT
    if not root.is_dir():
        print(f"Imagery root is not a directory: {root}")
        return 1

    hits = find_tursiops_targets(root)
    if not hits:
        print(f"No prediction caches with '{TARGET_SUBSTR}' found under {root}")
        return 0

    print(f"Tursiops candidates under {root} (sorted by images, then boxes):\n")
    header = f"{'#':>3}  {'Flight':<24}  {'Camera':<6}  {'Images':>8}  {'Boxes':>8}"
    print(header)
    print("-" * len(header))
    for idx, h in enumerate(hits):
        print(
            f"{idx:>3}  {h.flight.name:<24}  {h.camera:<6}  {h.n_images:>8}  {h.n_boxes:>8}"
        )

    print(
        "\nChoose an index to see the sbatch command for a flythrough, or press Enter to quit."
    )
    try:
        choice = input("Index: ").strip()
    except EOFError:
        return 0
    if not choice:
        return 0
    try:
        idx = int(choice)
    except ValueError:
        print("Not a valid integer index.")
        return 1
    if idx < 0 or idx >= len(hits):
        print("Index out of range.")
        return 1

    sel = hits[idx]
    flight_dir = sel.flight
    camera = sel.camera
    print("\nSelected:")
    print(f"  Flight: {flight_dir}")
    print(f"  Camera: {camera}")
    print("\nSubmit flythrough job with:")
    print(f"  sbatch submit_flythrough.sh {flight_dir} {camera}")
    print("\nAfter reviewing/cleaning annotations, you can re-run the same command to regenerate the video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

