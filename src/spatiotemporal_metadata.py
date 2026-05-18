import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def flight_datetime_key(flight_name: str) -> str:
    """Strip JPG_ prefix to get the metadata CSV key for a flight."""
    key = str(flight_name)
    return key[4:] if key.startswith("JPG_") else key


def flight_date(flight_name: str) -> str:
    """Parse YYYY-MM-DD from flight names like JPG_20241220_104800."""
    key = flight_datetime_key(flight_name)
    if len(key) >= 8 and key[:8].isdigit():
        return f"{key[:4]}-{key[4:6]}-{key[6:8]}"
    return ""


def _image_stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(str(image_path)))[0]


def load_flight_metadata(flight_name: str, metadata_dir: str | Path) -> dict[str, dict]:
    """Return image-stem keyed metadata dicts for one flight."""
    metadata_dir = Path(metadata_dir)
    captures_path = metadata_dir / f"{flight_datetime_key(flight_name)}_captures.csv"
    captures = pd.read_csv(captures_path)
    required = {"Basename", "Lat", "Lon"}
    missing = required - set(captures.columns)
    if missing:
        raise ValueError(f"{captures_path} missing required columns: {sorted(missing)}")

    date = flight_date(flight_name)
    rows = captures.drop_duplicates(subset=["Basename"])
    return {
        str(row.Basename): {
            "lat": float(row.Lat),
            "lon": float(row.Lon),
            "date": date,
        }
        for row in rows.itertuples(index=False)
    }


def metadata_for_image(
    image_path: str,
    metadata_dir: str | Path,
    flight_name: str,
    cache: dict[str, dict[str, dict]] | None = None,
) -> dict | None:
    """Look up spatial-temporal metadata for an image path."""
    cache = cache if cache is not None else {}
    if flight_name not in cache:
        cache[flight_name] = load_flight_metadata(flight_name, metadata_dir)
    return cache[flight_name].get(_image_stem(image_path))


def build_crop_metadata_rows(
    annotations: pd.DataFrame,
    metadata_dir: str | Path,
    default_flight_name: str,
) -> pd.DataFrame:
    """Build DeepForest CropModel metadata rows for crops written by classification.write_crops."""
    cache: dict[str, dict[str, dict]] = {}
    rows = []
    for crop_index, row in enumerate(annotations.itertuples(index=False)):
        image_path = getattr(row, "image_path")
        flight_name = getattr(row, "flight_name", default_flight_name)
        if pd.isna(flight_name) or not flight_name:
            flight_name = default_flight_name
        metadata = metadata_for_image(
            image_path=image_path,
            metadata_dir=metadata_dir,
            flight_name=str(flight_name),
            cache=cache,
        )
        if metadata is None or not metadata["date"]:
            continue
        rows.append({
            "filename": f"{_image_stem(image_path)}_{crop_index}.png",
            "lat": metadata["lat"],
            "lon": metadata["lon"],
            "date": metadata["date"],
        })
    return pd.DataFrame(rows, columns=["filename", "lat", "lon", "date"])


def write_crop_metadata_csv(
    annotations: pd.DataFrame,
    metadata_dir: str | Path,
    default_flight_name: str,
    output_csv: str | Path,
) -> str:
    """Write the DeepForest metadata sidecar CSV for classification crops."""
    rows = build_crop_metadata_rows(annotations, metadata_dir, default_flight_name)
    if rows.empty:
        raise ValueError(
            "No crop metadata rows were created. Check report.metadata_dir and "
            f"captures metadata for flight {default_flight_name}."
        )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    return str(output_csv)


def metadata_lookup_for_images(
    image_paths: Iterable[str],
    metadata_dir: str | Path,
    default_flight_name: str,
) -> dict[str, dict]:
    """Return basename/stem keyed metadata for prediction images."""
    cache: dict[str, dict[str, dict]] = {}
    lookup = {}
    for image_path in image_paths:
        metadata = metadata_for_image(
            image_path=image_path,
            metadata_dir=metadata_dir,
            flight_name=default_flight_name,
            cache=cache,
        )
        if metadata is None:
            continue
        lookup[os.path.basename(str(image_path))] = metadata
        lookup[_image_stem(str(image_path))] = metadata
    return lookup
