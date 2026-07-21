import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

# Detection crops are named "{parent_frame}_{tile_index}.png"; strip the trailing
# tile index to recover the original frame Basename used in the captures CSVs.
_TILE_SUFFIX_RE = re.compile(r"_\d+$")


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
    """Return image-stem keyed metadata dicts for one flight.

    Returns an empty dict (rather than raising) when no captures CSV exists for
    the flight — this happens for old-format UBFAI images whose filenames don't
    encode a parseable flight datetime.
    """
    metadata_dir = Path(metadata_dir)
    captures_path = metadata_dir / f"{flight_datetime_key(flight_name)}_captures.csv"
    if not captures_path.exists():
        return {}
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


def _parent_basename(image_path: str, bname_parent) -> str:
    """Recover the original frame Basename for a detection-crop row.

    Prefer the explicit ``bname_parent`` column; otherwise strip the trailing
    ``_{tile_index}`` that write_crops appends to the frame stem.
    """
    if bname_parent is not None and not pd.isna(bname_parent) and str(bname_parent):
        return str(bname_parent)
    return _TILE_SUFFIX_RE.sub("", _image_stem(image_path))


def build_capture_index(
    metadata_dir: str | Path,
    needed_basenames: set[str] | None = None,
) -> dict[str, dict]:
    """Build a ``{Basename: {lat, lon, date}}`` index across every captures CSV.

    Captures CSVs are named by flight-START datetime (``{YYYYMMDD_HHMMSS}_captures.csv``),
    which is NOT derivable from an individual frame's capture timestamp, so we scan
    all of them once and key by the per-frame ``Basename``. Passing ``needed_basenames``
    keeps only the frames we actually need, bounding memory.
    """
    metadata_dir = Path(metadata_dir)
    index: dict[str, dict] = {}
    for captures_path in sorted(metadata_dir.glob("*_captures.csv")):
        header = pd.read_csv(captures_path, nrows=0).columns
        if not {"Basename", "Lat", "Lon"} <= set(header):
            continue
        key = captures_path.name[: -len("_captures.csv")]
        date = flight_date(key)
        if not date:
            continue
        chunk = pd.read_csv(captures_path, usecols=["Basename", "Lat", "Lon"])
        chunk = chunk.drop_duplicates(subset=["Basename"])
        for basename, lat, lon in zip(chunk.Basename, chunk.Lat, chunk.Lon):
            basename = str(basename)
            if needed_basenames is not None and basename not in needed_basenames:
                continue
            if basename not in index:
                index[basename] = {"lat": float(lat), "lon": float(lon), "date": date}
    return index


def build_crop_metadata_rows(
    annotations: pd.DataFrame,
    metadata_dir: str | Path,
    default_flight_name: str,
) -> pd.DataFrame:
    """Build DeepForest CropModel metadata rows for crops written by classification.write_crops.

    Matches each detection crop to its original frame's lat/lon/date via the frame
    ``Basename`` (from the ``bname_parent`` column), looked up in a global index of
    all captures CSVs. ``default_flight_name`` is retained for signature compatibility
    and is unused now that matching is Basename-based.
    """
    image_paths = [getattr(r, "image_path") for r in annotations.itertuples(index=False)]
    parents = [
        _parent_basename(getattr(r, "image_path"), getattr(r, "bname_parent", None))
        for r in annotations.itertuples(index=False)
    ]
    index = build_capture_index(metadata_dir, needed_basenames=set(parents))

    rows = []
    for crop_index, (image_path, parent) in enumerate(zip(image_paths, parents)):
        metadata = index.get(parent)
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
