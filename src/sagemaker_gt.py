import os
import glob
import datetime as _dt
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _now_date_string() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M")


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _ensure_image_path(image_stem: str, image_dir: str) -> str:
    """
    Resolve an image filename from a stem by checking common extensions in the
    provided directory. Falls back to the stem as a path if not found.
    """
    candidates = [
        os.path.join(image_dir, f"{image_stem}.jpg"),
        os.path.join(image_dir, f"{image_stem}.JPG"),
        os.path.join(image_dir, f"{image_stem}.jpeg"),
        os.path.join(image_dir, f"{image_stem}.JPEG"),
        os.path.join(image_dir, f"{image_stem}.png"),
        os.path.join(image_dir, f"{image_stem}.PNG"),
        os.path.join(image_dir, image_stem),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_dir, image_stem)


def write_request_csv(
    images: Iterable[str],
    output_csv: str,
    job_name: str,
    source: str,
    preannotations: Optional[pd.DataFrame] = None,
) -> str:
    """
    Create a CSV for SageMaker Ground Truth team upload.

    The CSV schema follows the team's expected columns based on the provided
    example screenshot:
      - bname_parcel: image basename without extension
      - label: optional seed label (string)
      - conf: optional confidence (float)
      - left, top, width, height: bounding box in parent coordinates (pixels)
      - creation_dat: UTC timestamp (YYYY-mm-ddTHH:MM)
      - source: free-text source (e.g., flightline)
      - job_name: identifier of the job/batch

    If preannotations is provided, one row is written per proposed box per
    image. Otherwise, one empty row per image is written to request annotation.

    Returns the written CSV path.
    """
    rows: List[Dict] = []  # type: ignore[valid-type]
    creation_dat = _now_date_string()

    # Normalize preannotations to expected columns
    if preannotations is not None and not preannotations.empty:
        expected_cols = {"image_path", "xmin", "ymin", "xmax", "ymax"}
        missing = expected_cols - set(preannotations.columns)
        if missing:
            raise ValueError(
                f"preannotations is missing required columns: {sorted(missing)}"
            )

        # Prefer explicit class/score columns if present
        label_col = (
            "cropmodel_label"
            if "cropmodel_label" in preannotations.columns
            else ("label" if "label" in preannotations.columns else None)
        )
        score_col = "score" if "score" in preannotations.columns else None

        for _, row in preannotations.iterrows():
            image_stem = _basename_no_ext(str(row["image_path"]))
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            rows.append(
                {
                    "bname_parcel": image_stem,
                    "label": (str(row[label_col]) if label_col else ""),
                    "conf": (float(row[score_col]) if score_col else ""),
                    "left": xmin,
                    "top": ymin,
                    "width": max(0.0, xmax - xmin),
                    "height": max(0.0, ymax - ymin),
                    "creation_dat": creation_dat,
                    "source": source,
                    "job_name": job_name,
                }
            )
    else:
        # Write a single placeholder row per image (no preannotation boxes)
        for img in images:
            rows.append(
                {
                    "bname_parcel": _basename_no_ext(img),
                    "label": "",
                    "conf": "",
                    "left": "",
                    "top": "",
                    "width": "",
                    "height": "",
                    "creation_dat": creation_dat,
                    "source": source,
                    "job_name": job_name,
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str, str]:
    """
    Robustly detect the required columns from the Ground Truth CSV, which may
    vary slightly by naming.
    Returns tuple: (image_col, label_col, left_col, top_col, width_col, height_col)
    """
    # Image/name
    image_candidates = ["bname_parcel", "image", "image_name", "filename", "name"]
    label_candidates = ["label", "class", "species"]
    left_candidates = ["left", "xmin", "x_min"]
    top_candidates = ["top", "ymin", "y_min"]
    width_candidates = ["width", "w"]
    height_candidates = ["height", "h"]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    image_col = pick(image_candidates)
    label_col = pick(label_candidates) or "label"
    left_col = pick(left_candidates)
    top_col = pick(top_candidates)
    width_col = pick(width_candidates)
    height_col = pick(height_candidates)

    required = {
        "image": image_col,
        "left": left_col,
        "top": top_col,
        "width": width_col,
        "height": height_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required columns in SageMaker CSV: {', '.join(missing)}"
        )

    return image_col, label_col, left_col, top_col, width_col, height_col


def read_results_csv(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Read a SageMaker Ground Truth results CSV and convert to DeepForest format:
      - image_path: relative path to parent image
      - xmin, ymin, xmax, ymax: in pixels
      - label: taxonomy/class string
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])  # noqa: E501

    image_col, label_col, left_col, top_col, width_col, height_col = _detect_columns(df)

    # Coerce numeric columns
    for c in [left_col, top_col, width_col, height_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing any box numeric field
    df = df.dropna(subset=[left_col, top_col, width_col, height_col])

    # Compute DeepForest-friendly columns
    result = pd.DataFrame(
        {
            "image_path": [
                os.path.relpath(_ensure_image_path(str(stem), image_dir), image_dir)
                for stem in df[image_col].astype(str).tolist()
            ],
            "xmin": df[left_col].astype(float),
            "ymin": df[top_col].astype(float),
            "xmax": (df[left_col] + df[width_col]).astype(float),
            "ymax": (df[top_col] + df[height_col]).astype(float),
            "label": df.get(label_col, "Object").astype(str),
        }
    )

    # Sanity: clamp negatives
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        result[c] = result[c].clip(lower=0)

    return result


def gather_data(annotation_dir: str, image_dir: str) -> Optional[pd.DataFrame]:
    """
    Aggregate all CSVs in a directory into a single DeepForest-format DataFrame.
    """
    csvs = sorted(glob.glob(os.path.join(annotation_dir, "**", "*.csv"), recursive=True))
    if not csvs:
        return None

    parts: List[pd.DataFrame] = []
    for csv in csvs:
        try:
            parts.append(read_results_csv(csv, image_dir=image_dir))
        except Exception as exc:  # keep going on partial/invalid files
            print(f"Warning: failed to parse {csv}: {exc}")
    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    # Drop empty boxes
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df


# Notes on tiling and coordinates
# -------------------------------
# DeepForest training functions expect annotation coordinates to be in the
# coordinate system of the ORIGINAL/PARENT image. The preprocessing pipeline
# (see src/data_processing.preprocess_images/process_image) uses
# deepforest.preprocess.split_raster to handle splitting the parent image into
# tiles and internally transforms annotation coordinates into tile coordinates.
#
# Therefore, SageMaker results should be provided here in parent-image pixel
# coordinates. The returned DataFrame from read_results_csv matches
# DeepForest's expected format and can be passed directly to
# data_processing.preprocess_images for tiling and training.


