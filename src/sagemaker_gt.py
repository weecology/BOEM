import os
import glob
import json
import datetime as _dt
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _now_date_string() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


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


def write_sagemaker_manifest(
    images: Iterable[str],
    output_manifest: str,
    job_name: str,
    s3_prefix: str,
    preannotations: Optional[pd.DataFrame] = None,
    capture_date_col: Optional[str] = None,
    human_annotated: str = "yes",
    manifest_type: str = "groundtruth/object-detection",
) -> str:
    """
    Write a SageMaker-style manifest (JSON lines). Each object looks like:
    {
      "source-ref": "s3://bucket/path/to/image.jpg",
      "rootmanifest": {"annotations": [{left, top, width, height, label}, ...]},
      "rootmanifest-metadata": { "capture-date": "...", "job-name": "...", ... }
    }

    preannotations (optional) must contain: image_path, xmin, ymin, xmax, ymax
    and may contain cropmodel_label or label and an optional capture_date_col.
    """
    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    creation_date = _now_date_string()

    # Build map: stem -> {"annotations": [...], "capture_date": ...}
    pre_map: Dict[str, Dict] = {}
    if preannotations is not None and not preannotations.empty:
        required = {"image_path", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(set(preannotations.columns)):
            raise ValueError("preannotations must contain image_path,xmin,ymin,xmax,ymax")
        label_col = (
            "cropmodel_label"
            if "cropmodel_label" in preannotations.columns
            else ("label" if "label" in preannotations.columns else None)
        )
        for _, row in preannotations.iterrows():
            stem = _basename_no_ext(str(row["image_path"]))
            ann = {
                "left": float(row["xmin"]),
                "top": float(row["ymin"]),
                "width": float(max(0.0, row["xmax"] - row["xmin"])),
                "height": float(max(0.0, row["ymax"] - row["ymin"])),
                "label": str(row[label_col]) if label_col else "",
            }
            entry = pre_map.setdefault(stem, {"annotations": [], "capture_date": ""})
            entry["annotations"].append(ann)
            if capture_date_col and capture_date_col in row.index:
                entry["capture_date"] = str(row[capture_date_col])

    with open(output_manifest, "w", encoding="utf-8") as fh:
        for img in images:
            stem = _basename_no_ext(img)
            source_ref = os.path.join(s3_prefix.rstrip("/"), os.path.basename(img))
            entry = pre_map.get(stem, {"annotations": [], "capture_date": ""})
            manifest_obj = {
                "source-ref": source_ref,
                "rootmanifest": {"annotations": entry["annotations"]},
                "rootmanifest-metadata": {
                    "capture-date": entry.get("capture_date", ""),
                    "unique_x": "",
                    "unique_y": "",
                    "job-name": job_name,
                    "human-annotated": human_annotated,
                    "creation-date": creation_date,
                    "type": manifest_type,
                },
            }
            fh.write(json.dumps(manifest_obj) + "\n")
    return output_manifest


def read_sagemaker_manifest(manifest_path: str, image_dir: str) -> pd.DataFrame:
    """
    Read a SageMaker manifest (JSON lines) and convert to DeepForest-format DataFrame:
    columns: image_path (relative to image_dir), xmin, ymin, xmax, ymax, label
    """
    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip invalid JSON lines
                continue
            src = obj.get("source-ref", "")
            basename = _basename_no_ext(src)
            local_path = _ensure_image_path(basename, image_dir)
            # image_path relative to image_dir
            try:
                rel_path = os.path.relpath(local_path, image_dir)
            except Exception:
                rel_path = os.path.basename(local_path)
            anns = obj.get("rootmanifest", {}).get("annotations", []) or []
            for a in anns:
                try:
                    left = float(a.get("left", a.get("xmin", 0.0)))
                    top = float(a.get("top", a.get("ymin", 0.0)))
                    width = float(a.get("width", 0.0))
                    height = float(a.get("height", 0.0))
                except Exception:
                    continue
                rows.append(
                    {
                        "image_path": rel_path,
                        "xmin": left,
                        "ymin": top,
                        "xmax": left + width,
                        "ymax": top + height,
                        "label": str(a.get("label", "Object")),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
    df = pd.DataFrame(rows)
    # clamp negatives and drop invalid boxes
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = df[c].clip(lower=0)
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df


def gather_data(annotation_dir: str, image_dir: str) -> Optional[pd.DataFrame]:
    """
    Aggregate supported annotation files in a directory into a single DataFrame.
    Currently supports only SageMaker manifest JSON-lines (.jsonl/.manifest/.json).
    """
    manifests = sorted(
        glob.glob(os.path.join(annotation_dir, "**", "*.jsonl"), recursive=True)
        + glob.glob(os.path.join(annotation_dir, "**", "*.manifest"), recursive=True)
        + glob.glob(os.path.join(annotation_dir, "**", "*.json"), recursive=True)
    )

    parts: List[pd.DataFrame] = []
    for mf in manifests:
        try:
            parts.append(read_sagemaker_manifest(mf, image_dir=image_dir))
        except Exception as exc:
            print(f"Warning: failed to parse manifest {mf}: {exc}")

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df