import os
import glob
import json
import datetime as _dt
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import globus_sdk  # type: ignore
except Exception:  # pragma: no cover
    globus_sdk = None  # lazy optional import

# Default destination collection for UMESC-UF Pipeline
GLOBUS_UMESC_UF_PIPELINE_COLLECTION_ID = "e9612e0b-677c-4685-a721-7f4c2b6258d0"

import pandas as pd


def _now_date_string() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


def _today_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d")


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


def _split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _common_s3_prefix_and_basenames(s3_uris: Iterable[str]) -> Tuple[str, List[str]]:
    buckets: List[str] = []
    dirpaths: List[str] = []
    basenames: List[str] = []
    for uri in s3_uris:
        bucket, key = _split_s3_uri(uri)
        buckets.append(bucket)
        dirpath = os.path.dirname(key)
        dirpaths.append(dirpath)
        basenames.append(os.path.basename(key))

    if not buckets:
        raise ValueError("No S3 URIs provided")
    if len(set(buckets)) != 1 or len(set(dirpaths)) != 1:
        raise ValueError(
            "All S3 URIs must share the same bucket and directory to build a single manifest"
        )
    bucket = buckets[0]
    dirpath = dirpaths[0]
    s3_prefix = f"s3://{bucket}" if not dirpath else f"s3://{bucket}/{dirpath}"
    return s3_prefix, basenames


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


def write_daily_roster(
    s3_uris: Iterable[str],
    output_dir: str,
    stamp: Optional[str] = None,
    existing_roster_path: Optional[str] = None,
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    roster_path = os.path.join(output_dir, f"{stamp}_roster.txt")

    state: Dict[str, Dict[str, str]] = {}
    if existing_roster_path and os.path.exists(existing_roster_path):
        with open(existing_roster_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    uri, status, views = parts[0], parts[1], parts[2]
                    state[uri] = {"status": status, "views": views}

    for uri in s3_uris:
        if uri not in state:
            state[uri] = {"status": "open", "views": "0"}

    with open(roster_path, "w", encoding="utf-8") as fh:
        fh.write("s3_uri\tstatus\tviews\n")
        for uri, rec in state.items():
            fh.write(f"{uri}\t{rec['status']}\t{rec['views']}\n")
    return roster_path


def assign_jobs_from_roster(
    roster_path: str,
    output_dir: str,
    num_jobs: int,
    stamp: Optional[str] = None,
) -> Tuple[str, List[str]]:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    jobs_path = os.path.join(output_dir, f"{stamp}_jobs.txt")

    rows: List[Tuple[str, str, int]] = []
    with open(roster_path, "r", encoding="utf-8") as fh:
        header_seen = False
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if not header_seen and line.lower().startswith("s3_uri\t"):
                header_seen = True
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            uri, status, views_s = parts[0], parts[1], parts[2]
            try:
                views = int(views_s)
            except Exception:
                views = 0
            rows.append((uri, status, views))

    open_rows = [r for r in rows if r[1] == "open"]
    open_rows.sort(key=lambda r: r[2])
    selected = open_rows[: max(0, num_jobs)]
    selected_uris = [r[0] for r in selected]

    updated: List[Tuple[str, str, int]] = []
    for uri, status, views in rows:
        if uri in selected_uris:
            updated.append((uri, "in_progress", views + 1))
        else:
            updated.append((uri, status, views))

    with open(roster_path, "w", encoding="utf-8") as fh:
        fh.write("s3_uri\tstatus\tviews\n")
        for uri, status, views in updated:
            fh.write(f"{uri}\t{status}\t{views}\n")

    with open(jobs_path, "w", encoding="utf-8") as fh:
        for uri in selected_uris:
            fh.write(uri + "\n")

    return jobs_path, selected_uris


def write_daily_metadata(
    s3_uris: Iterable[str], output_dir: str, stamp: Optional[str] = None
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    meta_path = os.path.join(output_dir, f"{stamp}_metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as fh:
        for uri in s3_uris:
            _, key = _split_s3_uri(uri)
            fh.write(os.path.basename(key) + "\n")
    return meta_path


def write_daily_annotation_manifest(
    s3_uris: Iterable[str],
    output_dir: str,
    job_name: str,
    stamp: Optional[str] = None,
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{stamp}_annotation_manifest.jsonl")
    s3_prefix, basenames = _common_s3_prefix_and_basenames(s3_uris)
    return write_sagemaker_manifest(
        images=basenames,
        output_manifest=manifest_path,
        job_name=job_name,
        s3_prefix=s3_prefix,
    )


def globus_upload_files(
    local_paths: List[str],
    dest_dir: str,
    dest_collection_id: Optional[str] = None,
    source_collection_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> Optional[str]:
    if globus_sdk is None:
        raise RuntimeError(
            "globus-sdk is not installed. Please add 'globus-sdk' to your dependencies."
        )

    source_collection_id = source_collection_id or os.getenv("GLOBUS_SOURCE_COLLECTION_ID")
    if not source_collection_id:
        raise ValueError("source_collection_id is required (or set GLOBUS_SOURCE_COLLECTION_ID)")

    # Allow env override, then default to UMESC-UF Pipeline if not supplied
    dest_collection_id = (
        dest_collection_id
        or os.getenv("GLOBUS_DEST_COLLECTION_ID")
        or GLOBUS_UMESC_UF_PIPELINE_COLLECTION_ID
    )

    client_id = client_id or os.getenv("GLOBUS_NATIVE_APP_CLIENT_ID")
    if not client_id:
        raise ValueError("client_id is required (or set GLOBUS_NATIVE_APP_CLIENT_ID)")

    client = globus_sdk.NativeAppAuthClient(client_id)
    client.oauth2_start_flow(requested_scopes=globus_sdk.scopes.TransferScopes.all)
    authorize_url = client.oauth2_get_authorize_url()
    print("Please go to this URL and login:")
    print(authorize_url)
    print("Then paste the authorization code here and press Enter.")
    auth_code = input("Authorization Code: ")
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)

    transfer_tokens = token_response.by_resource_server["transfer.api.globus.org"]
    authorizer = globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"]) 
    tc = globus_sdk.TransferClient(authorizer=authorizer)

    tc.endpoint_autoactivate(source_collection_id)
    tc.endpoint_autoactivate(dest_collection_id)

    tdata = globus_sdk.TransferData(
        tc, source_collection_id, dest_collection_id, label=f"upload_{_today_stamp()}"
    )
    for p in local_paths:
        dest_path = os.path.join(dest_dir.rstrip("/"), os.path.basename(p))
        tdata.add_item(p, dest_path)

    submit_result = tc.submit_transfer(tdata)
    return submit_result.get("task_id")


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