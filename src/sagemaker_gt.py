import os
import glob
import datetime as _dt
from typing import Dict, Iterable, List, Optional, Tuple

import globus_sdk

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


def write_sagemaker_csv(
    images: Iterable[str],
    output_csv: str,
    flight_path: str,
    s3_prefix: str,
    instance_type: str = "",
    preannotations: Optional[pd.DataFrame] = None,
    capture_date_col: Optional[str] = None,
    human_annotated: str = "yes",
) -> str:
    """
    Write SageMaker annotations to CSV. Columns: bname_parent, label, left, top, width, height,
    cropmodel_label, score, flight_path, instance_type, human_annotated, creation_date, capture_date.

    preannotations (optional) must contain: image_path, xmin, ymin, xmax, ymax
    and may contain cropmodel_label, score, label, and an optional capture_date_col.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    creation_date = _now_date_string()
    has_cropmodel = preannotations is not None and not preannotations.empty and "cropmodel_label" in preannotations.columns
    has_score = preannotations is not None and not preannotations.empty and "score" in preannotations.columns

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
            bname = os.path.basename(str(row["image_path"]))
            stem = _basename_no_ext(bname)
            ann = {
                "bname_parent": bname,
                "left": float(row["xmin"]),
                "top": float(row["ymin"]),
                "width": float(max(0.0, row["xmax"] - row["xmin"])),
                "height": float(max(0.0, row["ymax"] - row["ymin"])),
                "label": str(row[label_col]) if label_col else "",
                "cropmodel_label": str(row["cropmodel_label"]) if has_cropmodel else "",
                "score": float(row["score"]) if has_score else None,
            }
            entry = pre_map.setdefault(stem, {"annotations": [], "capture_date": ""})
            entry["annotations"].append(ann)
            if capture_date_col and capture_date_col in row.index:
                entry["capture_date"] = str(row[capture_date_col])

    rows: List[Dict] = []
    for img in images:
        bname = os.path.basename(img)
        stem = _basename_no_ext(img)
        entry = pre_map.get(stem, {"annotations": [], "capture_date": ""})
        capture_date = entry.get("capture_date", "")
        for a in entry["annotations"]:
            rows.append({
                "bname_parent": a["bname_parent"],
                "label": a["label"],
                "left": a["left"],
                "top": a["top"],
                "width": a["width"],
                "height": a["height"],
                "cropmodel_label": a["cropmodel_label"],
                "score": a["score"],
                "flight_path": flight_path,
                "instance_type": instance_type,
                "human_annotated": human_annotated,
                "creation_date": creation_date,
                "capture_date": capture_date,
            })
        if not entry["annotations"]:
            rows.append({
                "bname_parent": bname,
                "label": "",
                "left": 0.0,
                "top": 0.0,
                "width": 0.0,
                "height": 0.0,
                "cropmodel_label": "",
                "score": None,
                "flight_path": flight_path,
                "instance_type": instance_type,
                "human_annotated": human_annotated,
                "creation_date": creation_date,
                "capture_date": capture_date,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return output_csv


def write_daily_roster(
    s3_uris: Iterable[str],
    output_dir: str,
    stamp: Optional[str] = None,
    existing_roster_path: Optional[str] = None,
    flight_name: Optional[str] = None,
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    filename = f"{stamp}_roster.txt" if not flight_name else f"{stamp}_{flight_name}_roster.txt"
    roster_path = os.path.join(output_dir, filename)

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
    flight_name: Optional[str] = None,
) -> Tuple[str, List[str]]:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    filename = f"{stamp}_jobs.txt" if not flight_name else f"{stamp}_{flight_name}_jobs.txt"
    jobs_path = os.path.join(output_dir, filename)

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
    s3_uris: Iterable[str], output_dir: str, stamp: Optional[str] = None, flight_name: Optional[str] = None
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    filename = f"{stamp}_metadata.txt" if not flight_name else f"{stamp}_{flight_name}_metadata.txt"
    meta_path = os.path.join(output_dir, filename)
    with open(meta_path, "w", encoding="utf-8") as fh:
        for uri in s3_uris:
            _, key = _split_s3_uri(uri)
            fh.write(os.path.basename(key) + "\n")
    return meta_path


def write_daily_annotation_csv(
    s3_uris: Iterable[str],
    output_dir: str,
    flight_path: str,
    instance_type: str = "",
    stamp: Optional[str] = None,
    preannotations: Optional[pd.DataFrame] = None,
    flight_name: Optional[str] = None,
) -> str:
    stamp = stamp or _today_stamp()
    os.makedirs(output_dir or ".", exist_ok=True)
    filename = f"{stamp}_annotation.csv" if not flight_name else f"{stamp}_{flight_name}_annotation.csv"
    csv_path = os.path.join(output_dir, filename)
    s3_prefix, basenames = _common_s3_prefix_and_basenames(s3_uris)
    return write_sagemaker_csv(
        images=basenames,
        output_csv=csv_path,
        flight_path=flight_path,
        instance_type=instance_type,
        s3_prefix=s3_prefix,
        preannotations=preannotations,
    )


def _get_globus_transfer_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> globus_sdk.TransferClient:
    """
    Get authenticated Globus Transfer client.
    
    If refresh_token is provided, uses refresh token flow (authenticates as user).
    Otherwise, uses client credentials flow (authenticates as application).
    """
    client_id = client_id or os.getenv("GLOBUS_CLIENT_ID")
    if not client_id:
        raise ValueError("client_id is required (set GLOBUS_CLIENT_ID)")

    client_secret = client_secret or os.getenv("GLOBUS_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("client_secret is required (set GLOBUS_CLIENT_SECRET)")

    client = globus_sdk.ConfidentialAppAuthClient(client_id, client_secret)
    
    # Use refresh token if provided (authenticates as user)
    if refresh_token:
        token_response = client.oauth2_refresh_token(
            refresh_token,
            requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all"
        )
    else:
        # Fall back to client credentials (authenticates as application)
        token_response = client.oauth2_client_credentials_tokens(
            requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all"
        )
    
    transfer_tokens = token_response.by_resource_server["transfer.api.globus.org"]
    authorizer = globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    return globus_sdk.TransferClient(authorizer=authorizer)


def get_authenticated_identity(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> Optional[Dict]:
    """
    Get information about the authenticated identity.
    
    If refresh_token is provided, returns user identity information.
    Otherwise, returns info about client credentials (no user identity).
    
    Returns:
        Dictionary with identity information
    """
    client_id = client_id or os.getenv("GLOBUS_CLIENT_ID")
    if not client_id:
        raise ValueError("client_id is required (set GLOBUS_CLIENT_ID)")

    client_secret = client_secret or os.getenv("GLOBUS_CLIENT_SECRET")
    if not client_secret:
        raise ValueError("client_secret is required (set GLOBUS_CLIENT_SECRET)")

    client = globus_sdk.ConfidentialAppAuthClient(client_id, client_secret)
    
    # Get tokens for both Auth and Transfer APIs
    if refresh_token:
        token_response = client.oauth2_refresh_token(
            refresh_token,
            requested_scopes=[
                "urn:globus:auth:scope:transfer.api.globus.org:all",
                "urn:globus:auth:scope:auth.globus.org:view_identities",
            ]
        )
    else:
        token_response = client.oauth2_client_credentials_tokens(
            requested_scopes=[
                "urn:globus:auth:scope:transfer.api.globus.org:all",
                "urn:globus:auth:scope:auth.globus.org:view_identities",
            ]
        )
    
    # Try to get identity from Auth API
    if "auth.globus.org" in token_response.by_resource_server:
        auth_tokens = token_response.by_resource_server["auth.globus.org"]
        auth_client = globus_sdk.AuthClient(
            authorizer=globus_sdk.AccessTokenAuthorizer(auth_tokens["access_token"])
        )
        try:
            # Get userinfo - works with refresh token, fails with client credentials
            userinfo = auth_client.oauth2_userinfo()
            return userinfo.data
        except Exception:
            # Client credentials tokens don't have user identity
            pass
    
    # For client credentials, return info about the client
    return {
        "client_id": client_id,
        "note": "Client credentials flow - no user identity associated with this token",
        "token_type": "client_credentials"
    }


def globus_upload_files(
    local_paths: List[str],
    dest_dir: str,
    dest_collection_id: Optional[str] = None,
    source_collection_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> Optional[str]:
    source_collection_id = source_collection_id or os.getenv("GLOBUS_SOURCE_COLLECTION_ID")
    if not source_collection_id:
        raise ValueError("source_collection_id is required (or set GLOBUS_SOURCE_COLLECTION_ID)")

    # Allow env override, then default to UMESC-UF Pipeline if not supplied
    dest_collection_id = (
        dest_collection_id
        or os.getenv("GLOBUS_DEST_COLLECTION_ID")
        or GLOBUS_UMESC_UF_PIPELINE_COLLECTION_ID
    )

    tc = _get_globus_transfer_client(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
    )

    # Note: endpoint_autoactivate removed in globus-sdk 4.x as modern endpoints don't require activation
    # For Globus Connect Server v5 endpoints, activation is automatic

    tdata = globus_sdk.TransferData(
        source_endpoint=source_collection_id,
        destination_endpoint=dest_collection_id,
        label=f"upload_{_today_stamp()}"
    )
    for p in local_paths:
        dest_path = os.path.join(dest_dir.rstrip("/"), os.path.basename(p))
        tdata.add_item(p, dest_path)

    try:
        submit_result = tc.submit_transfer(tdata)
        return submit_result.get("task_id")
    except globus_sdk.TransferAPIError as e:
        # Provide helpful error message with identity info for 403 errors
        if e.http_status == 403:
            identity_info = get_authenticated_identity(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
            )
            error_msg = (
                f"Globus transfer failed with 403 Forbidden.\n"
                f"Source endpoint: {source_collection_id}\n"
                f"Destination endpoint: {dest_collection_id}\n"
            )
            if identity_info.get("sub"):
                error_msg += (
                    f"Authenticated as user: {identity_info.get('preferred_username', 'Unknown')}\n"
                    f"User ID: {identity_info.get('sub', 'Unknown')}\n"
                )
            else:
                error_msg += (
                    f"Authentication: {identity_info.get('note', 'Unknown')}\n"
                    f"Client ID: {identity_info.get('client_id', 'Unknown')}\n"
                )
            error_msg += f"Original error: {e.message}"
            raise globus_sdk.TransferAPIError(e.http_status, error_msg, e.code) from e
        raise


# Label Studio-style columns so gather_data sees one format from both tools
LABELSTUDIO_COLUMNS = ["xmin", "ymin", "xmax", "ymax", "label", "image_path", "annotator", "flight_name"]


def normalize_sagemaker_output_to_annotation_dirs(
    sagemaker_output_dir: str,
    sagemaker_input_dir: str,
    annotations_base_dir: str,
) -> List[str]:
    """
    Read SageMaker output CSVs, join with job input to get flight_name and instance_type,
    normalize to Label Studio column format, and write into annotations_base_dir so
    gather_data picks them up together with Label Studio annotations.

    Output CSVs are written under annotations_base_dir/<instance_type>/<flight_name>/
    with names like <date>_<instance_type>.csv (not *_annotation.csv) so they are
    read as Label Studio CSVs.

    Returns:
        List of paths written.
    """
    written: List[str] = []
    pattern = os.path.join(sagemaker_output_dir, "*_annotation_output.csv")
    for out_path in sorted(glob.glob(pattern)):
        base = os.path.basename(out_path)
        # e.g. 20260131_JPG_20241220_104800_annotation_output.csv -> prefix = 20260131_JPG_20241220_104800
        if not base.endswith("_annotation_output.csv"):
            continue
        prefix = base[: -len("_annotation_output.csv")]
        parts = prefix.split("_", 1)
        if len(parts) != 2:
            continue
        date_stamp, flight_name = parts
        input_path = os.path.join(sagemaker_input_dir, f"{prefix}_annotation.csv")
        if not os.path.isfile(input_path):
            continue
        out_df = pd.read_csv(out_path)
        in_df = pd.read_csv(input_path)
        if "bname_parent" not in out_df.columns:
            continue
        # Input bname_parent has .jpg; output bname_parent is stem only
        in_df = in_df.copy()
        in_df["bname_stem"] = in_df["bname_parent"].astype(str).str.replace(r"\.(jpg|jpeg|JPG|JPEG)$", "", regex=True)
        if "flight_path" in in_df.columns:
            flight_col = "flight_path"
        elif "flight_name" in in_df.columns:
            flight_col = "flight_name"
        else:
            flight_col = None
        if "instance_type" not in in_df.columns or flight_col is None:
            continue
        job_info = in_df[["bname_stem", "instance_type", flight_col]].drop_duplicates("bname_stem")
        merged = out_df.merge(
            job_info,
            left_on="bname_parent",
            right_on="bname_stem",
            how="left",
        )
        # image_path: basename with extension to match Label Studio
        merged["image_path"] = merged["bname_parent"].astype(str) + ".jpg"
        merged["xmin"] = pd.to_numeric(merged["left"], errors="coerce")
        merged["ymin"] = pd.to_numeric(merged["top"], errors="coerce")
        merged["width"] = pd.to_numeric(merged["width"], errors="coerce").fillna(0)
        merged["height"] = pd.to_numeric(merged["height"], errors="coerce").fillna(0)
        merged["xmax"] = merged["xmin"] + merged["width"]
        merged["ymax"] = merged["ymin"] + merged["height"]
        valid = (merged["xmax"] > merged["xmin"]) & (merged["ymax"] > merged["ymin"])
        merged = merged.loc[valid].copy()
        if merged.empty:
            continue
        merged["flight_name"] = merged[flight_col]
        merged["annotator"] = "SageMaker"
        instance_type = str(merged["instance_type"].iloc[0]).strip() or "train"
        out_cols = [c for c in LABELSTUDIO_COLUMNS if c in merged.columns]
        merged = merged[out_cols]
        dest_dir = os.path.join(annotations_base_dir, instance_type, flight_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{date_stamp}_{instance_type}.csv")
        merged.to_csv(dest_path, index=False)
        written.append(dest_path)
    return written


def read_sagemaker_csv(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Read a SageMaker annotation CSV and return a DataFrame with all columns.
    Uses bname_parent (not image_path). Adds xmin, ymin, xmax, ymax, image_path for pipeline compatibility.
    """
    df = pd.read_csv(csv_path)
    required = {"bname_parent", "label", "left", "top", "width", "height"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(columns=["bname_parent", "label", "left", "top", "width", "height", "xmin", "ymin", "xmax", "ymax", "image_path"])
    df = df.copy()
    df["xmin"] = df["left"].astype(float).clip(lower=0)
    df["ymin"] = df["top"].astype(float).clip(lower=0)
    df["xmax"] = (df["left"] + df["width"]).astype(float).clip(lower=0)
    df["ymax"] = (df["top"] + df["height"]).astype(float).clip(lower=0)
    stem = df["bname_parent"].apply(_basename_no_ext)
    df["image_path"] = stem.apply(lambda s: _ensure_image_path(s, image_dir))
    try:
        df["image_path"] = df["image_path"].apply(lambda p: os.path.relpath(p, image_dir))
    except Exception:
        df["image_path"] = df["bname_parent"]
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df


def gather_data(annotation_dir: str, image_dir: str) -> Optional[pd.DataFrame]:
    """
    Aggregate SageMaker annotation CSV files in a directory into a single DataFrame.

    Args:
        annotation_dir: Flight-specific directory containing annotation files
                       (e.g., /path/to/annotations/train/JPG_20241220_145900)
        image_dir: Directory containing images
    """
    files = sorted(glob.glob(os.path.join(annotation_dir, "*_annotation.csv")))

    parts: List[pd.DataFrame] = []
    for fp in files:
        try:
            parts.append(read_sagemaker_csv(fp, image_dir=image_dir))
        except Exception as exc:
            print(f"Warning: failed to parse CSV {fp}: {exc}")

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    return df


def get_refresh_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    redirect_uri: str = "https://auth.globus.org/v2/web/auth-code",
) -> str:
    """
    Interactive helper to obtain a refresh token for user authentication.
    
    This function will:
    1. Generate an authorization URL
    2. Prompt you to visit it and authorize
    3. Exchange the authorization code for tokens
    4. Return the refresh token
    
    Args:
        client_id: Globus client ID (defaults to GLOBUS_CLIENT_ID env var or config)
        client_secret: Globus client secret (defaults to GLOBUS_CLIENT_SECRET env var or config)
        redirect_uri: Redirect URI configured in your Globus app (defaults to Globus web auth)
    
    Returns:
        Refresh token string that can be used for authentication
    """
    # Try to get from parameters, env vars, or config
    client_id = client_id or os.getenv("GLOBUS_CLIENT_ID")
    client_secret = client_secret or os.getenv("GLOBUS_CLIENT_SECRET")
    
    # Try loading from Hydra config if available
    if not client_id or not client_secret:
        try:
            from hydra import initialize, compose
            with initialize(version_base=None, config_path="../boem_conf"):
                cfg = compose(config_name="boem_config", overrides=["annotation=sagemaker"])
                if not client_id:
                    client_id = getattr(cfg.annotation.sagemaker.globus, "native_app_client_id", None)
                if not client_secret:
                    # Check if there's a client_secret in config (might be in a different location)
                    client_secret = getattr(cfg.annotation.sagemaker.globus, "native_app_client_secret", None)
        except Exception:
            pass  # Config loading failed, continue with env vars only
    
    if not client_id:
        raise ValueError(
            "client_id is required. Set GLOBUS_CLIENT_ID environment variable or "
            "annotation.sagemaker.globus.native_app_client_id in config file."
        )

    if not client_secret:
        raise ValueError(
            "client_secret is required. Set GLOBUS_CLIENT_SECRET environment variable or "
            "annotation.sagemaker.globus.native_app_client_secret in config file."
        )

    client = globus_sdk.ConfidentialAppAuthClient(client_id, client_secret)
    
    # Generate authorization URL
    client.oauth2_start_flow(
        redirect_uri,
        requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all"
    )
    auth_url = client.oauth2_get_authorize_url()
    
    print(f"\nPlease visit this URL to authorize:\n{auth_url}\n")
    print("After authorizing, you will be redirected to a page with an authorization code.")
    print("Copy the 'code' parameter from the URL.\n")
    
    auth_code = input("Enter the authorization code: ").strip()
    
    # Exchange code for tokens
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    
    # Extract refresh token
    refresh_token = token_response.by_resource_server["transfer.api.globus.org"]["refresh_token"]
    
    # Get user identity to confirm
    if "auth.globus.org" in token_response.by_resource_server:
        auth_tokens = token_response.by_resource_server["auth.globus.org"]
        auth_client = globus_sdk.AuthClient(
            authorizer=globus_sdk.AccessTokenAuthorizer(auth_tokens["access_token"])
        )
        try:
            userinfo = auth_client.oauth2_userinfo()
            print(f"\nRefresh token obtained successfully!")
            print(f"Authenticated as: {userinfo.data.get('preferred_username', 'Unknown')}")
            print(f"User ID: {userinfo.data.get('sub', 'Unknown')}")
        except Exception:
            pass
    
    print(f"\nAdd this to your config file (boem_conf/annotation/sagemaker.yaml):")
    print(f"  refresh_token: {refresh_token}\n")
    
    return refresh_token


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "get_refresh_token":
        get_refresh_token()
    elif len(sys.argv) > 1 and sys.argv[1] == "normalize_outputs":
        # Default paths: annotations/sagemaker/{input,output} -> annotations/{train,validation,review}
        try:
            from hydra import initialize, compose
            config_path = "boem_conf" if os.path.isdir("boem_conf") else "../boem_conf"
            with initialize(version_base=None, config_path=config_path):
                cfg = compose(config_name="boem_config", overrides=["annotation=sagemaker"])
                train_dir = str(cfg.annotation.sagemaker.instances.train.csv_dir)
                annotations_base = os.path.dirname(train_dir)
                sagemaker_root = os.path.join(annotations_base, "sagemaker")
                output_dir = os.path.join(sagemaker_root, "output")
                input_dir = os.path.join(sagemaker_root, "input")
        except Exception:
            annotations_base = os.getenv("BOEM_ANNOTATIONS_DIR", "/blue/ewhite/b.weinstein/BOEM/annotations")
            output_dir = os.path.join(annotations_base, "sagemaker", "output")
            input_dir = os.path.join(annotations_base, "sagemaker", "input")
        paths = normalize_sagemaker_output_to_annotation_dirs(
            sagemaker_output_dir=output_dir,
            sagemaker_input_dir=input_dir,
            annotations_base_dir=annotations_base,
        )
        for p in paths:
            print(p)
        print(f"Wrote {len(paths)} CSV(s) into annotation dirs.")
    else:
        print("Usage: python -m src.sagemaker_gt get_refresh_token | normalize_outputs")