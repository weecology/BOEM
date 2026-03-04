"""
Fetch and apply bulk annotations from the BOEM-webapp on Serenity.

Bulk annotations CSV (image_id, new_label, set, ...) overrides matching rows
in train/validation/review DataFrames. Optionally, bulk-only image_ids are
resolved via the predictions CSV to add new annotation rows.
"""
import io
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src import label_studio as ls_mod


def _sftp_client_kwargs(server_cfg: Any) -> Dict[str, str]:
    """Extract user, host, key_filename for create_sftp_client (ignore path keys)."""
    return {
        "user": str(server_cfg.user),
        "host": str(server_cfg.host),
        "key_filename": str(server_cfg.key_filename),
    }


def fetch_bulk_annotations_csv(server_cfg: Any, remote_path: str) -> Optional[pd.DataFrame]:
    """
    Fetch the bulk annotations CSV from Serenity via SFTP.

    On any failure (connection, file missing, parse error), log clearly and return None.
    """
    try:
        sftp = ls_mod.create_sftp_client(**_sftp_client_kwargs(server_cfg))
        with sftp.open(remote_path, "r") as f:
            content = f.read().decode("utf-8")
        return pd.read_csv(io.StringIO(content))
    except Exception as e:
        host = getattr(server_cfg, "host", "Serenity")
        print(f"Could not fetch bulk annotations from {host}: {e}. Continuing without bulk overrides.")
        return None


def fetch_predictions_csv(server_cfg: Any, remote_path: str) -> Optional[pd.DataFrame]:
    """
    Fetch the predictions CSV (manifest) from Serenity via SFTP.

    On failure, log and return None; overrides still apply, only "add new from bulk" is skipped.
    """
    try:
        sftp = ls_mod.create_sftp_client(**_sftp_client_kwargs(server_cfg))
        with sftp.open(remote_path, "r") as f:
            content = f.read().decode("utf-8")
        return pd.read_csv(io.StringIO(content))
    except Exception as e:
        host = getattr(server_cfg, "host", "Serenity")
        print(f"Could not fetch predictions CSV from {host}: {e}. Skipping add-new-from-bulk.")
        return None


def reduce_bulk_to_latest(bulk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per image_id with the latest timestamp.

    Returns DataFrame with columns image_id, new_label, set (and timestamp).
    """
    if bulk_df.empty or "image_id" not in bulk_df.columns:
        return bulk_df
    bulk_df = bulk_df.copy()
    if "timestamp" in bulk_df.columns:
        bulk_df["timestamp"] = pd.to_datetime(bulk_df["timestamp"], errors="coerce")
        bulk_df = bulk_df.sort_values("timestamp", ascending=False)
    reduced = bulk_df.groupby("image_id", as_index=False).first()
    return reduced


def annotation_row_to_crop_id(image_path: str, label: str, index: int) -> str:
    """Same convention as visualization.write_crops: {parent_stem}_{label}_{index}.png"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return f"{stem}_{label}_{index}.png"


def apply_bulk_overrides(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    review_df: Optional[pd.DataFrame],
    bulk_lookup: pd.DataFrame,
    flight_image_dir: str,
    predictions_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], int, int]:
    """
    Apply bulk annotation overrides and optionally add new rows from bulk-only image_ids.

    bulk_lookup must have columns image_id, new_label, set (set may be missing; default "review").
    Returns (train_df, val_df, review_df, n_overrides, n_added).
    """
    n_overrides = 0
    n_added = 0
    bulk_by_id = bulk_lookup.set_index("image_id") if not bulk_lookup.empty else pd.DataFrame()
    set_col = "set" if "set" in bulk_lookup.columns else None

    def add_crop_id_and_apply(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        df = df.copy()
        df["_crop_id"] = ""
        for (img, lbl), group in df.groupby(["image_path", "label"]):
            for i, idx in enumerate(group.index):
                df.loc[idx, "_crop_id"] = annotation_row_to_crop_id(str(img), str(lbl), i)
        return df

    def do_overrides(df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], set]:
        nonlocal n_overrides
        matched_ids = set()
        if df is None or df.empty:
            return df, matched_ids
        df = add_crop_id_and_apply(df)
        for idx, row in df.iterrows():
            cid = row["_crop_id"]
            if cid in bulk_by_id.index:
                matched_ids.add(cid)
                rec = bulk_by_id.loc[cid]
                df.loc[idx, "label"] = rec["new_label"]
                n_overrides += 1
        df = df.drop(columns=["_crop_id"], errors="ignore")
        return df, matched_ids

    train_df, matched_train = do_overrides(train_df)
    val_df, matched_val = do_overrides(val_df)
    review_df, matched_review = do_overrides(review_df)
    all_matched = matched_train | matched_val | matched_review

    # Phase 2: add new rows from bulk-only image_ids using predictions manifest
    if predictions_df is not None and not predictions_df.empty and not bulk_lookup.empty:
        flight_name = os.path.basename(flight_image_dir)
        req = {"crop_image_id", "image_path", "xmin", "ymin", "xmax", "ymax"}
        if not req.issubset(predictions_df.columns):
            return train_df, val_df, review_df, n_overrides, n_added
        if "flight_name" in predictions_df.columns:
            pred_flight = predictions_df[predictions_df["flight_name"] == flight_name]
        else:
            pred_flight = predictions_df
        if pred_flight.empty:
            return train_df, val_df, review_df, n_overrides, n_added
        manifest = pred_flight.set_index("crop_image_id")[["image_path", "xmin", "ymin", "xmax", "ymax"]]

        buckets: Dict[str, list] = {"train": [], "validation": [], "review": []}
        for _, rec in bulk_lookup.iterrows():
            image_id = rec["image_id"]
            if image_id in all_matched or image_id not in manifest.index:
                continue
            set_val = (rec.get("set", "review") if set_col else "review")
            if pd.isna(set_val) or set_val not in buckets:
                set_val = "review"
            row = manifest.loc[image_id]
            buckets[set_val].append({
                "image_path": row["image_path"],
                "xmin": float(row["xmin"]),
                "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]),
                "ymax": float(row["ymax"]),
                "label": rec["new_label"],
            })
            all_matched.add(image_id)
            n_added += 1

        for set_name, new_rows in buckets.items():
            if not new_rows:
                continue
            new_df = pd.DataFrame(new_rows)
            if set_name == "train":
                train_df = new_df if (train_df is None or train_df.empty) else pd.concat([train_df, new_df], ignore_index=True)
            elif set_name == "validation":
                val_df = new_df if (val_df is None or val_df.empty) else pd.concat([val_df, new_df], ignore_index=True)
            else:
                review_df = new_df if (review_df is None or review_df.empty) else pd.concat([review_df, new_df], ignore_index=True)

    return train_df, val_df, review_df, n_overrides, n_added
