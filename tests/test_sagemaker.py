import sys
import os
import time
from pathlib import Path
import pandas as pd

import pytest
import globus_sdk
from hydra import initialize, compose

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.sagemaker_gt import (
    write_sagemaker_csv,
    read_sagemaker_csv,
    gather_data,
    globus_upload_files,
    _get_globus_transfer_client,
)  # noqa: E401

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def make_dummy_images(tmp_path, names):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for n in names:
        (img_dir / n).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal binary to exist
    return img_dir


def test_write_csv_no_preannotations(tmp_path):
    images = ["C1.jpg", "C2.jpg"]
    img_dir = make_dummy_images(tmp_path, images)
    out_csv = tmp_path / "annotations_no_pre.csv"
    s3_prefix = "s3://bucket/prefix"

    path = write_sagemaker_csv(
        images=images,
        output_csv=str(out_csv),
        flight_path="test_flight",
        s3_prefix=s3_prefix,
        instance_type="train",
        preannotations=None,
    )
    assert str(out_csv) == path
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert "bname_parent" in df.columns
    assert set(df.columns) >= {"bname_parent", "label", "left", "top", "width", "height", "flight_path", "instance_type", "human_annotated", "creation_date", "capture_date"}
    assert len(df) == 2
    assert set(df["bname_parent"]) == {"C1.jpg", "C2.jpg"}
    assert df["flight_path"].iloc[0] == "test_flight"
    assert df["instance_type"].iloc[0] == "train"
    assert (df["label"].isna() | (df["label"] == "")).all()
    assert (df["left"] == 0.0).all() and (df["top"] == 0.0).all() and (df["width"] == 0.0).all() and (df["height"] == 0.0).all()


def test_write_csv_with_preannotations_and_readback(tmp_path):
    images = ["I1.jpg", "I2.jpg"]
    img_dir = make_dummy_images(tmp_path, images)
    out_csv = tmp_path / "annotations_pre.csv"
    s3_prefix = "s3://bucket/pfx"

    rows = [
        {"image_path": "I1.jpg", "xmin": 10.0, "ymin": 5.0, "xmax": 30.0, "ymax": 25.0, "cropmodel_label": "Anatidae", "score": 0.95, "capture_date": "2024-01-01 00:00:00"},
        {"image_path": "I1.jpg", "xmin": 50.0, "ymin": 40.0, "xmax": 70.0, "ymax": 60.0, "cropmodel_label": "Anatidae", "score": 0.88, "capture_date": "2024-01-01 00:00:00"},
        {"image_path": "I2.jpg", "xmin": 1.0, "ymin": 2.0, "xmax": 11.0, "ymax": 12.0, "cropmodel_label": "Other", "score": 0.75, "capture_date": "2024-02-02 00:00:00"},
    ]
    pre = pd.DataFrame(rows)

    path = write_sagemaker_csv(
        images=images,
        output_csv=str(out_csv),
        flight_path="JPG_20241220_104800",
        s3_prefix=s3_prefix,
        instance_type="validation",
        preannotations=pre,
        capture_date_col="capture_date",
    )
    assert out_csv.exists()

    raw = pd.read_csv(out_csv)
    assert "bname_parent" in raw.columns
    assert set(raw.columns) >= {"bname_parent", "label", "left", "top", "width", "height", "cropmodel_label", "score", "flight_path", "instance_type", "capture_date"}
    assert len(raw) == 3
    assert list(raw["bname_parent"].value_counts()) == [2, 1]
    assert set(raw["label"].unique()) == {"Anatidae", "Other"}
    assert raw["flight_path"].iloc[0] == "JPG_20241220_104800"
    assert raw["instance_type"].iloc[0] == "validation"
    other = raw[raw["label"] == "Other"].iloc[0]
    assert other["left"] == 1.0 and other["top"] == 2.0 and other["width"] == 10.0 and other["height"] == 10.0
    assert other["score"] == 0.75
    assert other["cropmodel_label"] == "Other"

    df = read_sagemaker_csv(str(out_csv), image_dir=str(img_dir))
    assert len(df) == 3
    assert set(df["label"].unique()) == {"Anatidae", "Other"}
    assert "image_path" in df.columns and "bname_parent" in df.columns
    row = df[df["label"] == "Other"].iloc[0]
    assert row["xmin"] == 1.0 and row["ymin"] == 2.0 and row["xmax"] == 11.0 and row["ymax"] == 12.0


def test_gather_data_csv(tmp_path):
    images = ["I1.jpg", "I2.jpg"]
    img_dir = make_dummy_images(tmp_path, images)
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    out_csv = ann_dir / "20250101_annotation.csv"
    write_sagemaker_csv(
        images=images,
        output_csv=str(out_csv),
        flight_path="test_flight",
        s3_prefix="s3://b/p",
        instance_type="train",
        preannotations=pd.DataFrame([
            {"image_path": "I1.jpg", "xmin": 10, "ymin": 5, "xmax": 30, "ymax": 25, "label": "A"},
            {"image_path": "I2.jpg", "xmin": 1, "ymin": 2, "xmax": 11, "ymax": 12, "label": "B"},
        ]),
    )
    df = gather_data(str(ann_dir), str(img_dir))
    assert df is not None
    assert len(df) == 2
    assert "bname_parent" in df.columns and "image_path" in df.columns
    assert set(df["label"]) == {"A", "B"}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test requires local Globus credentials and connection")
def test_globus_upload_and_delete(tmp_path):
    """Test Globus file upload, existence check, and deletion."""
    # Check for required authentication environment variables
    required_auth_vars = [
        "GLOBUS_CLIENT_ID",
        "GLOBUS_CLIENT_SECRET",
    ]
    missing_auth_vars = [var for var in required_auth_vars if not os.getenv(var)]
    if missing_auth_vars:
        pytest.skip(f"Missing required environment variables: {', '.join(missing_auth_vars)}")

    # Load config to get collection IDs from YAML
    with initialize(version_base=None, config_path="../boem_conf"):
        cfg = compose(config_name="boem_config", overrides=["annotation=sagemaker"])
    
    # Get collection IDs and dest_dir from config
    source_collection_id = cfg.annotation.sagemaker.globus.source_collection_id
    dest_collection_id = cfg.annotation.sagemaker.globus.dest_collection_id
    dest_dir = cfg.annotation.sagemaker.globus.dest_dir
    
    if not source_collection_id or not dest_collection_id:
        pytest.skip("Collection IDs not configured in sagemaker.yaml")

    # Create a test file
    test_filename = f"test_globus_{int(time.time())}.txt"
    test_file = tmp_path / test_filename
    test_content = "This is a test file for Globus transfer verification"
    test_file.write_text(test_content)

    # Upload the file
    task_id = globus_upload_files(
        local_paths=[str(test_file)],
        dest_dir=dest_dir,
        dest_collection_id=dest_collection_id,
        source_collection_id=source_collection_id,
    )
    assert task_id is not None, "Upload task should return a task_id"

    # Get transfer client to wait for completion and check file
    tc = _get_globus_transfer_client()
    # Note: endpoint_autoactivate removed in globus-sdk 4.x as modern endpoints don't require activation

    # Wait for transfer to complete (timeout after 5 minutes)
    done = tc.task_wait(task_id, timeout=300, polling_interval=10)
    assert done, f"Transfer task {task_id} did not complete within timeout"

    # Check task status to ensure it succeeded
    task_info = tc.get_task(task_id)
    assert task_info["status"] == "SUCCEEDED", f"Transfer task failed with status: {task_info['status']}"

    # Check if file exists on remote endpoint
    remote_path = os.path.join(dest_dir.rstrip("/"), test_filename)
    file_exists = False
    for entry in tc.operation_ls(dest_collection_id, path=dest_dir):
        if entry["name"] == test_filename and entry["type"] == "file":
            file_exists = True
            break

    assert file_exists, f"File {test_filename} should exist on remote endpoint after upload"

    # Delete the file
    delete_data = globus_sdk.DeleteData(tc, dest_collection_id)
    delete_data.add_item(remote_path)
    delete_task = tc.submit_delete(delete_data)
    delete_task_id = delete_task["task_id"]

    # Wait for delete to complete
    delete_done = tc.task_wait(delete_task_id, timeout=300, polling_interval=10)
    assert delete_done, f"Delete task {delete_task_id} did not complete within timeout"

    # Verify file is deleted
    file_still_exists = False
    for entry in tc.operation_ls(dest_collection_id, path=dest_dir):
        if entry["name"] == test_filename and entry["type"] == "file":
            file_still_exists = True
            break

    assert not file_still_exists, f"File {test_filename} should be deleted from remote endpoint"