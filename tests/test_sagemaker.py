import sys
import os
import json
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
    write_sagemaker_manifest,
    read_sagemaker_manifest,
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


def test_write_manifest_no_preannotations(tmp_path):
    images = ["C1.jpg", "C2.jpg"]
    img_dir = make_dummy_images(tmp_path, images)
    out_manifest = tmp_path / "manifest_no_pre.jsonl"
    s3_prefix = "s3://bucket/prefix"

    # call writer
    path = write_sagemaker_manifest(
        images=images,
        output_manifest=str(out_manifest),
        job_name="test-job",
        s3_prefix=s3_prefix,
        preannotations=None,
    )
    assert str(out_manifest) == path
    assert out_manifest.exists()

    # validate contents
    lines = out_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    objs = [json.loads(l) for l in lines]
    for img, obj in zip(images, objs):
        assert obj["source-ref"].endswith(img)
        assert obj["rootmanifest"]["annotations"] == []
        meta = obj["rootmanifest-metadata"]
        assert meta["job-name"] == "test-job"
        assert meta["type"] == "groundtruth/object-detection"


def test_write_manifest_with_preannotations_and_readback(tmp_path):
    images = ["I1.jpg", "I2.jpg"]
    img_dir = make_dummy_images(tmp_path, images)
    out_manifest = tmp_path / "manifest_pre.jsonl"
    s3_prefix = "s3://bucket/pfx"

    # build preannotations dataframe
    rows = [
        {"image_path": "I1.jpg", "xmin": 10.0, "ymin": 5.0, "xmax": 30.0, "ymax": 25.0, "cropmodel_label": "Anatidae", "capture_date": "2024-01-01 00:00:00"},
        {"image_path": "I1.jpg", "xmin": 50.0, "ymin": 40.0, "xmax": 70.0, "ymax": 60.0, "cropmodel_label": "Anatidae", "capture_date": "2024-01-01 00:00:00"},
        {"image_path": "I2.jpg", "xmin": 1.0, "ymin": 2.0, "xmax": 11.0, "ymax": 12.0, "cropmodel_label": "Other", "capture_date": "2024-02-02 00:00:00"},
    ]
    pre = pd.DataFrame(rows)

    # write manifest
    path = write_sagemaker_manifest(
        images=images,
        output_manifest=str(out_manifest),
        job_name="job-42",
        s3_prefix=s3_prefix,
        preannotations=pre,
        capture_date_col="capture_date",
    )
    assert out_manifest.exists()

    # inspect file
    lines = [json.loads(l) for l in out_manifest.read_text(encoding="utf-8").strip().splitlines()]
    # I1 should have 2 annotations, I2 one
    mapping = {Path(obj["source-ref"]).name: obj for obj in lines}
    assert mapping["I1.jpg"]["rootmanifest"]["annotations"][0]["label"] == "Anatidae"
    assert len(mapping["I1.jpg"]["rootmanifest"]["annotations"]) == 2
    assert mapping["I2.jpg"]["rootmanifest"]["annotations"][0]["label"] == "Other"
    # metadata capture-date populated from preannotations
    assert mapping["I1.jpg"]["rootmanifest-metadata"]["capture-date"] == "2024-01-01 00:00:00"
    assert mapping["I2.jpg"]["rootmanifest-metadata"]["capture-date"] == "2024-02-02 00:00:00"

    # round-trip read back into dataframe (use image_dir for resolution)
    df = read_sagemaker_manifest(str(out_manifest), image_dir=str(img_dir))
    # should contain three rows matching preannotations (order may differ)
    assert len(df) == 3
    assert set(df["label"].unique()) == {"Anatidae", "Other"}
    
    # numeric boxes consistent
    row = df[df["label"] == "Other"].iloc[0]
    assert row["xmin"] == 1.0 and row["ymin"] == 2.0 and row["xmax"] == 11.0 and row["ymax"] == 12.0


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