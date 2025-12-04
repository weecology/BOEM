import sys
import os
import json
from pathlib import Path
import pandas as pd

import pytest

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.sagemaker_gt import write_sagemaker_manifest, read_sagemaker_manifest  # noqa: E401


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