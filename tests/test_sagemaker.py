import sys
from pathlib import Path

import pandas as pd

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.sagemaker_gt import (  # noqa: E402
    write_sagemaker_csv,
    read_sagemaker_csv,
    gather_data,
)


def make_dummy_images(tmp_path, names):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for n in names:
        (img_dir / n).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal binary to exist
    return img_dir


def test_write_csv_no_preannotations(tmp_path):
    images = ["C1.jpg", "C2.jpg"]
    make_dummy_images(tmp_path, images)
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
