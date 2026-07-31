"""Tests for Serenity bulk annotator integration."""
import pandas as pd
import pytest

from src.bulk_annotations import (
    annotation_row_to_crop_id,
    apply_bulk_overrides,
    reduce_bulk_to_latest,
)


def test_annotation_row_to_crop_id():
    assert annotation_row_to_crop_id("path/to/img.JPG", "Bird", 0) == "img_Bird_0.png"
    assert annotation_row_to_crop_id("C2_L2_F7077_T20210511_102727_295.JPG", "Bucephala albeola", 0) == "C2_L2_F7077_T20210511_102727_295_Bucephala albeola_0.png"


def test_reduce_bulk_to_latest_keeps_latest_per_image_id():
    bulk = pd.DataFrame({
        "image_id": ["crop1.png", "crop1.png", "crop2.png"],
        "new_label": ["A", "B", "C"],
        "set": ["train", "review", "review"],
        "timestamp": ["2025-01-01T10:00:00", "2025-01-02T10:00:00", "2025-01-01T10:00:00"],
    })
    out = reduce_bulk_to_latest(bulk)
    assert len(out) == 2
    crop1 = out[out["image_id"] == "crop1.png"].iloc[0]
    assert crop1["new_label"] == "B"
    assert crop1["set"] == "review"


def test_reduce_bulk_to_latest_empty():
    out = reduce_bulk_to_latest(pd.DataFrame())
    assert out.empty


def test_apply_bulk_overrides_replaces_label_when_crop_id_matches():
    # One row: img.jpg with label "Bird" at index 0 -> crop_id = img_Bird_0.png
    train = pd.DataFrame({
        "image_path": ["img.jpg"],
        "xmin": [10.0],
        "ymin": [20.0],
        "xmax": [30.0],
        "ymax": [40.0],
        "label": ["Bird"],
    })
    bulk = pd.DataFrame({
        "image_id": ["img_Bird_0.png"],
        "new_label": ["FalsePositive"],
        "set": ["review"],
    })
    t, v, r, n_over, n_add = apply_bulk_overrides(train, None, None, bulk, "/path/to/flight", predictions_df=None)
    assert n_over == 1
    assert n_add == 0
    assert t is not None and t.iloc[0]["label"] == "FalsePositive"


def test_apply_bulk_overrides_leaves_others_unchanged():
    train = pd.DataFrame({
        "image_path": ["a.jpg", "a.jpg", "b.jpg"],
        "xmin": [0, 0, 0],
        "ymin": [0, 0, 0],
        "xmax": [1, 1, 1],
        "ymax": [1, 1, 1],
        "label": ["Bird", "Mammal", "Bird"],
    })
    # Override only a_Bird_0.png
    bulk = pd.DataFrame({
        "image_id": ["a_Bird_0.png"],
        "new_label": ["FalsePositive"],
        "set": ["train"],
    })
    t, v, r, n_over, n_add = apply_bulk_overrides(train, None, None, bulk, "/flight", None)
    assert n_over == 1
    assert t is not None
    labels = t["label"].tolist()
    assert "FalsePositive" in labels
    assert "Mammal" in labels
    assert "Bird" in labels
    assert labels.count("Bird") == 1
    assert labels.count("FalsePositive") == 1


def test_apply_bulk_overrides_add_new_from_bulk_with_manifest():
    # No existing train row for crop_xyz; bulk has image_id crop_xyz with new_label; predictions has crop_xyz
    train = pd.DataFrame({
        "image_path": ["other.jpg"],
        "xmin": [0],
        "ymin": [0],
        "xmax": [1],
        "ymax": [1],
        "label": ["Bird"],
    })
    bulk = pd.DataFrame({
        "image_id": ["crop_xyz.png"],
        "new_label": ["Larus argentatus"],
        "set": ["train"],
    })
    pred = pd.DataFrame({
        "crop_image_id": ["crop_xyz.png"],
        "image_path": ["parent.JPG"],
        "xmin": [100],
        "ymin": [200],
        "xmax": [150],
        "ymax": [250],
        "flight_name": ["my_flight"],
    })
    t, v, r, n_over, n_add = apply_bulk_overrides(
        train, None, None, bulk, "/path/to/my_flight", predictions_df=pred
    )
    assert n_over == 0
    assert n_add == 1
    assert t is not None and len(t) == 2
    new_row = t[t["image_path"] == "parent.JPG"].iloc[0]
    assert new_row["label"] == "Larus argentatus"
    assert new_row["xmin"] == 100 and new_row["ymax"] == 250


@pytest.mark.skip(reason="SFTP connection can hang; manual or mocked test for fetch")
def test_fetch_bulk_annotations_csv_returns_none_on_invalid_host():
    """Fetch with invalid host should log and return None, not raise."""
    from src.bulk_annotations import fetch_bulk_annotations_csv
    class Cfg:
        user = "u"
        host = "invalid-host-that-does-not-resolve.example"
        key_filename = "/nonexistent/key"
    result = fetch_bulk_annotations_csv(Cfg(), "/some/path.csv")
    assert result is None
