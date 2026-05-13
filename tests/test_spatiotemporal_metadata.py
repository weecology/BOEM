import pandas as pd

from src.spatiotemporal_metadata import (
    build_crop_metadata_rows,
    flight_date,
    metadata_lookup_for_images,
)


def test_flight_date_from_jpg_flight_name():
    assert flight_date("JPG_20241220_104800") == "2024-12-20"


def test_build_crop_metadata_rows(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame({
        "Basename": ["C1_L1_F0001_T20241220_104801_000"],
        "Lat": [28.5],
        "Lon": [-89.25],
    }).to_csv(metadata_dir / "20241220_104800_captures.csv", index=False)

    annotations = pd.DataFrame({
        "image_path": ["C1_L1_F0001_T20241220_104801_000.jpg"],
        "label": ["genus species"],
        "xmin": [10],
        "ymin": [10],
        "xmax": [20],
        "ymax": [20],
    })

    rows = build_crop_metadata_rows(
        annotations=annotations,
        metadata_dir=metadata_dir,
        default_flight_name="JPG_20241220_104800",
    )

    assert rows.to_dict(orient="records") == [{
        "filename": "C1_L1_F0001_T20241220_104801_000_0.png",
        "lat": 28.5,
        "lon": -89.25,
        "date": "2024-12-20",
    }]


def test_metadata_lookup_for_images(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame({
        "Basename": ["ATL_0001"],
        "Lat": [38.1],
        "Lon": [-74.2],
    }).to_csv(metadata_dir / "20250105_090000_captures.csv", index=False)

    lookup = metadata_lookup_for_images(
        image_paths=["/tmp/ATL_0001.jpg"],
        metadata_dir=metadata_dir,
        default_flight_name="JPG_20250105_090000",
    )

    assert lookup["ATL_0001.jpg"] == {
        "lat": 38.1,
        "lon": -74.2,
        "date": "2025-01-05",
    }
    assert lookup["ATL_0001"] == lookup["ATL_0001.jpg"]
