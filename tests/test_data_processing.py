import pytest
import pandas as pd
import numpy as np
import cv2
import os
from src.data_processing import density_cropping, adjust_coordinates, merge_crop_predictions

@pytest.fixture
def sample_predictions():
    """Create sample predictions with dense areas."""
    rng = np.random.default_rng(42)
    # Create two dense clusters and some noise
    cluster1 = pd.DataFrame({
        'xmin': rng.uniform(100, 200, 5),
        'ymin': rng.uniform(100, 200, 5),
        'xmax': rng.uniform(200, 300, 5),
        'ymax': rng.uniform(200, 300, 5),
        'label': ['Bird'] * 5,
        'score': rng.uniform(0.8, 0.9, 5)
    })
    
    cluster2 = pd.DataFrame({
        'xmin': rng.uniform(500, 600, 5),
        'ymin': rng.uniform(500, 600, 5),
        'xmax': rng.uniform(600, 700, 5),
        'ymax': rng.uniform(600, 700, 5),
        'label': ['Bird'] * 5,
        'score': rng.uniform(0.8, 0.9, 5)
    })
    
    noise = pd.DataFrame({
        'xmin': rng.uniform(0, 1000, 3),
        'ymin': rng.uniform(0, 1000, 3),
        'xmax': rng.uniform(0, 1000, 3),
        'ymax': rng.uniform(0, 1000, 3),
        'label': ['Bird'] * 3,
        'score': rng.uniform(0.8, 0.9, 3)
    })
    
    return pd.concat([cluster1, cluster2, noise], ignore_index=True)

@pytest.fixture
def test_image(tmp_path):
    """Create a test image."""
    img_path = tmp_path / "test.jpg"
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

def test_density_cropping(sample_predictions, test_image):
    """Test density-based cropping."""
    result = density_cropping(
        predictions=sample_predictions,
        image_path=test_image,
        min_density=3,
        eps=100,
        min_samples=3
    )
    
    assert 'crops' in result
    assert 'clusters' in result
    assert len(result['crops']) > 0
    assert len(result['clusters']) == len(sample_predictions)
    
    # Check crop structure
    for crop in result['crops']:
        assert all(key in crop for key in ['xmin', 'ymin', 'xmax', 'ymax', 'path'])
        assert crop['num_detections'] >= 3
        assert os.path.exists(crop['path'])

def test_adjust_coordinates():
    """Test coordinate adjustment for crops."""
    predictions = pd.DataFrame({
        'xmin': [100, 200],
        'ymin': [100, 200],
        'xmax': [150, 250],
        'ymax': [150, 250]
    })
    
    crop_info = {
        'xmin': 50,
        'ymin': 50,
        'xmax': 300,
        'ymax': 300
    }
    
    adjusted = adjust_coordinates(predictions, crop_info)
    assert adjusted['xmin'].iloc[0] == 50  # 100 - 50
    assert adjusted['ymin'].iloc[0] == 50  # 100 - 50

def test_merge_crop_predictions():
    """Test merging predictions from multiple crops."""
    crops = [
        {'xmin': 100, 'ymin': 100, 'xmax': 300, 'ymax': 300},
        {'xmin': 500, 'ymin': 500, 'xmax': 700, 'ymax': 700}
    ]
    
    predictions = pd.DataFrame({
        'xmin': [50, 50, 50],
        'ymin': [50, 50, 50],
        'xmax': [100, 100, 100],
        'ymax': [100, 100, 100]
    })
    
    labels = [0, 0, 1]
    
    merged = merge_crop_predictions(crops, predictions, labels)
    assert len(merged) == len(predictions)
    assert merged['xmin'].iloc[0] == predictions['xmin'].iloc[0] + crops[0]['xmin']

def test_empty_predictions(test_image):
    """Test handling of empty predictions."""
    empty_preds = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax'])
    result = density_cropping(empty_preds, test_image)
    assert result['crops'] == []
    assert result['clusters'] == []

@pytest.mark.parametrize("min_density", [2, 3, 4])
def test_different_densities(sample_predictions, test_image, min_density):
    """Test different minimum density thresholds."""
    result = density_cropping(
        predictions=sample_predictions,
        image_path=test_image,
        min_density=min_density
    )
    
    for crop in result['crops']:
        assert crop['num_detections'] >= min_density 

def test_gather_data_newest_download_replaces_the_frame(tmp_path):
    """A relabel in Label Studio arrives as a newer CSV for the same frame. The newer download
    must replace the frame's rows, not be unioned with them, or the box is counted under both labels."""
    import pandas as pd
    from src.annotators import gather_data

    box = dict(xmin=10, ymin=10, xmax=50, ymax=50, annotator="a", flight_name="f")
    pd.DataFrame([{**box, "label": "Tursiops truncatus", "image_path": "a.jpg"},
                  {**box, "label": "Larus argentatus", "image_path": "b.jpg"}]).to_csv(tmp_path / "20260903_120646.csv", index=False)
    pd.DataFrame([{**box, "label": "Fregata magnificens", "image_path": "a.jpg"}]).to_csv(tmp_path / "20261006_155225.csv", index=False)

    df = gather_data(str(tmp_path), str(tmp_path))
    assert dict(zip(df.image_path, df.label)) == {"a.jpg": "Fregata magnificens", "b.jpg": "Larus argentatus"}


def test_rejected_box_is_false_positive_not_its_prefilled_species(tmp_path):
    """Rejecting a box in Label Studio changes the rectangle label to FalsePositive but leaves the
    per-region taxonomy on the model's species. Both the parser and the reader of already-downloaded
    CSVs must resolve that box to FalsePositive, or the rejection reaches reports as a human label."""
    import pandas as pd
    from src.annotators import gather_data
    from src.label_studio import convert_json_to_dataframe

    value = dict(x=10, y=10, width=5, height=5)
    region = lambda id, **v: dict(id=id, original_width=100, original_height=100, value={**value, **v})
    parsed = convert_json_to_dataframe([
        region("r1", rectanglelabels=["FalsePositive"]), region("r1", taxonomy=[["Aves", "Somateria mollissima"]]),
        region("r2", rectanglelabels=["Object"]), region("r2", taxonomy=[["Aves", "Larus argentatus"]]),
    ])
    assert sorted(parsed.label) == ["FalsePositive", "Larus argentatus"]

    box = dict(xmin=10, ymin=10, xmax=50, ymax=50, image_path="a.jpg", annotator="a", flight_name="f")
    pd.DataFrame([{**box, "label": "FalsePositive"}, {**box, "label": "Somateria mollissima"},
                  {**box, "xmin": 60, "xmax": 90, "label": "Larus argentatus"}]).to_csv(tmp_path / "20260822_142510.csv", index=False)
    df = gather_data(str(tmp_path), str(tmp_path))
    assert sorted(df.label) == ["FalsePositive", "Larus argentatus"]


def test_download_records_what_the_model_proposed_for_each_box(tmp_path):
    """Label Studio keeps a prediction's region id when the annotator accepts or edits the box, so
    the download carries the model's scores and label for it. A box the annotator drew has a new
    id and no model view; a rejected box stays FalsePositive."""
    from src.label_studio import download_completed_tasks

    value = dict(x=10, y=10, width=5, height=5)
    region = lambda id, **v: dict(id=id, original_width=100, original_height=100, value={**value, **v})
    rect = lambda id, label, **kw: dict(region(id, rectanglelabels=[label]), type="rectanglelabels", **kw)
    taxo = lambda id, sp, **kw: dict(region(id, taxonomy=[["Aves", sp]]), type="taxonomy", **kw)

    task = dict(
        id=1,
        data=dict(image="a.jpg", flight_name="f"),
        predictions=[dict(result=[
            rect("region0", "Object", score=0.91), taxo("region0", "Larus argentatus", score=0.62),
            rect("region1", "Object", score=0.75), taxo("region1", "Gavia immer", score=0.88),
        ])],
        annotations=[dict(created_username="a", result=[
            rect("region0", "Object"), taxo("region0", "Larus marinus"),        # relabelled
            rect("region1", "FalsePositive"), taxo("region1", "Gavia immer"),   # rejected
            dict(region("drawn", rectanglelabels=["Object"]), type="rectanglelabels"),
            dict(region("drawn", taxonomy=[["Aves", "Sterna hirundo"]]), type="taxonomy"),
        ])],
    )
    project = type("Project", (), {"get_tasks": lambda self: [task]})()

    df = download_completed_tasks(project, str(tmp_path)).set_index("label")
    assert sorted(df.index) == ["FalsePositive", "Larus marinus", "Sterna hirundo"]
    assert df.loc["Larus marinus", ["model_det_score", "model_label", "model_cls_score"]].tolist() == [0.91, "Larus argentatus", 0.62]
    assert df.loc["FalsePositive", "model_label"] == "Gavia immer"
    assert df.loc["Sterna hirundo", ["model_det_score", "model_label"]].isna().all()
    assert "region" not in df.columns
