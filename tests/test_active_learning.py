import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from deepforest import main
from deepforest.model import CropModel
from deepforest.utilities import read_file

from src.active_learning import (
    generate_pool_predictions,
    get_leaf_labels_for_taxonomy_aliases,
    select_images,
)

@pytest.fixture
def performance():
    return {"detection": {"mAP":{"map":0.9}}, "confident_classification": {"accuracy": 0.8}}

@pytest.fixture
def detection_model(comet_logger):
    """Create a mock deepforest model that produces bounding box predictions."""
    class MockDeepForest(main.deepforest):
        def __init__(self, label_dict, random=True):
            super().__init__(label_dict=label_dict, num_classes=len(label_dict))
            self.random = random
            self.comet_logger = comet_logger

        def predict_tile(self, raster_paths, patch_size=450, patch_overlap=0, return_plot=False, crop_model=None):
            # Support list or single path
            if not isinstance(raster_paths, list):
                raster_paths = [raster_paths]

            frames = []
            for raster_path in raster_paths:
                if "empty" in raster_path:
                    continue
                num_predictions = np.random.randint(1, 4)
                df = pd.DataFrame({
                        'xmin': np.random.randint(0, 800, num_predictions),
                        'ymin': np.random.randint(0, 600, num_predictions),
                        'xmax': np.random.randint(800, 1000, num_predictions),
                        'ymax': np.random.randint(600, 800, num_predictions),
                        'label': ['Object'] * num_predictions,
                        'cropmodel_label': [0] * num_predictions,
                        'score': np.random.uniform(0.1, 0.99, num_predictions),
                        'image_path': [os.path.basename(raster_path)] * num_predictions
                    })
                frames.append(read_file(df))
            if len(frames) == 0:
                return None
            return pd.concat(frames, ignore_index=True)

              
    return MockDeepForest(label_dict={"Object": 0})

@pytest.fixture
def random_crop_model():
    m = CropModel()
    m.label_dict = {"Bird": 0,"Mammal":1}
    return m


def test_generate_train_image_pool(detection_model):
    pool = [os.path.join("tests/data", f) for f in os.listdir("tests/data") if f.lower().endswith(".jpg")]
    train_image_pool = generate_pool_predictions(
        pool=pool,
        model=detection_model,
        patch_size=450,
        patch_overlap=0,
        min_score=0,
    )
    assert len(train_image_pool) > 0

def test_select_train_images(detection_model):
    pool = [os.path.join("tests/data", f) for f in os.listdir("tests/data") if f.lower().endswith(".jpg")]
    train_image_pool = generate_pool_predictions(
        pool=pool,
        patch_size=450,
        model=detection_model,
        patch_overlap=0,
        min_score=0.5,
    )
    chosen_images, _ = select_images(
        preannotations=train_image_pool,
        strategy="random",
        n=1,
    )
    assert len(chosen_images) > 0


def test_get_leaf_labels_for_taxonomy_aliases():
    """Taxonomy expansion: Aves -> bird species, Cepphus -> Cepphus grylle."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "transformed_taxonomy.json"
    if not path.exists():
        pytest.skip("transformed_taxonomy.json not found")
    aves = get_leaf_labels_for_taxonomy_aliases(path, ["Aves"])
    assert len(aves) > 1
    cepphus = get_leaf_labels_for_taxonomy_aliases(path, ["Cepphus"])
    assert cepphus == {"Cepphus grylle"}
    combined = get_leaf_labels_for_taxonomy_aliases(path, ["Aves", "Cepphus"])
    assert "Cepphus grylle" in combined
    assert len(combined) >= len(aves)


def test_select_images_taxonomy_strategy():
    """Taxonomy strategy expands aliases to leaf labels and selects images with those labels."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "transformed_taxonomy.json"
    if not path.exists():
        pytest.skip("transformed_taxonomy.json not found")
    # Preannotations with one image that has a bird species label
    preannotations = pd.DataFrame({
        "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"],
        "cropmodel_label": ["Cepphus grylle", "Actitis macularius", "Object"],
        "score": [0.9, 0.8, 0.7],
    })
    chosen_images, chosen_pre = select_images(
        preannotations=preannotations,
        strategy="taxonomy",
        n=5,
        taxonomy_path=path,
        taxonomy_aliases=["Cepphus"],
    )
    assert len(chosen_images) >= 1
    assert "img1.jpg" in chosen_images
    # Only Cepphus grylle is under Cepphus; img2 has Object so should not be selected
    assert "img2.jpg" not in chosen_images


def test_select_images_target_labels_validates_against_crop_model():
    """Target-labels strategy raises when a label is not in the crop model (typo check)."""
    preannotations = pd.DataFrame({
        "image_path": ["img1.jpg"],
        "cropmodel_label": ["Cepphus grylle"],
        "score": [0.9],
    })
    valid = {"Cepphus grylle", "Actitis macularius"}

    # All valid: succeeds
    chosen_images, _ = select_images(
        preannotations=preannotations,
        strategy="target-labels",
        n=5,
        target_labels=["Cepphus grylle"],
        valid_labels=valid,
    )
    assert "img1.jpg" in chosen_images

    # Typo / unknown label: raises
    with pytest.raises(ValueError, match="not in crop model label dict"):
        select_images(
            preannotations=preannotations,
            strategy="target-labels",
            n=5,
            target_labels=["Cepphus grille"],  # typo: grille vs grylle
            valid_labels=valid,
        )
