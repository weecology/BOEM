import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from deepforest.utilities import read_file

from src.active_learning import (
    format_ensemble_suggestion_line,
    generate_pool_predictions,
    get_leaf_labels_for_taxonomy_aliases,
    row_crop_hcast_disagrees,
    select_images,
)


@pytest.fixture
def detection_model(comet_logger):
    """Duck-typed stand-in for a detection model (avoids deepforest ctor API churn)."""

    class MockDetectionModel:
        def __init__(self):
            self.comet_logger = comet_logger
            self.config = {"batch_size": 1, "workers": 0}

        def predict_tile(
            self,
            raster_paths,
            patch_size=450,
            patch_overlap=0,
            crop_model=None,
            **kwargs,
        ):
            if not isinstance(raster_paths, list):
                raster_paths = [raster_paths]

            frames = []
            for raster_path in raster_paths:
                if "empty" in raster_path:
                    continue
                num_predictions = np.random.randint(1, 4)
                df = pd.DataFrame(
                    {
                        "xmin": np.random.randint(0, 800, num_predictions),
                        "ymin": np.random.randint(0, 600, num_predictions),
                        "xmax": np.random.randint(800, 1000, num_predictions),
                        "ymax": np.random.randint(600, 800, num_predictions),
                        "label": ["Object"] * num_predictions,
                        "cropmodel_label": [0] * num_predictions,
                        "score": np.random.uniform(0.1, 0.99, num_predictions),
                        "image_path": [os.path.basename(raster_path)] * num_predictions,
                    }
                )
                frames.append(read_file(df, root_dir=os.path.dirname(raster_path)))
            if len(frames) == 0:
                return None
            return pd.concat(frames, ignore_index=True)

    return MockDetectionModel()


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
        min_score=0,
    )
    chosen_images, _, _stats = select_images(
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
    preannotations = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"],
            "cropmodel_label": ["Cepphus grylle", "Actitis macularius", "Object"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    chosen_images, _chosen_pre, stats = select_images(
        preannotations=preannotations,
        strategy="taxonomy",
        n=5,
        taxonomy_path=path,
        taxonomy_aliases=["Cepphus"],
    )
    assert len(chosen_images) >= 1
    assert "img1.jpg" in chosen_images
    assert "img2.jpg" not in chosen_images
    assert stats.get("al_target_crop_hits_rows") == 1


def test_select_images_target_labels_validates_against_crop_model():
    """Target-labels strategy raises when a label is not in the crop model (typo check)."""
    preannotations = pd.DataFrame(
        {
            "image_path": ["img1.jpg"],
            "cropmodel_label": ["Cepphus grylle"],
            "score": [0.9],
        }
    )
    valid = {"Cepphus grylle", "Actitis macularius"}

    chosen_images, _, _ = select_images(
        preannotations=preannotations,
        strategy="target-labels",
        n=5,
        target_labels=["Cepphus grylle"],
        valid_labels=valid,
    )
    assert "img1.jpg" in chosen_images

    with pytest.raises(ValueError, match="not in crop model label dict"):
        select_images(
            preannotations=preannotations,
            strategy="target-labels",
            n=5,
            target_labels=["Cepphus grille"],
            valid_labels=valid,
        )


def test_match_or_genus_consistent_requires_hcast():
    preannotations = pd.DataFrame(
        {
            "image_path": ["a.jpg"],
            "cropmodel_label": ["Foo bar"],
            "score": [0.9],
        }
    )
    with pytest.raises(ValueError, match="match_or_genus_consistent"):
        select_images(
            preannotations=preannotations,
            strategy="target-labels",
            n=5,
            target_labels=["Foo bar"],
            ensemble_target_mode="match_or_genus_consistent",
            species_to_genus={"Foo bar": "Foo"},
        )


def test_match_or_genus_consistent_filters_rows():
    sg = {"AAA bbb": "AAA", "CCC ddd": "CCC"}
    preannotations = pd.DataFrame(
        {
            "image_path": ["x.jpg", "y.jpg", "z.jpg"],
            "cropmodel_label": ["AAA bbb", "AAA bbb", "CCC ddd"],
            "score": [0.9, 0.9, 0.9],
            "hcast_species": ["AAA bbb", "XXX yyy", "CCC ddd"],
            "hcast_genus": ["AAA", "XXX", "CCC"],
        }
    )
    chosen, _, stats = select_images(
        preannotations=preannotations,
        strategy="target-labels",
        n=5,
        target_labels=["AAA bbb", "CCC ddd"],
        ensemble_target_mode="match_or_genus_consistent",
        species_to_genus=sg,
    )
    assert "x.jpg" in chosen
    assert "z.jpg" in chosen
    assert "y.jpg" not in chosen
    assert stats["al_target_crop_hits_rows"] == 3
    assert stats["al_target_after_ensemble_rows"] == 2


def test_model_disagreement_strategy_ranking():
    sg = {}
    preannotations = pd.DataFrame(
        {
            "image_path": ["a.jpg", "a.jpg", "b.jpg", "b.jpg", "b.jpg"],
            "cropmodel_label": ["Sp one", "Sp one", "Sp two", "Sp two", "Sp two"],
            "score": [0.9, 0.9, 0.85, 0.85, 0.85],
            "cropmodel_score": [0.9, 0.95, 0.8, 0.85, 0.9],
            "hcast_species": ["Sp other", "Sp other", "Sp two", "Sp alt", "Sp alt"],
            "hcast_genus": ["G", "G", "G2", "G2", "G2"],
            "hcast_species_score": [0.88, 0.9, 0.85, 0.82, 0.87],
        }
    )
    chosen, _, stats = select_images(
        preannotations=preannotations,
        strategy="model-disagreement",
        n=2,
        min_score=0.3,
        species_to_genus=sg,
    )
    assert chosen[0] == "a.jpg"
    assert stats["al_disagreement_boxes_after_filters"] >= 4


def test_row_crop_hcast_disagree_strict_excludes_congener():
    sg = {"Uria aalge": "Uria", "Uria lomvia": "Uria"}
    row_match = pd.Series(
        {
            "cropmodel_label": "Uria aalge",
            "hcast_species": "Uria lomvia",
            "hcast_genus": "Uria",
        }
    )
    assert row_crop_hcast_disagrees(row_match, sg, strict_genus_mismatch=False)
    assert not row_crop_hcast_disagrees(row_match, sg, strict_genus_mismatch=True)


def test_format_ensemble_suggestion_line():
    sg = {"A b": "A"}
    row_agree = pd.Series({"cropmodel_label": "A b", "hcast_species": "A b", "hcast_genus": "A"})
    assert "species agreement" in format_ensemble_suggestion_line(row_agree, sg)
    row_genus = pd.Series({"cropmodel_label": "A b", "hcast_species": "A c", "hcast_genus": "A"})
    assert "genus A" in format_ensemble_suggestion_line(row_genus, sg)
    row_amb = pd.Series({"cropmodel_label": "A b", "hcast_species": "X y", "hcast_genus": "X"})
    assert "ambiguous" in format_ensemble_suggestion_line(row_amb, sg)
