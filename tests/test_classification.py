import os

import pandas as pd
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def sample_annotations():
    data = {
        "image_path": ["birds.jpg", "birds.jpg"],
        "label": ["genus species1", "genus species2"],
        "xmin": [10, 20],
        "ymin": [10, 20],
        "xmax": [50, 60],
        "ymax": [50, 60],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    from deepforest.model import CropModel

    model = CropModel()
    model.create_model(num_classes=2)
    return model


def test_preprocess_images(sample_model, sample_annotations, tmp_path):
    from src.classification import preprocess_images

    root_dir = "tests/data/"
    save_dir = tmp_path / "crops"
    os.makedirs(save_dir, exist_ok=True)
    preprocess_images(sample_model, sample_annotations, root_dir, save_dir)
    assert os.path.exists(save_dir)


def test_preprocess_and_train(sample_annotations, tmp_path):
    from src.classification import preprocess_and_train

    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint = None
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    assert model is not None


def test_preprocess_and_train_with_checkpoint(sample_annotations, tmp_path):
    from src.classification import preprocess_and_train

    train_df = sample_annotations
    validation_df = sample_annotations
    checkpoint_dir = tmp_path / "checkpoints"
    train_image_dir = "tests/data/"
    train_crop_image_dir = tmp_path / "crops/train"
    val_crop_image_dir = tmp_path / "crops/val"
    os.makedirs(train_crop_image_dir, exist_ok=True)
    os.makedirs(val_crop_image_dir, exist_ok=True)

    model = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=None,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved = os.path.join(checkpoint_dir, "tmp.ckpt")
    model.trainer.save_checkpoint(saved)

    model2 = preprocess_and_train(
        train_df=train_df,
        validation_df=validation_df,
        checkpoint=saved,
        checkpoint_dir=checkpoint_dir,
        image_dir=train_image_dir,
        train_crop_image_dir=train_crop_image_dir,
        val_crop_image_dir=val_crop_image_dir,
        lr=0.0001,
        batch_size=2,
        fast_dev_run=True,
        max_epochs=1,
        workers=0,
        comet_logger=None,
    )
    assert model2 is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # An "A/B" label is annotator uncertainty, so it resolves to the genus -- never to
        # species A, which is what silently built the Larus delawarensis attractor class.
        ("Larus delawarensis/argentatus", "Larus sp"),
        ("Sterna hirundo/paradisaea", "Sterna sp"),
        ("Calonectris/Puffinus diomedea/gravis", "Calonectris sp"),
        ("Larus argentatus", "Larus argentatus"),
        ("Unknown/Other", "Unknown/Other"),  # not a taxon; stays droppable
    ],
)
def test_map_ambiguous_slash_labels(raw, expected):
    from src.classification import map_ambiguous_slash_labels

    assert map_ambiguous_slash_labels(pd.Series([raw])).iloc[0] == expected


def test_coarse_taxa_survive_the_two_word_filter():
    """Family-rank dolphins must reach the classifier, not be dropped as single tokens."""
    from src.classification import DOLPHIN_FAMILY_CLASS, map_dolphin_family_labels

    mapped = map_dolphin_family_labels(pd.Series(["Delphinidae", "Tursiops truncatus"]))
    assert mapped.iloc[0] == DOLPHIN_FAMILY_CLASS == "Delphinidae sp"
    assert mapped.iloc[1] == "Tursiops truncatus"
    assert all(len(str(v).split()) == 2 for v in mapped)


def test_indeterminate_classes_keep_real_ancestor_ids():
    """"Delphinidae sp" must share a family head target with a real dolphin species."""
    from scripts.taxonomy_hier import load_taxonomy_restricted_to_species

    nb_classes, name_to_ids = load_taxonomy_restricted_to_species(
        "taxonomy.json",
        ["Tursiops truncatus", "Larus argentatus", "Delphinidae sp", "Larus sp"],
        include_ancestor_labels=True,
    )
    assert name_to_ids["Delphinidae sp"][0] == name_to_ids["Tursiops truncatus"][0]
    assert name_to_ids["Delphinidae sp"][2] != name_to_ids["Tursiops truncatus"][2]
    # A genus-rank label determines both family and genus, only the species is synthetic.
    assert name_to_ids["Larus sp"][:2] == name_to_ids["Larus argentatus"][:2]
    assert name_to_ids["Larus sp"][2] != name_to_ids["Larus argentatus"][2]
    # Every id must be addressable by the model heads.
    for fid, gid, sid in name_to_ids.values():
        assert sid < nb_classes[0] and gid < nb_classes[1] and fid < nb_classes[2]
