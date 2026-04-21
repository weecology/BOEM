"""Smoke tests for H-CAST stack imports and optional checkpoint workflows."""

import os

import pandas as pd
import pytest
import torch

HCAST_MODULES = [
    "src.hcast.cast_models.cast_deit_hier",
    "src.hcast.cast_models.cast_deit",
    "src.hcast.cast_models.cast",
    "src.hcast.cast_models.modules",
    "src.hcast.cast_models.graph_pool",
    "src.hcast.deit.models_hier",
    "src.hcast.deit.models",
]

HCAST_CKPT = "output/usgs_hcast_300_b256/best_checkpoint.pth"
HCAST_SPECIES_CSV = "output/species.csv"


def test_hcast_imports_and_cast_models():
    """All hcast modules import; timm can construct CAST heads (timm / API regressions)."""
    from timm.models import create_model

    for module_name in HCAST_MODULES:
        __import__(module_name)

    for name in ("cast_small", "cast_base"):
        model = create_model(
            name,
            pretrained=False,
            num_classes=100,
            img_size=224,
            nb_classes=[100, 30, 14],
        )
        del model


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(HCAST_CKPT), reason="H-CAST checkpoint not present")
def test_load_checkpoint():
    """Load checkpoint and run one forward pass (requires local artifact)."""
    from src.hierarchical import load_hcast_model

    label_csv = HCAST_SPECIES_CSV if os.path.isfile(HCAST_SPECIES_CSV) else None
    model = load_hcast_model(checkpoint_path=HCAST_CKPT, label_csv=label_csv)

    device = model.device
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_superpixels = torch.randint(0, 10, (1, 224, 224)).to(device)

    with torch.no_grad():
        model.predict_logits(dummy_image, dummy_superpixels)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isfile(HCAST_CKPT) or not os.path.isfile(HCAST_SPECIES_CSV),
    reason="H-CAST checkpoint and output/species.csv required",
)
def test_load_and_classify_smoke():
    """End-to-end classify_dataframe on a tiny frame (requires checkpoint + species list)."""
    from src.hierarchical import load_hcast_model, classify_dataframe

    model = load_hcast_model(checkpoint_path=HCAST_CKPT, label_csv=HCAST_SPECIES_CSV)

    df = pd.DataFrame(
        {
            "image_path": ["turtle_crop.png"],
            "xmin": [0],
            "ymin": [0],
            "xmax": [220],
            "ymax": [195],
            "score": [0.9],
            "label": ["Objct"],
        }
    )

    out = classify_dataframe(
        predictions=df,
        image_dir="tests/data",
        model=model,
        batch_size=2,
        num_workers=0,
    )

    assert isinstance(out, pd.DataFrame)
    for col in ("hcast_species", "hcast_genus", "hcast_family"):
        assert col in out.columns
    for col in df.columns:
        assert col in out.columns
