import os
import pandas as pd

import pytest
from src.hierarchical import load_hcast_model, classify_dataframe

# Skip entire module if checkpoint is not found
pytestmark = pytest.mark.skipif(
    not os.path.exists("output/usgs_hcast_300_b256/best_checkpoint.pth"),
    reason="H-CAST checkpoint not found, skipping hierarchical tests"
)
def test_load_and_classify_smoke(tmp_path):
    # Try to load checkpoint if present
    ckpt = "output/usgs_hcast_300_b256/best_checkpoint.pth"
    model = load_hcast_model(checkpoint_path=ckpt, label_csv="output/species.csv")

    # Build a minimal predictions dataframe mimicking detection outputs
    # Use full image dimensions (220x195) to ensure enough patches for pooling
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

    # Check that output is a DataFrame and has expected extra columns
    assert isinstance(out, pd.DataFrame)
    # Expect hcast_species, hcast_genus, hcast_family added
    for col in ["hcast_species", "hcast_genus", "hcast_family"]:
        assert col in out.columns, f"Missing column: {col}"
    # Should retain all the input columns as well
    for col in df.columns:
        assert col in out.columns
