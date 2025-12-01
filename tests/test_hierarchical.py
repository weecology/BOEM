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
    model = load_hcast_model(checkpoint_path=ckpt)

    # Build a minimal predictions dataframe mimicking detection outputs
    df = pd.DataFrame(
        {
            "image_path": ["turtle_crop.png"],
            "xmin": [10],
            "ymin": [10],
            "xmax": [50],
            "ymax": [50],
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

    assert "cropmodel_label" in out.columns
    assert "cropmodel_score" in out.columns
    assert len(out) == len(df)

