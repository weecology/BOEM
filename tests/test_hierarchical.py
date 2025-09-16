import os
import pandas as pd
import pytest

from src.hierarchical import load_hcast_model, classify_dataframe, find_hcast_checkpoint


def test_find_checkpoint():
    ckpt = find_hcast_checkpoint("/home/b.weinstein/BOEM")
    # Optional: just assert function runs and returns None or a path
    assert ckpt is None or os.path.exists(ckpt)


def test_load_and_classify_smoke(tmp_path):
    # Try to load checkpoint if present
    repo_root = "/home/b.weinstein/BOEM"
    ckpt = find_hcast_checkpoint(repo_root)

    # If no checkpoint present, skip just this test (environment-dependent)
    if ckpt is None:
        pytest.skip("No H-CAST checkpoint found to load")

    model = load_hcast_model(repo_root=repo_root, checkpoint_path=ckpt)

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

