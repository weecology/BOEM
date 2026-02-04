"""Test hierarchical model loading and catch all timm 1.0 import errors.

This test imports all hcast modules and attempts to create models to catch
any import errors from moving to timm 1.0.
"""
import os
import sys
import traceback
import pandas as pd
import torch

import pytest
from timm.models import create_model


def test_import_all_hcast_modules():
    """Test that all hcast modules can be imported without errors."""
    import_errors = []
    
    # Try importing all hcast modules
    modules_to_test = [
        "src.hcast.cast_models.cast_deit_hier",
        "src.hcast.cast_models.cast_deit",
        "src.hcast.cast_models.cast",
        "src.hcast.cast_models.modules",
        "src.hcast.cast_models.graph_pool",
        "src.hcast.deit.models_hier",
        "src.hcast.deit.models",
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ Successfully imported {module_name}")
        except Exception as e:
            error_msg = f"✗ Failed to import {module_name}: {type(e).__name__}: {e}"
            print(error_msg)
            import_errors.append((module_name, str(e), traceback.format_exc()))
    
    if import_errors:
        print("\n" + "="*80)
        print("IMPORT ERRORS FOUND:")
        print("="*80)
        for module_name, error, tb in import_errors:
            print(f"\nModule: {module_name}")
            print(f"Error: {error}")
            print(f"Traceback:\n{tb}")
        raise AssertionError(f"Found {len(import_errors)} import errors. See output above.")


def test_create_cast_models():
    """Test creating cast models without checkpoint to catch initialization errors."""
    from src.hcast.cast_models import cast_deit_hier
    from src.hcast.deit import models_hier
    
    creation_errors = []
    
    # Test creating cast_small model
    try:
        model = create_model(
            'cast_small',
            pretrained=False,
            num_classes=100,
            img_size=224,
            nb_classes=[100, 30, 14],
        )
        print("✓ Successfully created cast_small model")
        del model
    except Exception as e:
        error_msg = f"✗ Failed to create cast_small: {type(e).__name__}: {e}"
        print(error_msg)
        creation_errors.append(('cast_small', str(e), traceback.format_exc()))
    
    # Test creating cast_base model
    try:
        model = create_model(
            'cast_base',
            pretrained=False,
            num_classes=100,
            img_size=224,
            nb_classes=[100, 30, 14],
        )
        print("✓ Successfully created cast_base model")
        del model
    except Exception as e:
        error_msg = f"✗ Failed to create cast_base: {type(e).__name__}: {e}"
        print(error_msg)
        creation_errors.append(('cast_base', str(e), traceback.format_exc()))
    
    if creation_errors:
        print("\n" + "="*80)
        print("MODEL CREATION ERRORS FOUND:")
        print("="*80)
        for model_name, error, tb in creation_errors:
            print(f"\nModel: {model_name}")
            print(f"Error: {error}")
            print(f"Traceback:\n{tb}")
        raise AssertionError(f"Found {len(creation_errors)} model creation errors. See output above.")


# Skip if checkpoint is not found
pytestmark_load = pytest.mark.skipif(
    not os.path.exists("output/usgs_hcast_300_b256/best_checkpoint.pth"),
    reason="H-CAST checkpoint not found, skipping checkpoint loading test"
)

def test_load_checkpoint():
    """Test loading a checkpoint to catch any errors during model loading."""
    from src.hierarchical import load_hcast_model
    
    ckpt = "output/usgs_hcast_300_b256/best_checkpoint.pth"
    label_csv = "output/species.csv" if os.path.exists("output/species.csv") else None
    
    try:
        model = load_hcast_model(checkpoint_path=ckpt, label_csv=label_csv)
        print("✓ Successfully loaded checkpoint")
        
        # Test that model can do a forward pass with dummy data
        device = model.device
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        dummy_superpixels = torch.randint(0, 10, (1, 224, 224)).to(device)
        
        with torch.no_grad():
            outputs = model.predict_logits(dummy_image, dummy_superpixels)
        print("✓ Successfully ran forward pass")
        
    except Exception as e:
        error_msg = f"✗ Failed to load checkpoint: {type(e).__name__}: {e}"
        print(error_msg)
        print(f"Traceback:\n{traceback.format_exc()}")
        raise


def test_load_and_classify_smoke(tmp_path):
    """Original test - load checkpoint and classify a dataframe."""
    from src.hierarchical import load_hcast_model, classify_dataframe
    
    ckpt = "output/usgs_hcast_300_b256/best_checkpoint.pth"
    model = load_hcast_model(checkpoint_path=ckpt, label_csv="output/species.csv")

    # Build a minimal predictions dataframe mimicking detection outputs
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
