"""Tests for the learned land/water filter scorer.

These use a synthetic model file rather than the fitted one under
/blue/ewhite/b.weinstein/BOEM/annotations/land_screen/, so they do not depend on a
shared-filesystem artifact that `scripts/fit_land_filter.py` rewrites.
"""
import json

import numpy as np
import pytest

from src.land_filter import land_probability, load_model


@pytest.fixture
def model(tmp_path):
    """Two features, unit scaler, so the expected logit is computable by hand."""
    path = tmp_path / "land_model.json"
    path.write_text(json.dumps({
        "features": ["struct", "chroma"],
        "coef": [2.0, -1.0],
        "intercept": 0.5,
        "scaler_mean": [1.0, 2.0],
        "scaler_scale": [1.0, 1.0],
        "threshold": 0.625,
    }))
    return load_model(path)


def test_land_probability_matches_hand_computed_logistic(model):
    # z = 2*(3-1) + -1*(5-2) + 0.5 = 1.5
    p = land_probability({"struct": 3.0, "chroma": 5.0}, model)
    assert p == pytest.approx(1 / (1 + np.exp(-1.5)))


def test_probability_is_bounded_and_monotone_in_a_positive_coefficient(model):
    probs = [land_probability({"struct": s, "chroma": 2.0}, model) for s in range(-5, 6)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs)


def test_feature_order_follows_the_model_not_the_dict(model):
    """The dict is keyed, so insertion order must not change the score."""
    a = land_probability({"struct": 3.0, "chroma": 5.0}, model)
    b = land_probability({"chroma": 5.0, "struct": 3.0}, model)
    assert a == b


def test_scaler_is_actually_applied(tmp_path):
    """A non-unit scale must divide the centred feature."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "features": ["struct"], "coef": [1.0], "intercept": 0.0,
        "scaler_mean": [0.0], "scaler_scale": [4.0], "threshold": 0.5}))
    m = load_model(path)
    assert land_probability({"struct": 4.0}, m) == pytest.approx(1 / (1 + np.exp(-1.0)))


def test_load_model_is_cached_per_path(model, tmp_path):
    assert load_model(tmp_path / "land_model.json") is model
