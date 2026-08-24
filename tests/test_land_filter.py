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


# --- validation-set band selection ---------------------------------------

@pytest.fixture
def scored_pool():
    """A pool shaped like a real flight: mostly water, a small land tail."""
    rng = np.random.default_rng(0)
    n = 20000
    prob = np.clip(rng.beta(1.2, 12, n), 0, 1)
    prob[:400] = rng.uniform(0.62, 0.99, 400)
    cam = rng.choice(["C1", "C2", "C3"], n)
    fno = rng.integers(0, 30000, n)
    import pandas as pd
    return pd.DataFrame({
        "image": [f"{c}_L1_F{f}_T.jpg" for c, f in zip(cam, fno)],
        "flight": rng.choice([f"F{i}" for i in range(8)], n),
        "prob": prob, "pred_land": prob > 0.610,
        "camera": cam, "frame_no": fno,
    }).drop_duplicates("image")


def test_bands_respect_their_probability_ranges(scored_pool):
    from scripts.upload_land_validation import select
    sel = select(scored_pool, 0.610)
    lo_hi = {"land_confident": (0.85, 1.01), "land_marginal": (0.610, 0.85),
             "boundary_below": (0.30, 0.610), "water_anchor": (0.0, 0.30)}
    for band, (lo, hi) in lo_hi.items():
        p = sel[sel.band == band].prob
        assert p.min() > lo and p.max() <= hi, band


def test_water_anchor_is_confident_water_not_threshold_hugging(scored_pool):
    """Regression: drawing this band boundary-first made it all p~=0.30 frames,
    which is not the confident-water tripwire it is supposed to be."""
    from scripts.upload_land_validation import select
    anchor = select(scored_pool, 0.610).query("band == 'water_anchor'").prob
    assert anchor.mean() < 0.15
    assert anchor.min() < 0.05


def test_selection_is_spread_across_flights(scored_pool):
    from scripts.upload_land_validation import select
    counts = select(scored_pool, 0.610).flight.value_counts()
    assert counts.max() - counts.min() <= 5, counts.to_dict()


def test_overlapping_frames_are_thinned(scored_pool):
    """Frames ~1 s apart show the same stretch of coast; don't pay to label both."""
    from scripts.mine_land_examples import MIN_FRAME_GAP
    from scripts.upload_land_validation import select
    sel = select(scored_pool, 0.610)
    for _, g in sel.groupby(["flight", "camera"]):
        if len(g) > 1:
            assert np.diff(np.sort(g.frame_no.values)).min() >= MIN_FRAME_GAP


def test_prediction_task_prefills_the_models_guess(scored_pool):
    from scripts.upload_land_validation import build_tasks, select
    sel = select(scored_pool, 0.610)
    tasks = build_tasks(sel, "m1")
    assert len(tasks) == len(sel)
    for task, row in zip(tasks, sel.itertuples()):
        choice = task["predictions"][0]["result"][0]["value"]["choices"][0]
        assert choice == ("Land" if row.pred_land else "Water")
        assert task["predictions"][0]["result"][0]["from_name"] == "surface"
        assert task["data"]["image"].endswith(row.image)
