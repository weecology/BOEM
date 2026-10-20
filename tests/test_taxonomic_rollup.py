"""Rollup rules: report the finest rank the CropModel and H-CAST agree on."""
import numpy as np
import pandas as pd
import pytest

from src import hierarchical


S2G = {
    "Gavia immer": "Gavia",
    "Gavia stellata": "Gavia",
    "Larus argentatus": "Larus",
    "Larus marinus": "Larus",
    "Sterna hirundo": "Sterna",
}
S2F = {
    "Gavia immer": "Gaviidae",
    "Gavia stellata": "Gaviidae",
    "Larus argentatus": "Laridae",
    "Larus marinus": "Laridae",
    "Sterna hirundo": "Laridae",
}


def _row(**kw):
    base = dict(
        cropmodel_label="Gavia immer", cropmodel_score=0.9,
        hcast_species="Gavia immer", hcast_genus="Gavia", hcast_family="Gaviidae",
        hcast_species_score=0.8, hcast_genus_score=0.85, hcast_family_score=0.95,
    )
    base.update(kw)
    return pd.Series(base)


def test_species_agreement_keeps_species():
    label, rank, score = hierarchical.resolve_row_rank(_row(), S2G, S2F)
    assert (label, rank) == ("Gavia immer", "species")
    assert score == pytest.approx(0.8)  # min(crop 0.9, hcast 0.8)


def test_species_disagreement_same_genus_rolls_up_to_genus():
    row = _row(hcast_species="Gavia stellata")
    label, rank, score = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert (label, rank) == ("Gavia", "genus")
    assert score == pytest.approx(0.85)  # min(crop 0.9, hcast_genus 0.85)


def test_genus_disagreement_same_family_rolls_up_to_family():
    row = _row(
        cropmodel_label="Larus argentatus",
        hcast_species="Sterna hirundo", hcast_genus="Sterna", hcast_family="Laridae",
    )
    label, rank, _ = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert (label, rank) == ("Laridae", "family")


def test_total_disagreement_is_unresolved_but_keeps_crop_label():
    row = _row(hcast_species="Larus argentatus", hcast_genus="Larus", hcast_family="Laridae")
    label, rank, _ = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert rank == "unresolved"
    assert label == "Gavia immer"  # traceability, not a count


def test_missing_hcast_passes_through_at_species():
    """hierarchical.checkpoint: null must not change reported labels."""
    row = _row(hcast_species=np.nan, hcast_genus=np.nan, hcast_family=np.nan)
    label, rank, score = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert (label, rank) == ("Gavia immer", "species")
    assert score == pytest.approx(0.9)


def test_genus_falls_back_to_first_token_when_unmapped():
    """Crop classes absent from the H-CAST label CSV still roll up."""
    row = _row(cropmodel_label="Chelonioidea sp", hcast_species="Chelonioidea whatever",
               hcast_genus="Chelonioidea")
    label, rank, _ = hierarchical.resolve_row_rank(row, {}, {})
    assert (label, rank) == ("Chelonioidea", "genus")


def test_min_consensus_score_demotes_one_rank():
    df = pd.DataFrame([_row()])  # species agreement, joint score 0.8
    out = hierarchical.resolve_taxonomic_rank(df, S2G, S2F, min_consensus_score=0.9)
    assert out["consensus_rank"].iloc[0] == "genus"
    assert out["consensus_label"].iloc[0] == "Gavia"


def test_min_consensus_score_leaves_confident_rows_alone():
    df = pd.DataFrame([_row()])
    out = hierarchical.resolve_taxonomic_rank(df, S2G, S2F, min_consensus_score=0.5)
    assert out["consensus_rank"].iloc[0] == "species"


def test_summary_orders_finest_rank_first():
    df = pd.DataFrame([
        _row(image_path="a.jpg"),
        _row(image_path="b.jpg", hcast_species="Gavia stellata"),
        _row(image_path="c.jpg", hcast_species="Larus argentatus",
             hcast_genus="Larus", hcast_family="Laridae"),
    ])
    out = hierarchical.resolve_taxonomic_rank(df, S2G, S2F)
    summary = hierarchical.summarize_taxonomic_rollup(out)
    assert summary["consensus_rank"].tolist() == ["species", "genus", "unresolved"]
    assert summary["n_observations"].sum() == 3


def test_empty_frame_is_handled():
    assert hierarchical.summarize_taxonomic_rollup(pd.DataFrame()).empty
    assert hierarchical.resolve_taxonomic_rank(pd.DataFrame()) is not None


# --- human annotations are never rolled up -------------------------------------


def test_human_row_keeps_human_label_at_verified():
    """A reviewed box reports the annotator's label, not the model's proposal for it, even when
    both models agree with each other."""
    row = _row(human_reviewed=True, human_label="Larus marinus", cropmodel_label="Gavia immer")
    label, rank, score = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert (label, rank) == ("Larus marinus", "verified")
    assert np.isnan(score)  # verified carries no confidence


def test_human_row_without_recorded_model_view_is_verified():
    """Annotations downloaded before the model view was captured carry no model columns."""
    row = _row(human_reviewed=True, human_label="Gavia immer", cropmodel_label=np.nan,
               cropmodel_score=np.nan, hcast_species=np.nan)
    assert hierarchical.resolve_row_rank(row, S2G, S2F)[:2] == ("Gavia immer", "verified")


def test_model_row_still_rolls_up_normally():
    row = _row(human_reviewed=False, hcast_species="Gavia stellata")
    label, rank, _ = hierarchical.resolve_row_rank(row, S2G, S2F)
    assert (label, rank) == ("Gavia", "genus")


def test_min_consensus_score_never_demotes_verified():
    df = pd.DataFrame([_row(human_reviewed=True, human_label="Gavia immer")])
    out = hierarchical.resolve_taxonomic_rank(df, S2G, S2F, min_consensus_score=0.99)
    assert out["consensus_rank"].iloc[0] == "verified"
    assert out["consensus_label"].iloc[0] == "Gavia immer"


def test_verified_sorts_ahead_of_species_in_summary():
    df = pd.DataFrame([
        _row(image_path="a.jpg", human_reviewed=True, human_label="Gavia immer"),
        _row(image_path="b.jpg", human_reviewed=False),
    ])
    out = hierarchical.resolve_taxonomic_rank(df, S2G, S2F)
    summary = hierarchical.summarize_taxonomic_rollup(out)
    assert summary["consensus_rank"].tolist() == ["verified", "species"]
    assert np.isnan(summary.loc[0, "mean_score"])          # verified carries no confidence
    assert summary.loc[1, "mean_score"] == pytest.approx(0.8)
