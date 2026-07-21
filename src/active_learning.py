import json
import os
import random
from pathlib import Path

import pandas as pd

from src import detection
from src import hierarchical


def load_species_to_genus_from_csv(label_csv_path: str | None) -> dict[str, str]:
    """Build species -> genus map from H-CAST label CSV (species + genus columns)."""
    if not label_csv_path or not os.path.isfile(label_csv_path):
        return {}
    df = pd.read_csv(label_csv_path).dropna(subset=["species"])
    if "genus" not in df.columns:
        return {}
    return dict(zip(df["species"].astype(str), df["genus"].astype(str)))


def crop_hcast_supported_match_or_genus_consistent(row: pd.Series, species_to_genus: dict[str, str]) -> bool:
    """H-CAST supports crop target species if species heads agree or genus agrees."""
    crop = row.get("cropmodel_label")
    if crop is None or pd.isna(crop):
        return False
    crop_s = str(crop)
    h_sp = row.get("hcast_species")
    if h_sp is None or pd.isna(h_sp):
        return False
    if crop_s == str(h_sp):
        return True
    crop_g = species_to_genus.get(crop_s)
    h_gn = row.get("hcast_genus")
    if crop_g is None or h_gn is None or pd.isna(h_gn):
        return False
    return str(crop_g) == str(h_gn)


def row_crop_hcast_disagrees(row: pd.Series, species_to_genus: dict[str, str], strict_genus_mismatch: bool) -> bool:
    """True when crop species != H-CAST species; if strict_genus_mismatch, exclude same-genus congeneric swaps."""
    crop = row.get("cropmodel_label")
    h_sp = row.get("hcast_species")
    if crop is None or pd.isna(crop) or h_sp is None or pd.isna(h_sp):
        return False
    crop_s, h_s = str(crop), str(h_sp)
    if crop_s == h_s:
        return False
    if not strict_genus_mismatch:
        return True
    crop_g = species_to_genus.get(crop_s)
    h_gn = row.get("hcast_genus")
    if crop_g is None or h_gn is None or pd.isna(h_gn):
        return True
    return str(crop_g) != str(h_gn)


def _row_min_class_confidence(row: pd.Series) -> float:
    vals = []
    for key in ("cropmodel_score", "hcast_species_score"):
        x = row.get(key)
        if x is not None and pd.notna(x) and isinstance(x, (int, float)):
            vals.append(float(x))
    return min(vals) if vals else 0.0


def format_ensemble_suggestion_line(row: pd.Series, species_to_genus: dict[str, str] | None) -> str | None:
    """Human-readable ensemble suggestion for Label Studio summary (genus fallback when species disagree)."""
    if "hcast_species" not in row or pd.isna(row.get("hcast_species")):
        return None
    crop = row.get("cropmodel_label")
    if crop is None or pd.isna(crop):
        return None
    crop_s, h_sp = str(crop), str(row["hcast_species"])
    h_gn = row.get("hcast_genus")
    gn_s = str(h_gn) if h_gn is not None and pd.notna(h_gn) else None
    if crop_s == h_sp:
        return f"Ensemble suggestion: {crop_s} (species agreement)"
    crop_genus = species_to_genus.get(crop_s) if species_to_genus else None
    if crop_genus is not None and gn_s is not None and str(crop_genus) == gn_s:
        return (
            f"Ensemble suggestion: genus {crop_genus} (species disagree: crop={crop_s}, H-CAST={h_sp})"
        )
    fa = row.get("hcast_family")
    fa_s = str(fa) if fa is not None and pd.notna(fa) else None
    parts = [f"Ensemble ambiguous: crop={crop_s}, H-CAST species={h_sp}"]
    if gn_s:
        parts.append(f", H-CAST genus={gn_s}")
    if fa_s:
        parts.append(f", H-CAST family={fa_s}")
    parts.append(" — verify taxonomy manually")
    return "".join(parts)


def _collect_leaf_aliases(node: dict) -> list[str]:
    """Collect all leaf (species) alias strings under a taxonomy node."""
    if node.get("isLeaf"):
        return [node["alias"]] if node.get("alias") else []
    return [
        a
        for child in node.get("children", [])
        for a in _collect_leaf_aliases(child)
    ]


def get_leaf_labels_for_taxonomy_aliases(
    taxonomy_path: str | Path,
    taxonomy_aliases: list[str],
) -> set[str]:
    """
    Expand taxonomy aliases (e.g. Aves, Mammalia, Cepphus) to all leaf (species) labels
    under those nodes in transformed_taxonomy.json. Use for strategy \"taxonomy\".
    """
    path = Path(taxonomy_path)
    data = json.loads(path.read_text())
    items = data.get("items", [])
    requested = set(taxonomy_aliases)
    result = set()

    def walk(node: dict) -> None:
        if node.get("alias") in requested:
            result.update(_collect_leaf_aliases(node))
        for child in node.get("children", []):
            walk(child)

    for item in items:
        walk(item)
    return result


def human_review(predictions, min_detection_score=0.6, min_classification_score=0.5, confident_threshold=0.5):
    """
    Predict on images and divide into confident and uncertain predictions.
    Args:
        confident_threshold (float): The threshold for confident predictions.
        min_classification_score (float, optional): The minimum class score for a prediction to be included. Defaults to 0.1.
        min_detection_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.1.
        predictions (pd.DataFrame, optional): A DataFrame of existing predictions. Defaults to None.
        Returns:
        tuple: A tuple of confident and uncertain predictions.
    """
    filtered_predictions = predictions[
        (predictions["score"] >= min_detection_score) &
        (predictions["cropmodel_score"] < min_classification_score)
    ]

    uncertain_predictions = filtered_predictions[
        filtered_predictions["cropmodel_score"] <= confident_threshold]

    confident_predictions = filtered_predictions[
        ~filtered_predictions["image_path"].isin(
            uncertain_predictions["image_path"])]

    return confident_predictions, uncertain_predictions


def generate_pool_predictions(
    pool,
    patch_size=512,
    patch_overlap=0.1,
    min_score=0.1,
    model=None,
    batch_size=16,
    pool_limit=1000,
    crop_model=None,
    hcast_model=None,
    image_dir=None,
    hcast_batch_size=None,
    hcast_workers=None,
    workers=0,
    metadata_lookup=None,
):
    """
    Generate predictions for the flight pool.

    Args:
        pool (str): List of image paths to predict on.
        patch_size (int, optional): The size of the image patches to predict on. Defaults to 512.
        patch_overlap (float, optional): The amount of overlap between image patches. Defaults to 0.1.
        min_score (float, optional): The minimum score for a prediction to be included. Defaults to 0.1.
        model (main.deepforest, optional): A trained deepforest model. Defaults to None.
        batch_size (int, optional): The batch size for prediction. Defaults to 16.
        crop_model (CropModel, optional): A deepforest.model.CropModel object. Defaults to None.
        pool_limit (int, optional): The maximum number of images to consider. Defaults to 1000.
        hcast_model (optional): H-CAST hierarchical model wrapper. Defaults to None.
        image_dir (str, optional): Root directory where images are located. Required if hcast_model is provided.
        hcast_batch_size (int, optional): Batch size for H-CAST classification.
        hcast_workers (int, optional): Number of workers for H-CAST DataLoader.
        workers (int, optional): Number of DataLoader workers for detection. Defaults to 0.
        metadata_lookup (dict, optional): Per-image metadata for metadata-aware CropModel inference.

    Returns:
        pd.DataFrame: A DataFrame of predictions (with hcast columns if hcast_model provided).
    """
    if pool_limit is not None and len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)
    print(f"Predicting on {len(pool)} images (pool_limit={pool_limit})")

    if not pool:
        # A flight can have nothing left to predict on (all images already annotated);
        # deepforest's predict_tile raises IndexError on an empty path list.
        return pd.DataFrame()

    preannotations = detection.predict(
        m=model,
        image_paths=pool,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model,
        workers=workers,
        metadata_lookup=metadata_lookup,
    )

    if preannotations is None:
        return None

    preannotations = preannotations[preannotations["score"] >= min_score]

    if hcast_model is not None:
        if image_dir is None:
            raise ValueError("image_dir is required when hcast_model is provided")
        preannotations = hierarchical.classify_dataframe(
            predictions=preannotations,
            image_dir=image_dir,
            model=hcast_model,
            batch_size=hcast_batch_size,
            num_workers=hcast_workers,
        )

    return preannotations


def _validate_target_labels(target_labels: list[str], valid_labels: set[str] | list[str] | None) -> None:
    """Raise ValueError if any target label is not in the crop model's label set (catches typos)."""
    if valid_labels is None:
        return
    valid = set(valid_labels)
    invalid = [lbl for lbl in target_labels if lbl not in valid]
    if invalid:
        raise ValueError(
            f"Target label(s) not in crop model label dict: {invalid}. "
            f"Valid labels ({len(valid)}): {sorted(valid)[:10]}{'...' if len(valid) > 10 else ''}. "
            "Check for typos or use a label that exists in the classification model."
        )


def select_images(
    preannotations,
    strategy,
    n=10,
    target_labels=None,
    min_score=0.3,
    drop_n_most_common=1,
    rarest_confidence_selection="lowest",
    min_classification_score=None,
    taxonomy_path=None,
    taxonomy_aliases=None,
    valid_labels=None,
    ensemble_target_mode: str = "crop_only",
    species_to_genus: dict[str, str] | None = None,
    disagreement_require_genus_mismatch: bool = False,
    disagreement_target_labels: list[str] | None = None,
):
    """
    Select images to annotate based on the strategy.

    Args:
        preannotations (pd.DataFrame): A DataFrame of predictions.
        strategy (str): One of random, most-detections, target-labels, taxonomy, rarest, model-disagreement.
        ensemble_target_mode (str): For target-labels/taxonomy: crop_only or match_or_genus_consistent.
        species_to_genus (dict): Species binomial -> genus for ensemble consistency checks.
        disagreement_require_genus_mismatch (bool): For model-disagreement: exclude congeneric species flips.
        disagreement_target_labels (list): Optional union filter on crop or H-CAST species labels.

    Returns:
        tuple: (chosen_image_paths, chosen_preannotations_df, al_stats_dict)
    """
    al_stats: dict = {}
    if preannotations.empty:
        return [], None, al_stats

    if strategy == "random":
        n = min(n, len(preannotations["image_path"].unique()))
        chosen_images = random.sample(preannotations["image_path"].unique().tolist(), n)

    else:
        preannotations = preannotations[preannotations["score"] >= min_score].copy()

        if strategy == "taxonomy":
            if taxonomy_aliases is None or not taxonomy_aliases:
                raise ValueError(
                    "taxonomy_aliases (e.g. ['Aves', 'Mammalia', 'Cepphus']) are required for the 'taxonomy' strategy."
                )
            if taxonomy_path is None:
                raise ValueError(
                    "taxonomy_path (path to transformed_taxonomy.json) is required for the 'taxonomy' strategy."
                )
            target_labels = list(get_leaf_labels_for_taxonomy_aliases(taxonomy_path, taxonomy_aliases))
            if not target_labels:
                return [], None, al_stats
            if valid_labels is not None:
                valid_set = set(valid_labels)
                target_labels = [lbl for lbl in target_labels if lbl in valid_set]
                if not target_labels:
                    raise ValueError(
                        "None of the taxonomy-expanded species are in the crop model label dict. "
                        "Check that the model was trained on species under the given taxonomy_aliases."
                    )
            strategy = "target-labels"

        if strategy == "target-labels":
            if target_labels is None:
                raise ValueError("Target labels are required for the 'target-labels' strategy.")
            _validate_target_labels(target_labels, valid_labels)

        if strategy == "most-detections":
            chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "target-labels":
            mask_crop_target = preannotations["cropmodel_label"].isin(target_labels)
            al_stats["al_target_crop_hits_rows"] = int(mask_crop_target.sum())

            if ensemble_target_mode == "match_or_genus_consistent":
                if "hcast_species" not in preannotations.columns:
                    raise ValueError(
                        "ensemble_target_mode='match_or_genus_consistent' requires hierarchical prediction columns "
                        "(hcast_species). Enable hierarchical.checkpoint or use ensemble_target_mode='crop_only'."
                    )
                sg = species_to_genus if species_to_genus is not None else {}
                supported = preannotations.apply(
                    lambda r: crop_hcast_supported_match_or_genus_consistent(r, sg),
                    axis=1,
                )
                mask = mask_crop_target & supported
                al_stats["al_target_after_ensemble_rows"] = int(mask.sum())
                al_stats["al_ensemble_target_mode"] = ensemble_target_mode
            elif ensemble_target_mode == "crop_only":
                mask = mask_crop_target
            else:
                raise ValueError(
                    f"Unknown ensemble_target_mode {ensemble_target_mode!r}. "
                    "Use 'crop_only' or 'match_or_genus_consistent'."
                )

            filtered = preannotations[mask]
            if filtered.empty:
                return [], None, al_stats
            chosen_images = (
                filtered.groupby("image_path")["score"].mean().sort_values(ascending=False).head(n).index.tolist()
            )
        elif strategy == "model-disagreement":
            if "hcast_species" not in preannotations.columns:
                raise ValueError(
                    "strategy 'model-disagreement' requires hcast_species (enable hierarchical.checkpoint)."
                )
            sg = species_to_genus if species_to_genus is not None else {}
            disagree_mask = preannotations.apply(
                lambda r: row_crop_hcast_disagrees(r, sg, disagreement_require_genus_mismatch),
                axis=1,
            )
            pool = preannotations[disagree_mask].copy()
            al_stats["al_disagreement_boxes_before_target_filter"] = len(pool)
            if disagreement_target_labels:
                tl = set(disagreement_target_labels)
                pool = pool[pool["cropmodel_label"].isin(tl) | pool["hcast_species"].isin(tl)]
            if min_classification_score is not None and "cropmodel_score" in pool.columns:
                pool = pool[pool["cropmodel_score"] >= min_classification_score]
            al_stats["al_disagreement_boxes_after_filters"] = len(pool)
            al_stats["al_disagreement_strict_genus"] = float(bool(disagreement_require_genus_mismatch))
            if pool.empty:
                print("model-disagreement: no disagreeing boxes after filters")
                return [], None, al_stats
            pool["_joint_conf"] = pool.apply(_row_min_class_confidence, axis=1)
            agg = (
                pool.groupby("image_path")
                .agg(disagreement_count=("image_path", "count"), mean_joint_conf=("_joint_conf", "mean"))
                .sort_values(["disagreement_count", "mean_joint_conf"], ascending=[False, False])
            )
            al_stats["al_disagreement_images_available"] = int(agg.shape[0])
            print(
                f"model-disagreement: {len(pool)} disagreeing boxes across "
                f"{agg.shape[0]} images (after filters); selecting top {n} images"
            )
            chosen_images = agg.head(n).index.tolist()
        elif strategy == "rarest":
            if min_classification_score is not None and "cropmodel_score" in preannotations.columns:
                preannotations = preannotations[preannotations["cropmodel_score"] >= min_classification_score]

            if drop_n_most_common > 0:
                most_common_labels = preannotations["cropmodel_label"].value_counts().nlargest(drop_n_most_common).index
                preannotations = preannotations[~preannotations["cropmodel_label"].isin(most_common_labels)]

            if preannotations.empty:
                return [], None, al_stats

            label_counts = preannotations.groupby("cropmodel_label").size().sort_values(ascending=True)
            preannotations["label_count"] = preannotations["cropmodel_label"].map(label_counts)

            if "cropmodel_score" in preannotations.columns:
                ascending_conf = rarest_confidence_selection == "lowest"
                preannotations.sort_values(["label_count", "cropmodel_score"], ascending=[True, ascending_conf], inplace=True)
            else:
                preannotations.sort_values("label_count", ascending=True, inplace=True)

            chosen_images = preannotations.drop_duplicates(subset=["image_path"], keep="first").head(n)["image_path"].tolist()
        else:
            raise ValueError(
                "Invalid strategy. Must be one of 'random', 'most-detections', 'target-labels', 'taxonomy', "
                "'rarest', or 'model-disagreement'."
            )

    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]
    al_stats["al_selected_images"] = len(chosen_images)
    return chosen_images, chosen_preannotations, al_stats
