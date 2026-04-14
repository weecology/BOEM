import json
import random
from pathlib import Path

from src import detection
from src import hierarchical


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

    # Split predictions into confident and uncertain
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

    Returns:
        pd.DataFrame: A DataFrame of predictions (with hcast columns if hcast_model provided).
    """
    if pool_limit is not None and len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)
    print(f"Predicting on {len(pool)} images (pool_limit={pool_limit})")

    preannotations = detection.predict(
        m=model,
        image_paths=pool,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model,
        workers=workers,
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
):
    """
    Select images to annotate based on the strategy.

    Args:
        preannotations (pd.DataFrame): A DataFrame of predictions.
        strategy (str): The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
            - "target-labels": Choose images with target labels (species-level).
            - "taxonomy": Like target-labels but taxonomy_aliases (e.g. Aves, Mammalia, Cepphus)
              are expanded to all leaf species under those nodes using transformed_taxonomy.json.
            - "rarest": Choose images with rarest class labels.
        n (int, optional): The number of images to choose. Defaults to 10.
        target_labels (list, optional): For target-labels: list of species labels. Defaults to None.
        min_score (float, optional): The minimum detection score for a prediction to be included. Defaults to 0.3.
        drop_n_most_common (int, optional): For rarest strategy, number of most common classes to drop. Defaults to 1.
        rarest_confidence_selection (str, optional): For rarest strategy, "highest" or "lowest" confidence selection. Defaults to "lowest".
        min_classification_score (float, optional): Minimum classification confidence score. Defaults to None (no filter).
        taxonomy_path (str | Path, optional): Path to transformed_taxonomy.json. Required for strategy "taxonomy".
        taxonomy_aliases (list[str], optional): For strategy "taxonomy": e.g. ["Aves", "Mammalia", "Cepphus"]. Defaults to None.
        valid_labels (set | list, optional): Crop model label set (e.g. label_dict.keys()). If provided, target-labels
            and taxonomy-expanded labels are validated to catch typos/misspellings.

    Returns:
        list: A list of image paths.
        pd.DataFrame: A DataFrame of preannotations for the chosen images.
    """
    if preannotations.empty:
        return [], None

    if strategy == "random":
        n = min(n, len(preannotations["image_path"].unique()))
        chosen_images = random.sample(preannotations["image_path"].unique().tolist(), n)

    else:
        preannotations = preannotations[preannotations["score"] >= min_score]

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
                return [], None
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
            # Sort images by total number of predictions
            chosen_images = preannotations.groupby("image_path").size().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "target-labels":
            # Filter images by target labels (already validated above if valid_labels provided)
            chosen_images = preannotations[preannotations.cropmodel_label.isin(target_labels)].groupby("image_path")["score"].mean().sort_values(ascending=False).head(n).index.tolist()
        elif strategy == "rarest":
            # Filter by minimum classification score if provided
            if min_classification_score is not None and "cropmodel_score" in preannotations.columns:
                preannotations = preannotations[preannotations["cropmodel_score"] >= min_classification_score]
            
            # Drop n most common classes
            if drop_n_most_common > 0:
                most_common_labels = preannotations["cropmodel_label"].value_counts().nlargest(drop_n_most_common).index
                preannotations = preannotations[~preannotations["cropmodel_label"].isin(most_common_labels)]
            
            if preannotations.empty:
                return [], None
            
            # Sort images by least common label
            label_counts = preannotations.groupby("cropmodel_label").size().sort_values(ascending=True)
            
            # Sort preannotations by least common label
            preannotations["label_count"] = preannotations["cropmodel_label"].map(label_counts)
            
            # Sort by label count first, then by confidence score
            if "cropmodel_score" in preannotations.columns:
                ascending_conf = rarest_confidence_selection == "lowest"
                preannotations.sort_values(["label_count", "cropmodel_score"], ascending=[True, ascending_conf], inplace=True)
            else:
                preannotations.sort_values("label_count", ascending=True, inplace=True)
            
            chosen_images = preannotations.drop_duplicates(subset=["image_path"], keep="first").head(n)["image_path"].tolist()
        else:
            raise ValueError(
                "Invalid strategy. Must be one of 'random', 'most-detections', 'target-labels', 'taxonomy', or 'rarest'."
            )

    # Get preannotations for chosen images
    chosen_preannotations = preannotations[preannotations["image_path"].isin(chosen_images)]

    # Chosen preannotations is a dict with image_path as the key
    return chosen_images, chosen_preannotations