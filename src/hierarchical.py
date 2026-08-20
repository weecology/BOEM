import os
from typing import Optional, Tuple, List, Dict

import torch
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model


def _infer_head_sizes_from_checkpoint(ckpt: Dict[str, torch.Tensor]) -> Tuple[int, Optional[int], Optional[int]]:
    species = None
    genus = None
    family = None

    if "head.weight" in ckpt:
        species = ckpt["head.weight"].shape[0]
    for key in ckpt.keys():
        if key.endswith("family_head.weight") and genus is None:
            genus = ckpt[key].shape[0]
        if key.endswith("manufacturer_head.weight"):
            family = ckpt[key].shape[0]

    if species is None:
        classifier_like = [v.shape[0] for k, v in ckpt.items() if k.endswith(".weight") and v.ndim == 2]
        species = max(classifier_like) if classifier_like else 1000

    return species, genus, family


def expand_bbox_to_square(
    xmin: float, ymin: float, xmax: float, ymax: float,
    width: int, height: int,
) -> Tuple[int, int, int, int]:
    """Grow a box to a square by extending the shorter side around the centre.

    Canonical implementation shared by H-CAST training
    (scripts/USGS_hierarchical.py) and inference (InferenceCropDataset) so the
    two paths cannot drift apart in crop geometry.
    """
    w = xmax - xmin
    h = ymax - ymin
    side = max(w, h)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = int(cx + side / 2), int(cy + side / 2)
    x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
    x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return x0, y0, x1, y1


def _default_transform(image_size: int = 224, eval_crop_ratio: Optional[float] = None):
    """Resize a crop to model input size.

    eval_crop_ratio=None squashes the crop to (image_size, image_size).
    eval_crop_ratio=r (e.g. 0.875) reproduces the training-time validation
    transform in scripts/USGS_hierarchical.py: resize the short side to
    image_size / r with bicubic interpolation, then centre-crop to image_size.
    """
    if eval_crop_ratio:
        resized = int(image_size / eval_crop_ratio)
        return transforms.Compose([
            transforms.Resize(resized, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])


class HCastWrapper:
    """Wrapper to run hierarchical DeiT ViT for crop classification.

    Model outputs: (species_logits, genus_logits, family_logits) for 3-level hierarchy.
    The head names in the model are: head (species), family_head (genus), manufacturer_head (family) --
    inherited from the HCAST Aircraft naming convention.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        label_dict: Dict[str, int],
        image_size: int = 224,
        species_to_genus: Optional[Dict[int, str]] = None,
        species_df: Optional[pd.DataFrame] = None,
        genus_label_dict: Optional[Dict[int, str]] = None,
        family_label_dict: Optional[Dict[int, str]] = None,
        expand_pixels: int = 0,
        square: bool = False,
        eval_crop_ratio: Optional[float] = None,
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.label_dict = label_dict
        self.species_numeric_to_label = {v: k for k, v in label_dict.items() if k.startswith("species_")}
        self.genus_numeric_to_label = genus_label_dict or {v: k for k, v in label_dict.items() if k.startswith("genus_")}
        self.family_numeric_to_label = family_label_dict or {v: k for k, v in label_dict.items() if k.startswith("family_")}
        self.image_size = image_size
        # Crop geometry used to turn a detection box into a model input. These
        # must match how the checkpoint was trained: HierarchicalCropDataset in
        # scripts/USGS_hierarchical.py pads the box by --expand-pixels, squares
        # it, then applies the eval_crop_ratio=0.875 validation transform.
        self.expand_pixels = int(expand_pixels)
        self.square = bool(square)
        self.eval_crop_ratio = eval_crop_ratio
        self._transform = _default_transform(image_size, eval_crop_ratio)
        self.species_to_genus = species_to_genus or {}
        self.species_df = species_df

    def get_transform(self, augment: bool = False):
        return self._transform

    @torch.no_grad()
    def predict_logits(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Predict logits for a batch of images.

        Returns:
            Tuple of (species_logits, genus_logits, family_logits) for 3-level,
            or (species_logits, genus_logits) for 2-level.
        """
        batch = batch.to(self.device)
        outputs = self.model(batch)
        if isinstance(outputs, (list, tuple)):
            return tuple(outputs)
        return (outputs,)


def load_hcast_model(
    checkpoint_path,
    label_csv: str,
    device: Optional[torch.device] = None,
    expand_pixels: int = 0,
    square: bool = False,
    eval_crop_ratio: Optional[float] = None,
) -> HCastWrapper:
    """Load hierarchical DeiT ViT model from checkpoint and return a wrapper ready for inference.

    Args:
        checkpoint_path: Path to the checkpoint file
        label_csv: Path to CSV with columns: species, genus, family (optional "index" = species
            class index; optional "genus_index"/"family_index" = genus/family class indices).
        device: Device to load model on. If None, uses 'cuda' if available.
        expand_pixels: Context pixels added to every side of a detection box before cropping.
            Must match the --expand-pixels the checkpoint was trained with.
        square: Whether to square the (expanded) box before resizing, as training does.
        eval_crop_ratio: Validation resize ratio; 0.875 reproduces the training transform.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        model_state_dict = checkpoint

    from src.hcast.deit import models_hier  # noqa: F401

    args = checkpoint.get("args")
    if args is not None:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes[0],
            drop_rate=getattr(args, "drop", 0.0),
            drop_path_rate=getattr(args, "drop_path", 0.0),
            drop_block_rate=None,
            img_size=getattr(args, "input_size", 224),
            nb_classes=args.nb_classes,
        )
        img_size = getattr(args, "input_size", 224)
        nb_classes = args.nb_classes
    else:
        species_classes, genus_classes, family_classes = _infer_head_sizes_from_checkpoint(model_state_dict)
        nb_classes = [c for c in [species_classes, genus_classes, family_classes] if c is not None]
        model = create_model(
            "deit_small_patch16_224",
            pretrained=False,
            num_classes=nb_classes[0],
            img_size=224,
            nb_classes=nb_classes,
        )
        img_size = 224

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

    model = model.eval().to(device)

    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"label_csv is required for H-CAST; file not found: {label_csv}")

    df = pd.read_csv(label_csv)
    df = df.dropna(subset=['species'])
    species_df = df.copy()
    species_to_genus = {}
    label_dict = {}
    use_index_col = 'index' in df.columns
    if 'species' not in df.columns:
        raise ValueError(f"label_csv must contain a 'species' column: {label_csv}")
    for idx, row in df.iterrows():
        species_idx = int(row['index']) if use_index_col else idx
        label_dict[f"species_{row['species']}"] = species_idx
        if 'genus' in df.columns and pd.notna(row['genus']):
            species_to_genus[species_idx] = row['genus']

    genus_label_dict = {}
    if 'genus' in df.columns:
        # Prefer an explicit genus_index column. Without one, ids are inferred
        # from first-appearance order, which only matches the trained head if
        # the CSV happens to be ordered the same way the training run ordered
        # its genera -- so a mismatched CSV silently mislabels every genus.
        if 'genus_index' in df.columns:
            genus_df = df.dropna(subset=['genus_index']).drop_duplicates(subset=['genus'])
            pairs = [(int(row['genus_index']), row['genus']) for _, row in genus_df.iterrows()]
        else:
            genus_df = df.drop_duplicates(subset=['genus']).reset_index(drop=True)
            pairs = [(idx, row['genus']) for idx, row in genus_df.iterrows()]
        for idx, name in pairs:
            label_dict[f"genus_{name}"] = idx
            genus_label_dict[idx] = f"genus_{name}"
    if len(nb_classes) > 1:
        for idx in range(nb_classes[1]):
            if idx not in genus_label_dict:
                genus_label_dict[idx] = f"genus_{idx}"

    family_label_dict = {}
    if 'family' in df.columns:
        if 'family_index' in df.columns:
            family_df = df.dropna(subset=['family_index']).drop_duplicates(subset=['family'])
            pairs = [(int(row['family_index']), row['family']) for _, row in family_df.iterrows()]
        else:
            family_df = df.drop_duplicates(subset=['family']).reset_index(drop=True)
            pairs = [(idx, row['family']) for idx, row in family_df.iterrows()]
        for idx, name in pairs:
            label_dict[f"family_{name}"] = idx
            family_label_dict[idx] = f"family_{name}"
    if len(nb_classes) > 2:
        for idx in range(nb_classes[2]):
            if idx not in family_label_dict:
                family_label_dict[idx] = f"family_{idx}"

    return HCastWrapper(
        model=model, device=device, label_dict=label_dict, image_size=img_size,
        species_to_genus=species_to_genus, species_df=species_df,
        genus_label_dict=genus_label_dict, family_label_dict=family_label_dict,
        expand_pixels=expand_pixels, square=square, eval_crop_ratio=eval_crop_ratio,
    )


class InferenceCropDataset(Dataset):
    """Dataset for inference that crops images from bounding boxes in a predictions DataFrame."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        image_dir: str,
        transform=None,
        expand_pixels: int = 0,
        square: bool = False,
    ):
        self.transform = transform or _default_transform()
        self.image_dir = image_dir
        self.predictions = predictions.reset_index(drop=True)
        self.expand_pixels = int(expand_pixels)
        self.square = bool(square)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        row = self.predictions.iloc[index]
        full_path = os.path.join(self.image_dir, row['image_path'])
        with open(full_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        width, height = image.size
        xmin, ymin = float(row['xmin']), float(row['ymin'])
        xmax, ymax = float(row['xmax']), float(row['ymax'])
        if self.expand_pixels:
            xmin = max(0, xmin - self.expand_pixels)
            ymin = max(0, ymin - self.expand_pixels)
            xmax = min(width, xmax + self.expand_pixels)
            ymax = min(height, ymax + self.expand_pixels)
        if self.square:
            box = expand_bbox_to_square(xmin, ymin, xmax, ymax, width, height)
        else:
            box = (int(xmin), int(ymin), int(xmax), int(ymax))
        crop = image.crop(box)
        return self.transform(crop)


@torch.no_grad()
def classify_dataframe(
    predictions,
    image_dir: str,
    model: HCastWrapper,
    batch_size: int = 64,
    num_workers: int = 2,
    expand_pixels: Optional[int] = None,
    square: Optional[bool] = None,
    eval_crop_ratio: Optional[float] = None,
):
    """Add crop-level hierarchical classification to predictions DataFrame.

    Adds columns: hcast_species, hcast_genus, hcast_family, hcast_species_score,
    hcast_genus_score, hcast_family_score.
    """
    if predictions is None or len(predictions) == 0:
        return predictions

    # Fall back to the geometry the wrapper was loaded with; explicit arguments
    # override it so a sweep can vary one knob at a time.
    if expand_pixels is None:
        expand_pixels = getattr(model, "expand_pixels", 0)
    if square is None:
        square = getattr(model, "square", False)
    if eval_crop_ratio is None:
        eval_crop_ratio = getattr(model, "eval_crop_ratio", None)

    transform = _default_transform(image_size=model.image_size, eval_crop_ratio=eval_crop_ratio)
    ds = InferenceCropDataset(
        predictions, image_dir, transform=transform,
        expand_pixels=expand_pixels, square=square,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    all_species_idx: List[int] = []
    all_species_prob: List[float] = []
    all_genus_idx: List[int] = []
    all_genus_prob: List[float] = []
    all_family_idx: List[int] = []
    all_family_prob: List[float] = []

    for batch_images in dl:
        logits_tuple = model.predict_logits(batch_images)

        species_logits = logits_tuple[0]
        species_probs = torch.softmax(species_logits, dim=1)
        species_conf, species_idx = torch.max(species_probs, dim=1)
        all_species_idx.extend(species_idx.cpu().tolist())
        all_species_prob.extend(species_conf.cpu().tolist())

        if len(logits_tuple) > 1:
            genus_logits = logits_tuple[1]
            genus_probs = torch.softmax(genus_logits, dim=1)
            genus_conf, genus_idx = torch.max(genus_probs, dim=1)
            all_genus_idx.extend(genus_idx.cpu().tolist())
            all_genus_prob.extend(genus_conf.cpu().tolist())
        else:
            all_genus_idx.extend([None] * len(species_idx))
            all_genus_prob.extend([None] * len(species_idx))

        if len(logits_tuple) > 2:
            family_logits = logits_tuple[2]
            family_probs = torch.softmax(family_logits, dim=1)
            family_conf, family_idx = torch.max(family_probs, dim=1)
            all_family_idx.extend(family_idx.cpu().tolist())
            all_family_prob.extend(family_conf.cpu().tolist())
        else:
            all_family_idx.extend([None] * len(species_idx))
            all_family_prob.extend([None] * len(species_idx))

    def _map_labels(indices, numeric_to_label, prefix):
        labels = []
        for idx in indices:
            if idx is None:
                labels.append(None)
                continue
            label_key = numeric_to_label.get(idx, f"{prefix}_{idx}")
            if label_key.startswith(f"{prefix}_"):
                labels.append(label_key[len(prefix) + 1:])
            else:
                labels.append(label_key)
        return labels

    predictions = predictions.copy(deep=True)
    predictions["hcast_species"] = _map_labels(all_species_idx, model.species_numeric_to_label, "species")
    predictions["hcast_genus"] = _map_labels(all_genus_idx, model.genus_numeric_to_label, "genus")
    predictions["hcast_family"] = _map_labels(all_family_idx, model.family_numeric_to_label, "family")
    predictions["hcast_species_score"] = all_species_prob
    predictions["hcast_genus_score"] = all_genus_prob
    predictions["hcast_family_score"] = all_family_prob
    return predictions


def infer_head_sizes_from_checkpoint(checkpoint_path: str) -> List[int]:
    """Return list of output sizes for each head found in checkpoint state_dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    sizes = []
    seen = set()
    for k, v in state.items():
        kn = k.lower()
        if "head" in kn and k.endswith("weight") and v.ndim == 2:
            out_features = v.shape[0]
            if out_features not in seen:
                sizes.append(out_features)
                seen.add(out_features)
    if not sizes:
        for k, v in state.items():
            if v.ndim == 2 and ("classifier" in k or "fc" in k or "head" in k):
                sizes.append(v.shape[0])
    return sizes


# ---------------------------------------------------------------------------
# Taxonomic consensus rollup
#
# The crop CropModel and H-CAST are two independent opinions on the same box.
# Neither overrides the other: instead we report the FINEST rank at which they
# agree, so a disputed species becomes a genus-level (or family-level) record
# rather than a coin flip between two species names.
# ---------------------------------------------------------------------------

CONSENSUS_RANKS = ("verified", "species", "genus", "family", "unresolved")

# `set` values that mark a row as a human annotation rather than a model prediction.
# src/pipeline.py concatenates these into final_predictions alongside the model pool.
HUMAN_SETS = frozenset({"train", "validation", "reviewed"})


def _is_human_row(row) -> bool:
    """True for a human annotation, which is never rolled up.

    Preferred signal is the `set` column written by pipeline.py. The score==2.0
    sentinel is the fallback for callers that drop `set` (pipeline.py assigns 2.0
    to train/reviewed rows; validation rows may carry neither, hence both checks).
    """
    st = row.get("set")
    if st is not None and not pd.isna(st) and str(st) in HUMAN_SETS:
        return True
    for key in ("cropmodel_score", "score"):
        val = row.get(key)
        if val is not None and not pd.isna(val):
            try:
                if float(val) >= 2.0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def load_species_to_ranks(
    label_csv: Optional[str] = None,
    taxonomy_path: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build species -> genus and species -> family maps.

    Sources are merged in increasing order of priority, so the H-CAST label CSV
    (which defines the names H-CAST itself emits) wins any conflict:

    1. ``taxonomy_path`` (taxonomy.json) — broad coverage, includes crop-model
       species that H-CAST has never heard of.
    2. ``label_csv`` — the H-CAST species/genus/family table.

    Neither source covers everything: the crop model has 70 classes against
    H-CAST's 37, and synthetic labels such as ``Chelonioidea sp`` exist in no
    real taxonomy. Callers should fall back to :func:`_genus_of` for misses.
    """
    species_to_genus: Dict[str, str] = {}
    species_to_family: Dict[str, str] = {}

    if taxonomy_path and os.path.exists(taxonomy_path):
        try:
            from scripts.taxonomy_hier import load_taxonomy
            triples, _ = load_taxonomy(taxonomy_path)
            for family, genus, species in triples:
                species_to_genus[str(species)] = str(genus)
                species_to_family[str(species)] = str(family)
        except Exception as e:  # taxonomy is optional; never break the report over it
            print(f"[rollup] could not read taxonomy {taxonomy_path}: {e}")

    if label_csv and os.path.exists(label_csv):
        df = pd.read_csv(label_csv).dropna(subset=["species"])
        if "genus" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row["genus"]):
                    species_to_genus[str(row["species"])] = str(row["genus"])
        if "family" in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row["family"]):
                    species_to_family[str(row["species"])] = str(row["family"])

    return species_to_genus, species_to_family


def _genus_of(label, species_to_genus: Dict[str, str]) -> Optional[str]:
    """Genus for a species label, falling back to the first token of the binomial.

    The fallback is what keeps rollup working for the crop model's classes that
    are absent from the H-CAST label CSV. It is also correct for the synthetic
    ``Chelonioidea sp``, whose first token is the supra-generic name we want in
    the genus slot.
    """
    if label is None or pd.isna(label):
        return None
    label = str(label)
    mapped = species_to_genus.get(label)
    if mapped:
        return mapped
    first = label.split()[0] if label.split() else None
    return first or None


def _family_of(label, species_to_family: Dict[str, str]) -> Optional[str]:
    """Family for a species label. No token fallback — a binomial carries no family."""
    if label is None or pd.isna(label):
        return None
    return species_to_family.get(str(label))


def _score(row, key) -> Optional[float]:
    val = row.get(key)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _joint(*scores) -> float:
    """Conservative joint confidence: the minimum of the available scores.

    Matches ``active_learning._row_min_class_confidence`` so the number in the
    report means the same thing as the number used for image selection.
    """
    vals = [s for s in scores if s is not None]
    return min(vals) if vals else float("nan")


def resolve_row_rank(row, species_to_genus, species_to_family):
    """Resolve one box to (label, rank, score) at the finest rank the models agree on.

    The ladder, in order:

    * ``species``    — crop species == H-CAST species.
    * ``genus``      — species disagree, but the crop label's genus == H-CAST genus.
    * ``family``     — genus disagrees, but the crop label's family == H-CAST family.
    * ``unresolved`` — the two models agree at no rank. The crop label is kept in
      ``consensus_label`` for traceability but the rank marks it as unsupported,
      so downstream counts can exclude it.

    Human annotations short-circuit the whole ladder at ``verified`` — their label
    is kept regardless of any model prediction, and never demoted by
    ``min_consensus_score``.

    When the H-CAST columns are absent or null the row passes through at
    ``species`` rank with the crop label, preserving pre-rollup behaviour for
    runs with ``hierarchical.checkpoint: null``.
    """
    crop = row.get("cropmodel_label")
    if crop is None or pd.isna(crop):
        return None, "unresolved", float("nan")
    crop = str(crop)

    # A human annotation is ground truth: keep its label exactly, whatever any
    # model says. Score is NaN because "confidence" is meaningless here, and
    # carrying the 2.0 sentinel would poison mean_score in the summary.
    if _is_human_row(row):
        return crop, "verified", float("nan")

    crop_score = _score(row, "cropmodel_score")

    h_sp = row.get("hcast_species")
    if h_sp is None or pd.isna(h_sp):
        return crop, "species", _joint(crop_score)
    h_sp = str(h_sp)

    if crop == h_sp:
        return crop, "species", _joint(crop_score, _score(row, "hcast_species_score"))

    crop_genus = _genus_of(crop, species_to_genus)
    h_genus = row.get("hcast_genus")
    h_genus = None if (h_genus is None or pd.isna(h_genus)) else str(h_genus)
    if crop_genus is not None and h_genus is not None and crop_genus == h_genus:
        return crop_genus, "genus", _joint(crop_score, _score(row, "hcast_genus_score"))

    crop_family = _family_of(crop, species_to_family)
    h_family = row.get("hcast_family")
    h_family = None if (h_family is None or pd.isna(h_family)) else str(h_family)
    if crop_family is not None and h_family is not None and crop_family == h_family:
        return crop_family, "family", _joint(crop_score, _score(row, "hcast_family_score"))

    return crop, "unresolved", _joint(crop_score, _score(row, "hcast_species_score"))


def resolve_taxonomic_rank(
    predictions,
    species_to_genus: Optional[Dict[str, str]] = None,
    species_to_family: Optional[Dict[str, str]] = None,
    min_consensus_score: Optional[float] = None,
):
    """Add ``consensus_label`` / ``consensus_rank`` / ``consensus_score`` columns.

    Args:
        predictions: DataFrame with cropmodel_label/cropmodel_score and, optionally,
            the hcast_* columns written by :func:`classify_dataframe`.
        species_to_genus: species name -> genus name. See :func:`load_species_to_ranks`.
        species_to_family: species name -> family name.
        min_consensus_score: if set, a row whose joint confidence falls below this
            is demoted one rank (species -> genus -> family -> unresolved). This is
            the "based on confidence" half of the rollup: agreement decides how far
            up the tree we CAN report, confidence decides how far up we MUST.

    Returns:
        A copy of ``predictions`` with the three consensus columns appended.
    """
    if predictions is None or len(predictions) == 0:
        return predictions

    species_to_genus = species_to_genus or {}
    species_to_family = species_to_family or {}

    out = predictions.copy(deep=True)
    labels, ranks, scores = [], [], []
    for _, row in out.iterrows():
        label, rank, score = resolve_row_rank(row, species_to_genus, species_to_family)
        if (
            min_consensus_score is not None
            and rank not in ("unresolved", "verified")
            and score == score  # not NaN
            and score < min_consensus_score
        ):
            label, rank = _demote(label, rank, row, species_to_genus, species_to_family)
        labels.append(label)
        ranks.append(rank)
        scores.append(score)

    out["consensus_label"] = labels
    out["consensus_rank"] = ranks
    out["consensus_score"] = scores
    return out


def _demote(label, rank, row, species_to_genus, species_to_family):
    """Move one step up the tree when joint confidence is below the floor."""
    crop = row.get("cropmodel_label")
    crop = None if (crop is None or pd.isna(crop)) else str(crop)
    if rank == "species":
        genus = _genus_of(crop, species_to_genus)
        return (genus, "genus") if genus else (label, "unresolved")
    if rank == "genus":
        family = _family_of(crop, species_to_family)
        return (family, "family") if family else (label, "unresolved")
    return label, "unresolved"


def summarize_taxonomic_rollup(predictions):
    """Counts per (rank, label) for the consensus columns, finest rank first.

    Returns a DataFrame: consensus_rank, consensus_label, n_observations,
    n_images, mean_score, min_score, max_score.
    """
    if predictions is None or len(predictions) == 0 or "consensus_rank" not in predictions.columns:
        return pd.DataFrame(
            columns=["consensus_rank", "consensus_label", "n_observations",
                     "n_images", "mean_score", "min_score", "max_score"]
        )

    df = predictions.copy()
    df["consensus_label"] = df["consensus_label"].fillna("(none)")
    grouped = df.groupby(["consensus_rank", "consensus_label"], dropna=False).agg(
        n_observations=("consensus_label", "size"),
        n_images=("image_path", "nunique"),
        mean_score=("consensus_score", "mean"),
        min_score=("consensus_score", "min"),
        max_score=("consensus_score", "max"),
    ).reset_index()

    order = {rank: i for i, rank in enumerate(CONSENSUS_RANKS)}
    grouped["_rank_order"] = grouped["consensus_rank"].map(order).fillna(len(order))
    grouped = grouped.sort_values(
        ["_rank_order", "n_observations"], ascending=[True, False]
    ).drop(columns=["_rank_order"]).reset_index(drop=True)
    return grouped
