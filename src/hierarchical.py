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


def _default_transform(image_size: int = 224):
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
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.label_dict = label_dict
        self.species_numeric_to_label = {v: k for k, v in label_dict.items() if k.startswith("species_")}
        self.genus_numeric_to_label = genus_label_dict or {v: k for k, v in label_dict.items() if k.startswith("genus_")}
        self.family_numeric_to_label = family_label_dict or {v: k for k, v in label_dict.items() if k.startswith("family_")}
        self.image_size = image_size
        self._transform = _default_transform(image_size)
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
) -> HCastWrapper:
    """Load hierarchical DeiT ViT model from checkpoint and return a wrapper ready for inference.

    Args:
        checkpoint_path: Path to the checkpoint file
        label_csv: Path to CSV with columns: species, genus, family (optional "index" = species class index).
        device: Device to load model on. If None, uses 'cuda' if available.
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
        genus_df = df.drop_duplicates(subset=['genus']).reset_index(drop=True)
        for idx, row in genus_df.iterrows():
            label_dict[f"genus_{row['genus']}"] = idx
            genus_label_dict[idx] = f"genus_{row['genus']}"
    if len(nb_classes) > 1:
        for idx in range(nb_classes[1]):
            if idx not in genus_label_dict:
                genus_label_dict[idx] = f"genus_{idx}"

    family_label_dict = {}
    if 'family' in df.columns:
        family_df = df.drop_duplicates(subset=['family']).reset_index(drop=True)
        for idx, row in family_df.iterrows():
            label_dict[f"family_{row['family']}"] = idx
            family_label_dict[idx] = f"family_{row['family']}"
    if len(nb_classes) > 2:
        for idx in range(nb_classes[2]):
            if idx not in family_label_dict:
                family_label_dict[idx] = f"family_{idx}"

    return HCastWrapper(
        model=model, device=device, label_dict=label_dict, image_size=img_size,
        species_to_genus=species_to_genus, species_df=species_df,
        genus_label_dict=genus_label_dict, family_label_dict=family_label_dict,
    )


class InferenceCropDataset(Dataset):
    """Dataset for inference that crops images from bounding boxes in a predictions DataFrame."""

    def __init__(self, predictions: pd.DataFrame, image_dir: str, transform=None):
        self.transform = transform or _default_transform()
        self.image_dir = image_dir
        self.predictions = predictions.reset_index(drop=True)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        row = self.predictions.iloc[index]
        full_path = os.path.join(self.image_dir, row['image_path'])
        with open(full_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        crop = image.crop((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
        return self.transform(crop)


@torch.no_grad()
def classify_dataframe(
    predictions,
    image_dir: str,
    model: HCastWrapper,
    batch_size: int = 64,
    num_workers: int = 2,
):
    """Add crop-level hierarchical classification to predictions DataFrame.

    Adds columns: hcast_species, hcast_genus, hcast_family, hcast_species_score,
    hcast_genus_score, hcast_family_score.
    """
    if predictions is None or len(predictions) == 0:
        return predictions

    transform = _default_transform(image_size=model.image_size)
    ds = InferenceCropDataset(predictions, image_dir, transform=transform)
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
