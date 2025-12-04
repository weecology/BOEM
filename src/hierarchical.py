import os
from typing import Optional, Tuple, List, Dict, Callable

import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model

def _infer_head_sizes_from_checkpoint(ckpt: Dict[str, torch.Tensor]) -> Tuple[int, Optional[int], Optional[int]]:
    species = None
    family = None
    manufacturer = None

    # Common head names used in CAST hierarchical models
    if "head.weight" in ckpt:
        species = ckpt["head.weight"].shape[0]
    # Try other possible keys
    for key in ckpt.keys():
        if key.endswith("family_head.weight") and family is None:
            family = ckpt[key].shape[0]
        if key.endswith("manu_head.weight") or key.endswith("mf_head.weight"):
            manufacturer = ckpt[key].shape[0]

    if species is None:
        # Fall back: find the largest classifier-like matrix
        classifier_like = [v.shape[0] for k, v in ckpt.items() if k.endswith(".weight") and v.ndim == 2]
        species = max(classifier_like) if classifier_like else 1000

    return species, family, manufacturer


def _default_transform(image_size: int = 224):
    """Standard ImageNet normalization transform with resize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])


def _transform_without_resize(image_size: int = 224):
    """Transform with resize and normalization (resize needed for model input size).
    
    Note: Despite the name, we do resize here because the model requires fixed-size inputs.
    The name reflects that we're not doing additional augmentation resizing.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])


def _collate_with_superpixels(batch):
    """Collate function that batches (image, superpixel) tuples."""
    images = []
    superpixels = []
    for item in batch:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            images.append(item[0])
            superpixels.append(item[1])
        else:
            images.append(item[0])
            superpixels.append(item[1])
    
    # Stack images
    images = torch.stack(images)
    
    # Stack superpixels (they should already be tensors)
    superpixels = torch.stack(superpixels)
    
    return images, superpixels


class HCastWrapper:
    """Minimal wrapper to run H-CAST species head for crop classification.

    Exposes a similar surface to DeepForest's CropModel for inference-only use.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, label_dict: Dict[str, int], image_size: int = 224, species_to_genus: Optional[Dict[int, str]] = None, species_df: Optional[pd.DataFrame] = None, genus_label_dict: Optional[Dict[int, str]] = None, family_label_dict: Optional[Dict[int, str]] = None):
        self.model = model.eval().to(device)
        self.device = device
        self.label_dict = label_dict  # maps label string -> index
        # Separate mappings for species, genus, and family (they have overlapping 0-based indices)
        self.species_numeric_to_label = {v: k for k, v in label_dict.items() if k.startswith("species_")}
        self.genus_numeric_to_label = genus_label_dict or {v: k for k, v in label_dict.items() if k.startswith("genus_")}
        self.family_numeric_to_label = family_label_dict or {v: k for k, v in label_dict.items() if k.startswith("family_")}
        self.image_size = image_size
        self._transform = _default_transform(image_size)
        self.species_to_genus = species_to_genus or {}
        self.species_df = species_df
        
        # Check if this is a CAST model (requires superpixel labels)
        # CAST models have 'pool' layers in their state dict
        self.is_cast_model = any('pool' in key for key in model.state_dict().keys())

    def get_transform(self, augment: bool = False):
        return self._transform

    @torch.no_grad()
    def predict_logits(self, batch: torch.Tensor, superpixel_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Predict logits for a batch of images.
        
        Args:
            batch: Image tensor of shape (batch_size, 3, H, W)
            superpixel_labels: Superpixel label tensor of shape (batch_size, H, W).
                             Required for CAST models.
        
        Returns:
            Tuple of logits: (species_logits, family_logits, ...) for hierarchical models,
            or just species_logits for non-hierarchical models
        """
        batch = batch.to(self.device)
        
        if self.is_cast_model:
            if superpixel_labels is None:
                raise ValueError("CAST models require superpixel_labels as input")
            superpixel_labels = superpixel_labels.to(self.device)
            outputs = self.model(batch, superpixel_labels)
        else:
            # Standard DeiT models only need images
            outputs = self.model(batch)
            
        if isinstance(outputs, (list, tuple)):
            # For 3-class hierarchical models: (species, genus, family)
            # Note: model code labels them as (head, family_head, manufacturer_head)
            # but they actually represent (species, genus, family)
            return tuple(outputs)
        else:
            return (outputs,)


def load_hcast_model(
    checkpoint_path, 
    device: Optional[torch.device] = None,
    label_csv: Optional[str] = None
) -> HCastWrapper:
    """Load H-CAST model from checkpoint and return a wrapper ready for inference.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on. If None, uses 'cuda' if available.
        label_csv: Optional path to CSV file with columns: species, genus, family.
                   If None, uses numeric labels (species_0, species_1, etc.)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import cast models to register them with timm (required for create_model)
    from src.hcast.cast_models import cast_deit_hier
    from src.hcast.deit import models_hier
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        model_state_dict = checkpoint
    
    if 'args' in checkpoint:
        args = checkpoint['args']
        # Create model using the saved arguments (most reliable)
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes[0],
            drop_rate=getattr(args, 'drop', 0.0),
            drop_path_rate=getattr(args, 'drop_path', 0.0),
            drop_block_rate=None,
            img_size=getattr(args, 'input_size', 224),
            nb_classes=args.nb_classes
        )
        img_size = getattr(args, 'input_size', 224)
        nb_classes = args.nb_classes
    else:
        # Fallback: try to infer from state dict
        species_classes, family_classes, manu_classes = _infer_head_sizes_from_checkpoint(model_state_dict)
        nb_classes = [c for c in [species_classes, family_classes, manu_classes] if c is not None]
        
        # Default to cast_small if we can't determine
        model_name = 'cast_small'
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=nb_classes[0],
            img_size=224,
            nb_classes=nb_classes,
        )
        img_size = 224

    # Load the state dict with strict=False for robustness
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

    model = model.eval().to(device)

    # Build label dict from CSV if provided, otherwise use numeric labels
    species_df = None
    species_to_genus = {}
    if label_csv and os.path.exists(label_csv):
        df = pd.read_csv(label_csv)
        species_df = df.copy()
        label_dict = {}
        # Build species labels - map species index to species name
        if 'species' in df.columns:
            for idx, row in df.iterrows():
                label_dict[f"species_{row['species']}"] = idx
                # Create species index -> genus mapping
                if 'genus' in df.columns and pd.notna(row['genus']):
                    species_to_genus[idx] = row['genus']
        
        # Build genus labels - map genus index to genus name
        # Note: Model's "family_head" (nb_classes[1] = 30) is actually genus
        genus_label_dict = {}
        if 'genus' in df.columns:
            genus_df = df.drop_duplicates(subset=['genus']).reset_index(drop=True)
            for idx, row in genus_df.iterrows():
                label_dict[f"genus_{row['genus']}"] = idx
                genus_label_dict[idx] = f"genus_{row['genus']}"
        
        # Fill in missing genus indices with numeric fallbacks
        if len(nb_classes) > 1:
            num_genus = nb_classes[1]  # Model's "family_head" is actually genus
            for idx in range(num_genus):
                if idx not in genus_label_dict:
                    genus_label_dict[idx] = f"genus_{idx}"
        
        # Build family labels - map family index to family name
        # Note: Model's "manufacturer_head" (nb_classes[2] = 14) is actually family
        family_label_dict = {}
        if 'family' in df.columns:
            family_df = df.drop_duplicates(subset=['family']).reset_index(drop=True)
            for idx, row in family_df.iterrows():
                label_dict[f"family_{row['family']}"] = idx
                family_label_dict[idx] = f"family_{row['family']}"
        
        # Fill in missing family indices with numeric fallbacks
        if len(nb_classes) > 2:
            num_families = nb_classes[2]  # Model's "manufacturer_head" is actually family
            for idx in range(num_families):
                if idx not in family_label_dict:
                    family_label_dict[idx] = f"family_{idx}"
    else:
        # Fall back to numeric labels
        label_dict = {f"species_{i}": i for i in range(nb_classes[0])}
        if len(nb_classes) > 1:
            start_idx = nb_classes[0]
            for i in range(nb_classes[1]):
                label_dict[f"family_{i}"] = start_idx + i
        if len(nb_classes) > 2:
            start_idx = nb_classes[0] + nb_classes[1]
            for i in range(nb_classes[2]):
                label_dict[f"manufacturer_{i}"] = start_idx + i

    # Create fallback label dicts if not provided
    if not genus_label_dict and len(nb_classes) > 1:
        genus_label_dict = {i: f"genus_{i}" for i in range(nb_classes[1])}
    if not family_label_dict and len(nb_classes) > 2:
        family_label_dict = {i: f"family_{i}" for i in range(nb_classes[2])}
    
    return HCastWrapper(model=model, device=device, label_dict=label_dict, image_size=img_size, species_to_genus=species_to_genus, species_df=species_df, genus_label_dict=genus_label_dict, family_label_dict=family_label_dict)

class USGSDataset(Dataset):
    def __init__(self, 
                 predictions: pd.DataFrame,
                 image_dir: str,
                 transform=None,
                 mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
                 std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor: float = 1.0):
        """Dataset for inference that crops images from bounding boxes in predictions DataFrame.
        
        Args:
            predictions: DataFrame with columns: image_path, xmin, ymin, xmax, ymax
            image_dir: Root directory where images are located
            transform: Optional transform to apply (should not resize for superpixel generation)
            mean: Mean for normalization
            std: Std for normalization
            n_segments: Number of superpixels to generate
            compactness: Superpixel compactness parameter
            blur_ops: Optional blur operation
            scale_factor: Scale factor for superpixel generation
        """
        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_dir = image_dir
        
        # Store predictions DataFrame
        self.predictions = predictions.reset_index(drop=True)

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        """Get a cropped image and its superpixel labels for inference.
        
        Returns:
            Tuple of (image_tensor, superpixel_labels)
        """
        row = self.predictions.iloc[index]
        image_path = row['image_path']
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        
        # Build full path to image
        full_path = os.path.join(self.image_dir, image_path)
        
        # Load and crop image
        with open(full_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        
        # Crop based on bounding box
        crop = image.crop((xmin, ymin, xmax, ymax))
        
        # Apply transform if provided (should not resize)
        if self.transform is not None:
            sample = self.transform(crop)
        else:
            # Default: just convert to tensor and normalize
            sample = transforms.ToTensor()(crop)
            normalize = transforms.Normalize(self.mean, self.std)
            sample = normalize(sample)

        # Generate superpixels
        if self.blur_ops is not None:
            samp = self.blur_ops(sample)
        else:
            samp = sample
        
        # Convert tensor back to numpy for superpixel generation
        samp = (samp.data.numpy().transpose(1, 2, 0) * np.array(self.std) + np.array(self.mean))
        samp = (samp * 255).astype(np.uint8)
        samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
        
        # Generate superpixels
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            samp.shape[1], samp.shape[0], 3, 
            num_superpixels=self.n_segments, 
            num_levels=1, 
            prior=2,
            histogram_bins=5, 
            double_step=False
        )
        seeds.iterate(samp, num_iterations=15)
        segments = seeds.getLabels()
        segments = torch.LongTensor(segments)

        return sample, segments


@torch.no_grad()
def classify_dataframe(
    predictions,
    image_dir: str,
    model: HCastWrapper,
    batch_size: int = 64,
    num_workers: int = 2,
):
    """Add crop-level hierarchical classification to predictions DataFrame.

    Adds columns: hcast_species, hcast_genus, hcast_family, hcast_species_score, hcast_family_score.
    """
    if predictions is None or len(predictions) == 0:
        return predictions

    # Create transform with resize to model's expected input size
    transform = _transform_without_resize(image_size=model.image_size)

    # Use USGSDataset to create a DataLoader
    ds = USGSDataset(predictions, image_dir, transform=transform)

    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=_collate_with_superpixels
    )

    all_species_idx: List[int] = []
    all_species_prob: List[float] = []
    all_genus_idx: List[int] = []
    all_genus_prob: List[float] = []
    all_family_idx: List[int] = []
    all_family_prob: List[float] = []

    for batch_images, batch_superpixels in dl:
        logits_tuple = model.predict_logits(batch_images, batch_superpixels)
        
        # Extract species predictions (first output)
        species_logits = logits_tuple[0]
        species_probs = torch.softmax(species_logits, dim=1)
        species_conf, species_idx = torch.max(species_probs, dim=1)
        all_species_idx.extend(species_idx.cpu().tolist())
        all_species_prob.extend(species_conf.cpu().tolist())
        
        # Extract genus predictions if available (second output)
        # Note: model calls this "family_head" but it's actually genus
        if len(logits_tuple) > 1:
            genus_logits = logits_tuple[1]
            genus_probs = torch.softmax(genus_logits, dim=1)
            genus_conf, genus_idx = torch.max(genus_probs, dim=1)
            all_genus_idx.extend(genus_idx.cpu().tolist())
            all_genus_prob.extend(genus_conf.cpu().tolist())
        else:
            all_genus_idx.extend([None] * len(species_idx))
            all_genus_prob.extend([None] * len(species_idx))
        
        # Extract family predictions if available (third output)
        # Note: model calls this "manufacturer_head" but it's actually family
        if len(logits_tuple) > 2:
            family_logits = logits_tuple[2]
            family_probs = torch.softmax(family_logits, dim=1)
            family_conf, family_idx = torch.max(family_probs, dim=1)
            all_family_idx.extend(family_idx.cpu().tolist())
            all_family_prob.extend(family_conf.cpu().tolist())
        else:
            all_family_idx.extend([None] * len(species_idx))
            all_family_prob.extend([None] * len(species_idx))

    # Map species indices to labels
    species_labels = []
    for idx in all_species_idx:
        label_key = model.species_numeric_to_label.get(idx, f"species_{idx}")
        # Extract species name from label_key (format: "species_<name>")
        if label_key.startswith("species_"):
            species_name = label_key.replace("species_", "")
            species_labels.append(species_name)
        else:
            species_labels.append(label_key)
    
    # Map genus indices to labels (from model's second output - "family_head" is actually genus)
    genus_labels = []
    for idx in all_genus_idx:
        if idx is None:
            genus_labels.append(None)
        else:
            label_key = model.genus_numeric_to_label.get(idx, f"genus_{idx}")
            # Extract genus name from label_key (format: "genus_<name>")
            if label_key.startswith("genus_"):
                genus_name = label_key.replace("genus_", "")
                genus_labels.append(genus_name)
            else:
                genus_labels.append(label_key)
    
    # Map family indices to labels (from model's third output - "manufacturer_head" is actually family)
    family_labels = []
    for idx in all_family_idx:
        if idx is None:
            family_labels.append(None)
        else:
            label_key = model.family_numeric_to_label.get(idx, f"family_{idx}")
            # Extract family name from label_key (format: "family_<name>")
            if label_key.startswith("family_"):
                family_name = label_key.replace("family_", "")
                family_labels.append(family_name)
            else:
                family_labels.append(label_key)

    predictions = predictions.copy(deep=True)
    predictions["hcast_species"] = species_labels
    predictions["hcast_genus"] = genus_labels
    predictions["hcast_family"] = family_labels
    predictions["hcast_species_score"] = all_species_prob
    predictions["hcast_genus_score"] = all_genus_prob
    predictions["hcast_family_score"] = all_family_prob
    return predictions


def infer_head_sizes_from_checkpoint(checkpoint_path: str) -> List[int]:
    """Return list of output sizes for each head found in checkpoint state_dict.
    Looks for weight tensors named like '*head*weight' (robust heuristic).
    """
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
    # fallback: if nothing found, try final classifier or heads list
    if not sizes:
        for k, v in state.items():
            if v.ndim == 2 and ("classifier" in k or "fc" in k or "head" in k):
                sizes.append(v.shape[0])
    return sizes







