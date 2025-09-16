import os
import sys
import glob
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_transform(image_size: int = 224):
    try:
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for hierarchical classification inference") from e

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _ensure_hcast_on_path(repo_root: str):
    base_dir = os.path.join(repo_root, "tamu_hcast")
    deit_dir = os.path.join(base_dir, "deit")
    # Add both base (for cast_models) and deit (for dataset/utils) to path
    for p in (base_dir, deit_dir):
        if p not in sys.path:
            sys.path.append(p)
    # Register the CAST hierarchical model with timm via side-effect import
    import importlib
    importlib.import_module("cast_models.cast_deit_hier")


def find_hcast_checkpoint(repo_root: str) -> Optional[str]:
    """Find a best_checkpoint.pth under tamu_hcast/ by recency.

    Returns the most recently modified path if found, else None.
    """
    candidates: List[str] = []
    base = os.path.join(repo_root, "tamu_hcast")
    for pattern in [
        os.path.join(base, "best_checkpoint.pth"),
        os.path.join(base, "output", "**", "best_checkpoint.pth"),
    ]:
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


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


class HCastWrapper:
    """Minimal wrapper to run H-CAST species head for crop classification.

    Exposes a similar surface to DeepForest's CropModel for inference-only use.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, label_dict: Dict[str, int], image_size: int = 224):
        self.model = model.eval().to(device)
        self.device = device
        self.label_dict = label_dict  # maps label string -> index
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}
        self.image_size = image_size
        self._transform = _default_transform(image_size)

    def get_transform(self, augment: bool = False):
        return self._transform

    @torch.no_grad()
    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(batch.to(self.device))
        if isinstance(outputs, (list, tuple)):
            species_logits = outputs[0]
        else:
            species_logits = outputs
        return species_logits


def load_hcast_model(repo_root: str, checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None) -> HCastWrapper:
    """Load H-CAST model from checkpoint and return a wrapper ready for inference.

    If checkpoint_path is None, the most recent best_checkpoint.pth under tamu_hcast/ is used.
    """
    repo_root = os.path.abspath(repo_root)
    _ensure_hcast_on_path(repo_root)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = find_hcast_checkpoint(repo_root)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No H-CAST checkpoint found. Place best_checkpoint.pth under tamu_hcast/ or tamu_hcast/output/**/")

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        ckpt = state["model"]
    elif "state_dict" in state:
        ckpt = state["state_dict"]
    else:
        ckpt = state

    species_classes, family_classes, manu_classes = _infer_head_sizes_from_checkpoint(ckpt)

    from timm.models import create_model
    model = create_model(
        "cast_small",
        pretrained=False,
        num_classes=species_classes,
        img_size=224,
        nb_classes=[c for c in [species_classes, family_classes, manu_classes] if c is not None],
    )

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if len(unexpected) > 0:
        # keys from optimizer/ema or extra heads are fine
        pass

    # Build a placeholder label dict if none provided. Users can override externally.
    label_dict = {f"species_{i}": i for i in range(species_classes)}

    return HCastWrapper(model=model, device=device, label_dict=label_dict, image_size=224)


class _CropDataset(Dataset):
    def __init__(self, predictions, root_dir: str, transform):
        self.predictions = predictions.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        row = self.predictions.iloc[idx]
        path = os.path.join(self.root_dir, row["image_path"])
        with Image.open(path) as img:
            img = img.convert("RGB")
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            crop = img.crop((xmin, ymin, xmax, ymax))
            crop = self.transform(crop)
        return crop


@torch.no_grad()
def classify_dataframe(
    predictions,
    image_dir: str,
    model: HCastWrapper,
    batch_size: int = 64,
    num_workers: int = 2,
):
    """Add crop-level hierarchical classification to predictions DataFrame in-place.

    Adds/overwrites columns: cropmodel_label, cropmodel_score.
    """
    if predictions is None or len(predictions) == 0:
        return predictions

    ds = _CropDataset(predictions, root_dir=image_dir, transform=model.get_transform(augment=False))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    all_top_idx: List[int] = []
    all_top_prob: List[float] = []

    for batch in dl:
        logits = model.predict_logits(batch)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        all_top_idx.extend(idx.cpu().tolist())
        all_top_prob.extend(conf.cpu().tolist())

    idx_to_label = model.numeric_to_label_dict
    labels = [idx_to_label.get(i, f"species_{i}") for i in all_top_idx]

    predictions = predictions.copy(deep=True)
    predictions["cropmodel_label"] = labels
    predictions["cropmodel_score"] = all_top_prob
    return predictions


