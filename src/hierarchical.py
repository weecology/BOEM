import os
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.hcast.cast_models import cast_deit_hier  
import pandas as pd

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


def load_hcast_model(checkpoint_path, device: Optional[torch.device] = None) -> HCastWrapper:
    """Load H-CAST model from checkpoint and return a wrapper ready for inference.

    If checkpoint_path is None, the most recent best_checkpoint.pth under tamu_hcast/ is used.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, weights_only=False)
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

    model.load_state_dict(ckpt, strict=True)

    # Build a placeholder label dict if none provided. Users can override externally.
    # Read species labels from CSV

    df = pd.read_csv(species_csv)
    species_label_dict = {row['species']: idx for idx, row in df.iterrows()}
    genus_label_dict = {row['genus']: idx for idx, row in df(subset=['genus']).iterrows()}
    family_label_dict = {row['family']: idx for idx, row in df(subset=['family']).iterrows()}

    # If no CSV provided, fall back to numeric labels
    label_dict = {**species_label_dict, **genus_label_dict, **family_label_dict}

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


def load_hierarchical_label_table(csv_path: str, index_col: str = None) -> pd.DataFrame:
    """Load labels CSV exported from your spreadsheet.
    Expect columns like: species, genus, family and an optional numeric index column.
    If index_col is None the dataframe index will be used as numeric class index.
    """
    df = pd.read_csv(csv_path)
    if index_col:
        df = df.set_index(index_col)
    return df


def build_label_maps(df: pd.DataFrame, head_cols: List[str]) -> List[Dict[int, str]]:
    """Given label table and head column names (e.g. ['species','genus','family']),
    return list of mappings numeric_index -> label for each head in same order.
    """
    maps = []
    for col in head_cols:
        if col not in df.columns:
            raise KeyError(f"Column {col} not in label table")
        mapping = {int(idx): str(lbl) for idx, lbl in df[col].items()}
        maps.append(mapping)
    return maps


def split_logits_and_map(logits: torch.Tensor, head_sizes: List[int], maps: List[Dict[int,str]]) -> Tuple[List[int], List[str]]:
    """Split model logits per head, return numeric predictions and mapped labels per head.
    logits: (N, sum(head_sizes)) or list of per-head arrays.
    returns per-head predicted indices and per-head labels (each list length = num_heads).
    """
    if isinstance(logits, (list, tuple)):
        per_head = logits
    else:
        # split along last dim
        splits = torch.split(logits, head_sizes, dim=-1)
        per_head = splits
    numeric_preds = []
    label_preds = []
    for h_out, mapping in zip(per_head, maps):
        # h_out shape (N, C_h)
        pred_idx = torch.argmax(h_out, dim=-1).cpu().numpy()
        numeric_preds.append(pred_idx)
        # map each numeric to string label
        label_preds.append([mapping.get(int(i), f"UNK_{i}") for i in pred_idx])
    return numeric_preds, label_preds


