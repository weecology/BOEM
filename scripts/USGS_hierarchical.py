"""Train hierarchical (family, genus, species) ViT on UBFAI crop annotations.

Uses the same data pipeline as USGS_classification.py (crop annotations, filter,
balance, split by parent image) but trains the DeiT hierarchical transformer
from src.hcast (family/genus/species heads) instead of CropModel. Labels are
mapped via taxonomy.json to family, genus, and species IDs.

For direct comparability with CropModel (same test/val data), run USGS_classification
first; it writes usgs_train_split.csv and usgs_val_split.csv under
checkpoint_dir/buffer_<N>/<comet_id>/. Then run this script with:

  uv run python scripts/USGS_hierarchical.py \\
    --train-split-csv /path/to/.../usgs_train_split.csv \\
    --val-split-csv /path/to/.../usgs_val_split.csv \\
    --image-dir /path/to/crops \\
    --taxonomy taxonomy.json \\
    --output-dir output/hier

Val (test) set is always taken from the CropModel val CSV unchanged. Train set is
the CropModel train CSV; if --annotations-dir is also provided, train is extended
with any annotation rows whose label is a higher taxon (family or genus) above
those species in the taxonomy (e.g. labels like Delphinidae when Tursiops truncatus
is in the species set). No extra row may come from a parent image that is in the
val set (no train/val leak).

Other options: --train-crop-dir/--val-crop-dir (crop dirs) or
--annotations-dir + --image-dir (recompute split from annotations, no CropModel CSVs).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, accuracy
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.USGS_classification import (
    _crop_path_to_parent_stem,
    gentle_class_balance,
    normalize_label,
    train_test_split_by_image,
)
from scripts.taxonomy_hier import load_taxonomy_restricted_to_species
from src.classification import TURTLE_CLASS
from src.hierarchical import expand_bbox_to_square as _expand_bbox_to_square


class HierarchicalCropDataset(Dataset):
    """Dataset that returns (image_tensor, species_id, genus_id, family_id).

    Supports two modes:
    - From crop dirs: csv_or_dirs is (train_crop_dir, val_crop_dir).
    - From annotations: csv_or_dirs is (train_df, val_df) with image_path, xmin, ymin, xmax, ymax, label.
    """

    def __init__(
        self,
        csv_or_dirs,
        name_to_ids: dict[str, tuple[int, int, int]],
        transform=None,
        is_train: bool = True,
        image_dir: str | None = None,
        expand_pixels: int = 0,
        square: bool = True,
    ):
        self.transform = transform
        self.name_to_ids = name_to_ids
        self.image_dir = image_dir
        self.expand_pixels = expand_pixels
        self.square = square
        self.from_dirs = image_dir is None
        if self.from_dirs:
            crop_dir = csv_or_dirs[0] if is_train else csv_or_dirs[1]
            self.samples = []
            for label_dir in Path(crop_dir).iterdir():
                if not label_dir.is_dir():
                    continue
                label_name = label_dir.name
                if label_name not in name_to_ids:
                    continue
                for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG"):
                    for p in label_dir.glob(ext):
                        self.samples.append((str(p), label_name))
        else:
            df = csv_or_dirs[0] if is_train else csv_or_dirs[1]
            df = df[df["label"].isin(name_to_ids)].copy()
            self.samples = [
                (row["image_path"], row["label"],
                 (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])))
                for _, row in df.iterrows()
            ]
        self.targets = [name_to_ids[s[1]][2] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        if self.from_dirs:
            path, label = s[0], s[1]
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        else:
            image_path, label, (xmin, ymin, xmax, ymax) = s[0], s[1], s[2]
            full_path = os.path.join(self.image_dir or "", image_path)
            with open(full_path, "rb") as f:
                img = Image.open(f).convert("RGB")
            w, h = img.size
            xmin = max(0, xmin - self.expand_pixels)
            ymin = max(0, ymin - self.expand_pixels)
            xmax = min(w, xmax + self.expand_pixels)
            ymax = min(h, ymax + self.expand_pixels)
            if self.square:
                box = _expand_bbox_to_square(xmin, ymin, xmax, ymax, w, h)
            else:
                box = (int(xmin), int(ymin), int(xmax), int(ymax))
            img = img.crop(box)
        if self.transform is not None:
            img = self.transform(img)
        fid, gid, sid = self.name_to_ids[label]
        return img, sid, gid, fid


def build_transform(is_train: bool, input_size: int = 224, eval_crop_ratio: float | None = 0.875):
    if is_train:
        return create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
    from torchvision import transforms
    if not eval_crop_ratio:
        # Squash the whole crop to input_size. Correct companion to --no-square:
        # centre-cropping a non-square crop would throw away the long axis, which
        # is exactly the content that made the box non-square.
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    size = int(input_size / eval_crop_ratio)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, mixup_fn, args):
    """Train one epoch with species + genus + family losses."""
    model.train(True)
    total_loss = 0.0
    n_batches = 0
    for batch in data_loader:
        samples, targets, genus_targets, family_targets = [x.to(device, non_blocking=True) for x in batch]
        if mixup_fn is not None:
            samples, targets, genus_targets, family_targets = mixup_fn(samples, [targets, genus_targets, family_targets])
        with torch.cuda.amp.autocast():
            outputs, genus_out, family_out = model(samples)
            loss_species = criterion(outputs, targets)
            loss_genus = criterion(genus_out, genus_targets)
            loss_family = criterion(family_out, family_targets)
            loss = loss_species + loss_genus + loss_family
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch} train loss: {avg_loss:.4f}")
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Evaluate with species, genus, and family accuracy."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    species_acc1_sum, species_acc5_sum = 0.0, 0.0
    genus_acc1_sum, family_acc1_sum = 0.0, 0.0
    sp_loss_sum, genus_loss_sum, family_loss_sum = 0.0, 0.0, 0.0
    n = 0

    for images, target, genus_targets, family_targets in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        genus_targets = genus_targets.to(device, non_blocking=True)
        family_targets = family_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, genus_out, family_out = model(images)
            sp_loss_sum += criterion(output, target).item() * images.size(0)
            genus_loss_sum += criterion(genus_out, genus_targets).item() * images.size(0)
            family_loss_sum += criterion(family_out, family_targets).item() * images.size(0)

        a1, a5 = accuracy(output, target, topk=(1, 5))
        genus_a1, _ = accuracy(genus_out, genus_targets, topk=(1, 5))
        family_a1, _ = accuracy(family_out, family_targets, topk=(1, 5))

        bs = images.size(0)
        species_acc1_sum += a1.item() * bs
        species_acc5_sum += a5.item() * bs
        genus_acc1_sum += genus_a1.item() * bs
        family_acc1_sum += family_a1.item() * bs
        n += bs

    return {
        "acc1": species_acc1_sum / n if n else 0,
        "acc5": species_acc5_sum / n if n else 0,
        "genus_acc1": genus_acc1_sum / n if n else 0,
        "family_acc1": family_acc1_sum / n if n else 0,
        "sploss": sp_loss_sum / n if n else 0,
        "genus_loss": genus_loss_sum / n if n else 0,
        "family_loss": family_loss_sum / n if n else 0,
    }


def _load_annotation_pool_preprocessed(annotations_dir: str) -> pd.DataFrame:
    """Load and preprocess annotation CSVs to match USGS_classification (before balance/split)."""
    crop_annotations = glob.glob(os.path.join(annotations_dir, "*.csv"))
    df = pd.concat([pd.read_csv(x) for x in crop_annotations])
    df = df.groupby("label").filter(lambda x: len(x) > 25)
    df = df[df["label"].str.contains(" ", na=False)]
    df = df[~df["label"].isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"])]
    df["label"] = df["label"].apply(normalize_label)
    df["label"] = df["label"].apply(lambda x: " ".join(str(x).split()[:2]))
    df = df[df["label"].apply(lambda x: len(str(x).split()) == 2)]
    df = df[(df["xmin"] != 0) & (df["ymin"] != 0) & (df["xmax"] != 0) & (df["ymax"] != 0)]
    df = df[(df["xmin"] >= 0) & (df["ymin"] >= 0) & (df["xmax"] >= 0) & (df["ymax"] >= 0)]
    return df


def _build_hierarchical_train_with_ancestors(
    args,
    crop_model_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    name_to_ids: dict[str, tuple[int, int, int]],
) -> pd.DataFrame:
    """Extend CropModel train with annotation rows whose label is a higher taxon (family/genus)."""
    pool = _load_annotation_pool_preprocessed(args.annotations_dir)
    pool = pool[pool["label"].isin(name_to_ids)].copy()
    pool["parent_image"] = pool["image_path"].map(_crop_path_to_parent_stem)
    val_parents = set(val_df["image_path"].map(_crop_path_to_parent_stem))
    extra = pool[~pool["parent_image"].isin(val_parents)].drop(columns=["parent_image"])
    train_df = pd.concat([crop_model_train_df, extra], ignore_index=True)
    train_df = train_df.drop_duplicates(
        subset=["image_path", "xmin", "ymin", "xmax", "ymax"], keep="first"
    )
    n_extra = len(train_df) - len(crop_model_train_df)
    if n_extra > 0:
        print(f"[hierarchical train] added {n_extra} rows with higher-taxon labels (family/genus)")
    return train_df


def _build_datasets(args, nb_classes, name_to_ids):
    """Build train/val datasets from the parsed arguments."""
    if args.train_crop_dir:
        csv_or_dirs = (args.train_crop_dir, args.val_crop_dir)
        kw = {}
    elif args.train_split_csv and args.val_split_csv:
        val_df = pd.read_csv(args.val_split_csv)
        val_df = val_df[val_df["label"].isin(name_to_ids)]
        if val_df.empty:
            raise ValueError("No rows left in val after filtering to taxonomy labels")
        crop_model_train_df = pd.read_csv(args.train_split_csv)
        crop_model_train_df = crop_model_train_df[
            crop_model_train_df["label"].isin(name_to_ids)
        ]
        if crop_model_train_df.empty:
            raise ValueError("No rows left in train after filtering to taxonomy labels")
        if args.annotations_dir:
            train_df = _build_hierarchical_train_with_ancestors(
                args, crop_model_train_df, val_df, name_to_ids
            )
            print(
                f"Using CropModel val (unchanged), train extended with higher-taxon labels: "
                f"train {len(train_df)}, val {len(val_df)}"
            )
        else:
            train_df = crop_model_train_df
            print(f"Using CropModel split: train {len(train_df)}, val {len(val_df)}")
        csv_or_dirs = (train_df, val_df)
        kw = {"image_dir": args.image_dir, "expand_pixels": args.expand_pixels}
    else:
        crop_annotations = glob.glob(os.path.join(args.annotations_dir, "*.csv"))
        df = pd.concat([pd.read_csv(x) for x in crop_annotations])
        df = df[df["label"].str.contains(" ", na=False)]
        df = df[~df["label"].isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"])]
        df["label"] = df["label"].apply(normalize_label)
        df = df[df["label"].apply(lambda x: len(str(x).split()) == 2)]
        df = df[(df["xmin"] != 0) & (df["ymin"] != 0) & (df["xmax"] != 0) & (df["ymax"] != 0)]
        df = df[(df["xmin"] >= 0) & (df["ymin"] >= 0) & (df["xmax"] >= 0) & (df["ymax"] >= 0)]
        df = df.groupby("label").filter(lambda x: len(x) > 25)
        df = df[df["label"].isin(name_to_ids)]
        df = gentle_class_balance(df, factor=args.class_balance_factor, random_state=args.seed)
        train_df, val_df, report = train_test_split_by_image(df, test_size=0.1, min_test_images=args.min_test_images, random_state=args.seed)
        print(f"Split: kept {report['n_classes_kept']} classes, dropped {report['n_dropped_classes']}")
        csv_or_dirs = (train_df, val_df)
        kw = {"image_dir": args.image_dir, "expand_pixels": args.expand_pixels}

    train_ds = HierarchicalCropDataset(csv_or_dirs, name_to_ids, transform=build_transform(True, args.input_size), is_train=True, **kw)
    val_ds = HierarchicalCropDataset(csv_or_dirs, name_to_ids, transform=build_transform(False, args.input_size), is_train=False, **kw)
    return train_ds, val_ds


def _discover_labels(args):
    """Discover unique species labels from the data source."""
    if args.train_crop_dir:
        unique_labels = set()
        for d in (Path(args.train_crop_dir), Path(args.val_crop_dir)):
            if d.exists():
                for sub in d.iterdir():
                    if sub.is_dir():
                        unique_labels.add(sub.name)
        return unique_labels
    elif args.train_split_csv and args.val_split_csv:
        train_df = pd.read_csv(args.train_split_csv)
        val_df = pd.read_csv(args.val_split_csv)
        return set(train_df["label"].unique()) | set(val_df["label"].unique())
    else:
        crop_annotations = glob.glob(os.path.join(args.annotations_dir, "*.csv"))
        df = pd.concat([pd.read_csv(x) for x in crop_annotations])
        df["label"] = df["label"].apply(normalize_label)
        return set(df["label"].unique())


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical (family, genus, species) ViT")
    parser.add_argument("--taxonomy", default="taxonomy.json")
    parser.add_argument("--output-dir", default="output/hier")
    parser.add_argument("--train-crop-dir")
    parser.add_argument("--val-crop-dir")
    parser.add_argument("--train-split-csv")
    parser.add_argument("--val-split-csv")
    parser.add_argument("--annotations-dir")
    parser.add_argument("--image-dir")
    parser.add_argument("--expand-pixels", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--min-test-images", type=int, default=5)
    parser.add_argument("--class-balance-factor", type=float, default=3.0)
    args = parser.parse_args()

    if args.train_crop_dir:
        if not args.val_crop_dir:
            parser.error("--val-crop-dir required when using --train-crop-dir")
    elif args.train_split_csv or args.val_split_csv:
        if not args.train_split_csv or not args.val_split_csv or not args.image_dir:
            parser.error("--train-split-csv, --val-split-csv, and --image-dir required")
    else:
        if not args.annotations_dir or not args.image_dir:
            parser.error("Use (--train-crop-dir + --val-crop-dir), (--train-split-csv + --val-split-csv + --image-dir), or (--annotations-dir + --image-dir)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_labels = _discover_labels(args)
    use_ancestor_labels = bool(
        getattr(args, "annotations_dir", None)
        and args.train_split_csv
        and args.val_split_csv
    )
    nb_classes, name_to_ids = load_taxonomy_restricted_to_species(
        args.taxonomy, unique_labels, include_ancestor_labels=use_ancestor_labels
    )
    if TURTLE_CLASS in unique_labels and TURTLE_CLASS not in name_to_ids:
        # Sea turtles are annotated at family/order rank and collapsed by the
        # CropModel pipeline (src/classification.py) onto one synthetic
        # two-word class that has no scientificName in taxonomy.json, so the
        # taxonomy walk above never sees it. Give it its own id at every
        # level rather than silently dropping every turtle crop.
        sid, gid, fid = nb_classes[0], nb_classes[1], nb_classes[2]
        name_to_ids[TURTLE_CLASS] = (fid, gid, sid)
        nb_classes = [nb_classes[0] + 1, nb_classes[1] + 1, nb_classes[2] + 1]
        print(f"[hierarchical] added synthetic taxonomy entry for {TURTLE_CLASS!r} (species_id={sid})")
    if not name_to_ids:
        print("No labels found in taxonomy; check taxonomy.json and data")
        return 1
    print(f"Hierarchy sizes: species={nb_classes[0]} genus={nb_classes[1]} family={nb_classes[2]}")

    train_ds, val_ds = _build_datasets(args, nb_classes, name_to_ids)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    import src.hcast.deit.models_hier  # noqa: F401
    from timm.models import create_model

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=nb_classes[0],
        drop_rate=0.0,
        drop_path_rate=0.1,
        img_size=args.input_size,
        nb_classes=nb_classes,
    )
    model.to(device)

    class OptArgs:
        opt = "adamw"
        opt_eps = 1e-8
        opt_betas = None
        weight_decay = 0.05
        lr = args.lr
        momentum = 0.9
    optimizer = create_optimizer(OptArgs(), model)
    loss_scaler = NativeScaler()
    sched_args = type("SchedArgs", (), {
        "sched": "cosine", "lr": args.lr, "warmup_epochs": 5, "epochs": args.epochs,
        "min_lr": 1e-5, "warmup_lr": 1e-6, "cooldown_epochs": 10, "decay_rate": 0.1,
    })()
    lr_scheduler, _ = create_scheduler(sched_args, optimizer)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        from src.hcast.deit.mixup_hier import Mixup
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=1.0,
            switch_prob=0.5, mode="batch", label_smoothing=args.smoothing,
            num_classes=nb_classes,
        )

    class TrainArgs:
        clip_grad = None
    train_args = TrainArgs()

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler, mixup_fn, train_args)
        lr_scheduler.step(epoch)
        stats = evaluate(val_loader, model, device)
        print(
            f"Epoch {epoch} val: Species@1 {stats['acc1']:.2f}  Species@5 {stats['acc5']:.2f}  "
            f"Genus@1 {stats['genus_acc1']:.2f}  Family@1 {stats['family_acc1']:.2f}"
        )
        if stats["acc1"] > best_acc:
            best_acc = stats["acc1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "accuracy": best_acc,
                "nb_classes": nb_classes,
                "stats": stats,
            }, os.path.join(args.output_dir, "best_checkpoint.pth"))
    print(f"Best Species@1: {best_acc:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
