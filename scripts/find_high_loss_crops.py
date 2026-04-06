#!/usr/bin/env python3
"""Find classification crops with highest loss and upload to Label Studio for review.

Uses DeepForest CropModel on training crop data: per-crop cross-entropy loss.
Uploads the parent image (patch) for each high-loss crop to Label Studio with
detections overlaid (same UI as pipeline: RectangleLabels + Taxonomy dropdown).
Strategy: top crops by loss -> unique parent stems -> parent patch image +
detections for that image -> upload parent with bbox/taxonomy preannotations.
Config: boem_conf/classification_model/finetune.yaml.
"""

import json
import os
import re
import sys
import argparse
import tempfile
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepforest.model import CropModel
from src import label_studio as ls_mod

# Paths to resolve parent patch and annotation CSVs (match investigate_duplicate_crops / prepare_USGS).
UBFAI_CROPS = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"

# Crop filename like C1_L6_F560_T20241219_173703_737_23.png -> parent stem C1_L6_F560_T20241219_173703_737
_CROP_SUFFIX_RE = re.compile(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$")


def crop_path_to_parent_stem(crop_path: str) -> str:
    """From crop filename return parent stem; if no match, return basename without ext."""
    basename = os.path.basename(str(crop_path))
    m = _CROP_SUFFIX_RE.match(basename)
    if m is None:
        return Path(basename).stem
    return m.group(1)


def find_parent_patch(parent_stem: str) -> Path | None:
    """Locate parent patch image (parent_stem.png) from which classification crops were cut."""
    for base in (UBFAI_CROPS, DETECTION_CROPS_BASE):
        if not os.path.isdir(base):
            continue
        p = Path(base) / f"{parent_stem}.png"
        if p.exists():
            return p
        if base == DETECTION_CROPS_BASE:
            for flight_dir in Path(base).iterdir():
                if not flight_dir.is_dir():
                    continue
                q = flight_dir / f"{parent_stem}.png"
                if q.exists():
                    return q
    return None


def find_detections_for_patch(parent_stem: str) -> pd.DataFrame | None:
    """Load annotation rows for image_path == parent_stem.png. Returns DataFrame with xmin, ymin, xmax, ymax, label, cropmodel_label, score."""
    csv_dir = Path(UBFAI_CROPS)
    if not csv_dir.exists():
        return None
    target = f"{parent_stem}.png"
    tried = set()
    for csv_path in [csv_dir / "train.csv", *sorted(csv_dir.glob("*.csv"))]:
        if csv_path in tried or not csv_path.exists():
            continue
        tried.add(csv_path)
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if "image_path" not in df.columns:
            continue
        df["image_path"] = df["image_path"].astype(str).apply(lambda p: os.path.basename(p))
        patch_rows = df[df["image_path"] == target].copy()
        if patch_rows.empty:
            continue
        if "left" in patch_rows.columns and "xmin" not in patch_rows.columns:
            patch_rows["xmin"] = patch_rows["left"]
            patch_rows["ymin"] = patch_rows["top"]
            patch_rows["xmax"] = patch_rows["left"] + patch_rows["width"]
            patch_rows["ymax"] = patch_rows["top"] + patch_rows["height"]
        for col in ("xmin", "ymin", "xmax", "ymax"):
            if col not in patch_rows.columns:
                return None
        species_col = patch_rows.get("cropmodel_label", patch_rows["label"])
        patch_rows["cropmodel_label"] = species_col
        patch_rows["label"] = "Object"
        patch_rows["score"] = patch_rows.get("score", 1.0)
        patch_rows["comet_id"] = patch_rows.get("comet_id", "high_loss_review")
        return patch_rows[["image_path", "xmin", "ymin", "xmax", "ymax", "label", "cropmodel_label", "score", "comet_id"]]
    return None


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_crops_from_imagefolder(crop_root):
    """List of (crop_path, label) from crop_root/ClassName/*.png."""
    pairs = []
    if not crop_root.exists():
        return pairs
    for class_dir in sorted(crop_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG"):
            for img_path in class_dir.glob(ext):
                pairs.append((img_path, label))
    return pairs


def find_crop_dirs(config):
    """Resolve train crop dir(s). Handles buffer_30/comet_id/ClassName or flat ClassName layout."""
    train_dir = config.get("checkpoint_train_dir") or config.get("train_crop_image_dir")
    if not train_dir:
        return []
    root = Path(train_dir)
    if not root.exists():
        return []
    if discover_crops_from_imagefolder(root):
        return [root]
    nested = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        pairs = discover_crops_from_imagefolder(d)
        if pairs:
            return [d]
        for sub in d.iterdir():
            if sub.is_dir() and discover_crops_from_imagefolder(sub):
                nested.append(sub)
    return nested if nested else [root]


def compute_loss_for_crop(model, crop_path, label, device):
    """Return (loss, pred_label) or None."""
    if label not in model.label_dict:
        return None
    true_idx = model.label_dict[label]
    try:
        img = Image.open(crop_path).convert("RGB")
    except Exception:
        return None
    img = np.array(img)
    if img.ndim != 3:
        return None
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    logits = logits.cpu()
    loss = F.cross_entropy(logits, torch.tensor([true_idx], dtype=torch.long)).item()
    pred_idx = logits.argmax(dim=1).item()
    pred_label = model.numeric_to_label_dict.get(pred_idx, str(pred_idx))
    return (loss, pred_label)


def find_high_loss_crops(crop_dirs, checkpoint_path, top_n=100, max_crops=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = CropModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(dev)
    print(f"Loaded CropModel from {checkpoint_path}, {len(model.label_dict)} classes")

    all_pairs = []
    for d in crop_dirs:
        pairs = discover_crops_from_imagefolder(d)
        if pairs:
            all_pairs.extend(pairs)
    if not all_pairs:
        for d in crop_dirs:
            for sub in d.iterdir():
                if sub.is_dir():
                    # pairs = list of (crop_path, label) from sub/ClassName/*.png
                    pairs = discover_crops_from_imagefolder(sub)
                    if pairs:
                        all_pairs.extend(pairs)
                        break

    # For quick testing: optionally limit how many crops we score
    if max_crops is not None:
        all_pairs = all_pairs[:max_crops]
    print(f"Found {len(all_pairs)} crops (after max_crops filter)")

    results = []
    for i, (crop_path, label) in enumerate(all_pairs):
        if (i + 1) % 500 == 0:
            print(f"Processing crop {i+1}/{len(all_pairs)}")
        out = compute_loss_for_crop(model, crop_path, label, dev)
        if out is not None:
            loss, pred_label = out
            results.append({
                "crop_path": str(crop_path),
                "true_label": label,
                "pred_label": pred_label,
                "loss": loss,
            })
    loss_df = pd.DataFrame(results)
    if loss_df.empty:
        return loss_df
    loss_df = loss_df.sort_values("loss", ascending=False)
    top = loss_df.head(top_n)
    print(f"\nTop {top_n} crops by loss (first 10):")
    print(top[["crop_path", "true_label", "pred_label", "loss"]].head(10).to_string())
    return top


def leaf_labels_from_taxonomy(taxonomy_path):
    """Collect leaf alias strings from transformed_taxonomy.json."""
    with open(taxonomy_path, encoding="utf-8") as f:
        data = json.load(f)

    def visit(node, out):
        if node.get("isLeaf") and node.get("alias"):
            out.add(node["alias"])
        for c in node.get("children", []):
            visit(c, out)

    out = set()
    for item in data.get("items", []):
        visit(item, out)
    return out


def detection_label_config_from_dict(label_dict, taxonomy_path=None):
    """Label Studio config for this review workflow.

    - Image with bounding boxes
    - RectangleLabels (Object / FalsePositive)
    - Per-region Taxonomy dropdown populated from transformed_taxonomy.json via apiUrl
    - We still set prediction_summary in task data; you can add a Header/Text in the config later if desired.
    """
    return (
        '<View>\n'
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        '    <Label value="Object" background="#D4380D"/>\n'
        '    <Label value="FalsePositive" background="#FFA39E"/>\n'
        '  </RectangleLabels>\n'
        '  <Text name="text" value="Select Taxonomy for Classification"/>\n'
        '  <Taxonomy name="taxonomy" perRegion="true" minWidth="600px" toName="image" '
        'apiUrl="https://raw.githubusercontent.com/weecology/BOEM/refs/heads/main/transformed_taxonomy.json" />\n'
        '</View>'
    )


def upload_parent_images_to_label_studio(
    top_crops_df,
    project_name,
    url,
    folder_name,
    server_cfg,
    label_dict,
    taxonomy_path=None,
):
    """Upload parent (patch) images for high-loss crops with detections overlaid (bbox + taxonomy)."""
    if top_crops_df.empty:
        print("No crops to upload.")
        return
    api_key = ls_mod.get_api_key()
    if api_key is None:
        raise ValueError("Label Studio API key not found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    label_config = detection_label_config_from_dict(label_dict, taxonomy_path)
    project = ls_mod.connect_to_label_studio(url=url, project_name=project_name, label_config=label_config)
    sftp = ls_mod.create_sftp_client(
        user=server_cfg["user"],
        host=server_cfg["host"],
        key_filename=os.path.expanduser(server_cfg["key_filename"]),
    )

    # Group by parent stem so we upload each parent image once
    top_crops_df = top_crops_df.copy()
    top_crops_df["parent_stem"] = top_crops_df["crop_path"].map(crop_path_to_parent_stem)
    parent_stems = top_crops_df["parent_stem"].unique().tolist()

    # Resolve parent patch path and detections for each stem; collect high-loss crop info for summary
    parent_paths = []
    preannotations = {}
    high_loss_by_basename = {}

    for stem in parent_stems:
        parent_path = find_parent_patch(stem)
        if parent_path is None:
            print(f"Skip parent stem (patch not found): {stem}")
            continue
        parent_paths.append(parent_path)
        basename = parent_path.name
        detections = find_detections_for_patch(stem)
        if detections is not None:
            detections = detections.copy()
            detections["image_path"] = basename
            # Use the species label from the high-loss crops for this parent to prefill taxonomy.
            # Prefer true_label; fall back to pred_label if needed.
            rows = top_crops_df[top_crops_df["parent_stem"] == stem]
            if not rows.empty:
                species_label = rows["true_label"].iloc[0]
                if pd.isna(species_label) or species_label in ("", "Object"):
                    species_label = rows["pred_label"].iloc[0]
                detections["cropmodel_label"] = species_label
            preannotations[basename] = detections
        else:
            preannotations[basename] = pd.DataFrame()
        # Build summary text for this parent from all high-loss crops tied to the stem
        rows = top_crops_df[top_crops_df["parent_stem"] == stem]
        lines = [f"{os.path.basename(r['crop_path'])}: true={r['true_label']} pred={r['pred_label']} loss={r['loss']:.3f}" for _, r in rows.iterrows()]
        high_loss_by_basename[basename] = "High-loss crops on this image:\n" + "\n".join(lines)

    if not parent_paths:
        print("No parent patches found for high-loss crops.")
        return

    # Copy parent images into a single temp dir so import_image_tasks can resolve image dimensions
    with tempfile.TemporaryDirectory(prefix="high_loss_parents_") as tmpdir:
        tmpdir = Path(tmpdir)
        for p in parent_paths:
            (tmpdir / p.name).write_bytes(p.read_bytes())
        image_list = [str(tmpdir / p.name) for p in parent_paths]
        ls_mod.upload_images(sftp, image_list, folder_name)

        # Build prediction_summary per task: high-loss crop info + detection summary
        for basename in preannotations:
            det_df = preannotations[basename]
            summary = high_loss_by_basename.get(basename, "")
            if not det_df.empty:
                det_summary = ls_mod.format_prediction_summary_for_task(det_df)
                summary = (summary + "\n\n" + det_summary) if summary else det_summary
            if not summary:
                summary = "No detections for this image."
            high_loss_by_basename[basename] = summary

        tasks = []
        for image_path in image_list:
            basename = os.path.basename(image_path)
            data_dict = {
                "image": os.path.join("/data/local-files/?d=BOEM/input/", basename),
                "flight_name": Path(image_path).parent.name or "high_loss_review",
                "prediction_summary": high_loss_by_basename.get(basename, "No detections for this image."),
            }
            pred_df = preannotations.get(basename, pd.DataFrame())
            if pred_df.empty:
                result_dict = []
            else:
                result_dict = [ls_mod.label_studio_bbox_format(str(tmpdir), pred_df, taxonomy_path=taxonomy_path)]
            tasks.append({"data": data_dict, "predictions": result_dict})
        project.import_tasks(tasks)

    print(f"Uploaded {len(tasks)} parent-image tasks to Label Studio project: {project_name}")


def run():
    parser = argparse.ArgumentParser(description="Find high-loss classification crops and upload to Label Studio")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "boem_conf" / "classification_model" / "finetune.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train_crop_dir", type=Path, default="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/crops/train/buffer_30/8d6309e78a4b49c9947e2100f6df93de")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument(
        "--max_crops",
        type=int,
        default=None,
        help="For testing: only score the first N crops (None = all)",
    )
    parser.add_argument("--project_name", type=str, default="USGS Classification - High Loss Review")
    parser.add_argument("--label_studio_config", type=Path, default=None)
    parser.add_argument("--no_upload", action="store_true", help="Only save CSV, do not upload")
    parser.add_argument("--rerun", action="store_false", default=True, help="Recompute high-loss crops even if output CSV already exists")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "transformed_taxonomy.json", help="Taxonomy JSON to restrict Label Studio choices")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if not checkpoint or not os.path.isfile(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)
    train_crop_dir = args.train_crop_dir or cfg.get("checkpoint_train_dir") or cfg.get("train_crop_image_dir")
    if train_crop_dir:
        train_crop_dir = Path(train_crop_dir)
    if not train_crop_dir or not train_crop_dir.exists():
        print(f"Train crop dir not found: {train_crop_dir}")
        sys.exit(1)
    out_csv = PROJECT_ROOT / "output" / f"high_loss_crops_top{args.top_n}.csv"
    if out_csv.exists() and not args.rerun:
        top_df = pd.read_csv(out_csv)
        print(f"Loaded existing {out_csv} ({len(top_df)} crops). Use --rerun to recompute.")
    else:
        crop_dirs = find_crop_dirs(cfg)
        if not crop_dirs:
            crop_dirs = [Path(train_crop_dir)]

        top_df = find_high_loss_crops(crop_dirs, checkpoint, top_n=args.top_n, max_crops=args.max_crops)
        if top_df.empty:
            print("No crops with valid loss.")
            sys.exit(0)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        top_df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

    if args.no_upload:
        return
    if args.label_studio_config and args.label_studio_config.exists():
        with open(args.label_studio_config, encoding="utf-8") as f:
            ls_cfg = yaml.safe_load(f).get("label_studio", {})
    else:
        ls_path = PROJECT_ROOT / "boem_conf" / "annotation" / "label_studio.yaml"
        if ls_path.exists():
            with open(ls_path, encoding="utf-8") as f:
                ls_cfg = yaml.safe_load(f).get("label_studio", {})
        else:
            ls_cfg = {
                "url": os.getenv("LABEL_STUDIO_URL", "https://labelstudio.naturecast.org/"),
                "folder_name": os.getenv("LABEL_STUDIO_FOLDER", "/media/T/lab-white-ernest/label_studio_data/BOEM"),
                "server": {"user": os.getenv("LABEL_STUDIO_USER", "b.weinstein"), "host": os.getenv("LABEL_STUDIO_HOST", "serenity.ifas.ufl.edu"), "key_filename": os.path.expanduser("~/.ssh/id_ed25519")},
            }
    server_cfg_path = PROJECT_ROOT / "boem_conf" / "server" / "serenity.yaml"
    if server_cfg_path.exists():
        with open(server_cfg_path, encoding="utf-8") as f:
            server_cfg = yaml.safe_load(f)
    else:
        server_cfg = dict(ls_cfg.get("server", {}))
    if not server_cfg:
        server_cfg = {"user": os.getenv("LABEL_STUDIO_USER", "b.weinstein"), "host": os.getenv("LABEL_STUDIO_HOST", "serenity.ifas.ufl.edu"), "key_filename": os.path.expanduser("~/.ssh/id_ed25519")}
    if "key_filename" in server_cfg:
        server_cfg["key_filename"] = os.path.expanduser(server_cfg["key_filename"])
    model = CropModel.load_from_checkpoint(checkpoint)
    upload_parent_images_to_label_studio(
        top_df, args.project_name, ls_cfg["url"], ls_cfg["folder_name"], server_cfg, model.label_dict, taxonomy_path=args.taxonomy
    )


if __name__ == "__main__":
    run()
