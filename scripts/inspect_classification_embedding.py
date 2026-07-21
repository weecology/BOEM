"""Visualize classifier embeddings and compare no-metadata vs metadata models.

Extracts 2048-d ResNet avgpool embeddings from both checkpoints, projects to 2D
with t-SNE, and highlights target classes. Also plots per-class accuracy deltas
and confusion detail for classes of interest.

Usage:
    uv run python scripts/inspect_classification_embedding.py

Outputs saved to output/embedding_comparison/
"""
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from deepforest.model import CropModel
from collections import defaultdict

# ── checkpoints & crop dirs ──────────────────────────────────────────────────
BASE = Path("/blue/ewhite/b.weinstein/BOEM/training")
CKPT_NO_META  = BASE / "classification/checkpoints/buffer_30/63980ef6a6fd4c45a69e7955150875b3.ckpt"
CKPT_WITH_META = BASE / "classification/checkpoints/buffer_30/2a634ae82ec04eeb865fd9cc93a1b701.ckpt"
VAL_DIR_NO_META  = BASE / "classification/crops/val/buffer_30/63980ef6a6fd4c45a69e7955150875b3"
VAL_DIR_WITH_META = BASE / "classification/crops/val/buffer_30/2a634ae82ec04eeb865fd9cc93a1b701"

OUT_DIR = PROJECT_ROOT / "output" / "embedding_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── classes to highlight ─────────────────────────────────────────────────────
HIGHLIGHT = {
    "Somateria mollissima": ("tab:orange", "Common Eider"),
    "Morus bassanus":       ("tab:blue",   "Northern Gannet"),
    "Larus fuscus":         ("tab:red",    "Lesser Black-backed Gull (regressed)"),
    "Fratercula arctica":   ("tab:purple", "Atlantic Puffin (regressed)"),
}

# ── known val accuracies from log output ─────────────────────────────────────
VAL_ACC_NO_META = {
    "Alca torda":0.75,"Alle alle":0.796,"Anas platyrhynchos":0.818,"Anas rubripes":0.5,
    "Anas strepera":0.875,"Ardea alba":0.6,"Ardea herodias":0.2,"Aythya affinis":0.0,
    "Aythya marila":0.946,"Balaenoptera acutorostrata":0.667,"Balaenoptera physalus":0.4,
    "Branta canadensis":0.0,"Bucephala albeola":0.704,"Bucephala clangula":0.97,
    "Calidris alba":0.083,"Calonectris Puffinus":0.308,"Calonectris diomedea":0.88,
    "Caretta caretta":0.75,"Chlidonias niger":0.909,"Chroicocephalus philadelphia":0.9,
    "Clangula hyemalis":0.7,"Coryphaena hippurus":1.0,"Delphinus delphis":1.0,
    "Dermochelys coriacea":0.8,"Fratercula arctica":0.4,"Fulmarus glacialis":0.918,
    "Gavia immer":0.854,"Gavia stellata":0.89,"Gelochelidon nilotica":0.333,
    "Halichoerus grypus":0.273,"Larus argentatus":0.65,"Larus delawarensis":0.72,
    "Larus fuscus":0.4,"Larus marinus":0.85,"Lepidochelys kempii":0.6,
    "Leucophaeus atricilla":0.85,"Melanitta fusca":0.57,"Melanitta nigra":0.444,
    "Melanitta perspicillata":0.62,"Mergus serrator":0.919,"Mola mola":0.333,
    "Morus bassanus":0.78,"Oceanites oceanicus":0.947,"Pelecanus occidentalis":0.714,
    "Phalacrocorax auritus":0.925,"Phalaropus fulicarius":0.85,"Phoca vitulina":0.737,
    "Podiceps auritus":0.308,"Puffinus gravis":0.784,"Rachycentron canadum":0.333,
    "Rhinoptera bonasus":0.98,"Rissa tridactyla":0.95,"Somateria mollissima":0.87,
    "Sphyrna zygaena":0.8,"Squalus acanthias":0.0,"Stenella frontalis":0.4,
    "Sterna forsteri":0.237,"Sterna hirundo":0.769,"Stomolophus meleagris":1.0,
    "Thalasseus maximus":0.733,"Thunnus thynnus":1.0,"Tursiops truncatus":0.375,
}
VAL_ACC_WITH_META = {
    "Alca torda":0.5,"Alle alle":0.755,"Anas platyrhynchos":0.909,"Anas rubripes":0.5,
    "Anas strepera":0.8125,"Ardea alba":0.4,"Ardea herodias":0.2,"Aythya affinis":0.0,
    "Aythya marila":0.892,"Balaenoptera acutorostrata":0.667,"Balaenoptera physalus":0.6,
    "Branta canadensis":0.0,"Bucephala albeola":0.852,"Bucephala clangula":0.97,
    "Calidris alba":0.083,"Calonectris Puffinus":0.308,"Calonectris diomedea":0.76,
    "Caretta caretta":0.875,"Chlidonias niger":0.909,"Chroicocephalus philadelphia":0.91,
    "Clangula hyemalis":0.71,"Coryphaena hippurus":0.95,"Delphinus delphis":0.8,
    "Dermochelys coriacea":0.8,"Fratercula arctica":0.0,"Fulmarus glacialis":0.902,
    "Gavia immer":0.854,"Gavia stellata":0.96,"Gelochelidon nilotica":0.333,
    "Halichoerus grypus":0.182,"Larus argentatus":0.75,"Larus delawarensis":0.65,
    "Larus fuscus":0.0,"Larus marinus":0.85,"Lepidochelys kempii":0.6,
    "Leucophaeus atricilla":0.83,"Melanitta fusca":0.624,"Melanitta nigra":0.75,
    "Melanitta perspicillata":0.8,"Mergus serrator":0.935,"Mola mola":1.0,
    "Morus bassanus":0.8,"Oceanites oceanicus":0.947,"Pelecanus occidentalis":0.857,
    "Phalacrocorax auritus":0.871,"Phalaropus fulicarius":0.89,"Phoca vitulina":0.895,
    "Podiceps auritus":0.077,"Puffinus gravis":0.882,"Rachycentron canadum":0.444,
    "Rhinoptera bonasus":0.99,"Rissa tridactyla":0.93,"Somateria mollissima":0.95,
    "Sphyrna zygaena":0.8,"Squalus acanthias":0.0,"Stenella frontalis":0.4,
    "Sterna forsteri":0.136,"Sterna hirundo":0.769,"Stomolophus meleagris":1.0,
    "Thalasseus maximus":0.8,"Thunnus thynnus":0.833,"Tursiops truncatus":0.5,
}


def extract_embeddings(ckpt_path: Path, val_dir: Path, batch_size: int = 64) -> tuple:
    """Load checkpoint, hook avgpool, return (embeddings [N,2048], labels [N], class_names)."""
    print(f"Loading {ckpt_path.name} ...")
    model = CropModel.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = model.get_transform(augmentations=None)
    dataset = ImageFolder(root=str(val_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    embeddings, targets = [], []
    hook_output = {}

    def _hook(module, input, output):
        hook_output["feat"] = output.flatten(1).detach().cpu()

    handle = model.model.avgpool.register_forward_hook(_hook)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            _ = model.model(imgs)
            embeddings.append(hook_output["feat"])
            targets.append(lbls)
    handle.remove()

    emb = torch.cat(embeddings).numpy()
    tgt = torch.cat(targets).numpy()
    class_names = [cls for cls, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])]
    print(f"  → {emb.shape[0]} crops, {len(class_names)} classes")
    return emb, tgt, class_names


def tsne_project(embeddings: np.ndarray, perplexity: int = 40, seed: int = 42) -> np.ndarray:
    print(f"  t-SNE on {embeddings.shape} ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_jobs=-1)
    return tsne.fit_transform(embeddings)


def plot_tsne(xy: np.ndarray, targets: np.ndarray, class_names: list, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 10))
    highlight_idx = {cls: i for i, cls in enumerate(class_names) if cls in HIGHLIGHT}

    # All background points
    bg_mask = np.isin(targets, list(highlight_idx.values()), invert=True)
    ax.scatter(xy[bg_mask, 0], xy[bg_mask, 1], c="lightgray", s=6, alpha=0.4, linewidths=0)

    # Highlighted classes on top
    legend_handles = [mpatches.Patch(color="lightgray", label="Other classes")]
    for cls, idx in highlight_idx.items():
        color, label = HIGHLIGHT[cls]
        mask = targets == idx
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=25, alpha=0.85,
                   linewidths=0.3, edgecolors="black", zorder=5)
        legend_handles.append(mpatches.Patch(color=color, label=f"{label} (n={mask.sum()})"))

    ax.set_title(title, fontsize=13)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_accuracy_delta(out_path: Path):
    """Bar chart of per-class accuracy delta (with_meta - no_meta), sorted."""
    classes = sorted(VAL_ACC_NO_META.keys())
    deltas = [VAL_ACC_WITH_META.get(c, 0) - VAL_ACC_NO_META.get(c, 0) for c in classes]
    order = np.argsort(deltas)
    classes_s = [classes[i] for i in order]
    deltas_s  = [deltas[i]  for i in order]
    colors = ["tab:red" if d < 0 else "tab:green" for d in deltas_s]

    fig, ax = plt.subplots(figsize=(8, 14))
    bars = ax.barh(range(len(classes_s)), deltas_s, color=colors, height=0.7)
    ax.set_yticks(range(len(classes_s)))
    ax.set_yticklabels(classes_s, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy delta (with_meta − no_meta)")
    ax.set_title("Per-class accuracy change from adding metadata", fontsize=11)

    # Annotate highlight classes
    for i, cls in enumerate(classes_s):
        if cls in HIGHLIGHT:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(HIGHLIGHT[cls][0])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_confusion_for_class(
    emb: np.ndarray, tgt: np.ndarray, preds: np.ndarray,
    class_names: list, focus_class: str, title: str, out_path: Path
):
    """Show what the model confused the focus class with (top-10 predicted classes)."""
    if focus_class not in class_names:
        print(f"  {focus_class} not in dataset, skipping confusion plot")
        return
    idx = class_names.index(focus_class)
    focus_mask = tgt == idx
    if focus_mask.sum() == 0:
        print(f"  No val samples for {focus_class}")
        return

    focus_preds = preds[focus_mask]
    counts = defaultdict(int)
    for p in focus_preds:
        counts[class_names[p]] += 1
    total = focus_mask.sum()
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:12]
    labels = [f"{cls}\n({cnt}/{total})" for cls, cnt in sorted_counts]
    values = [cnt for _, cnt in sorted_counts]
    bar_colors = ["tab:green" if cls == focus_class else "tab:red" for cls, _ in sorted_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\nTrue class: {focus_class} (n={total})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def get_predictions(model_path: Path, val_dir: Path, batch_size: int = 64) -> np.ndarray:
    """Run full forward pass and return predicted class indices."""
    model = CropModel.load_from_checkpoint(str(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    transform = model.get_transform(augmentations=None)
    dataset = ImageFolder(root=str(val_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model.model(imgs.to(device))
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def main():
    # ── accuracy delta bar chart (no GPU needed) ──────────────────────────────
    print("\n[1/5] Accuracy delta bar chart")
    plot_accuracy_delta(OUT_DIR / "accuracy_delta.png")

    # ── embeddings ───────────────────────────────────────────────────────────
    print("\n[2/5] Extracting no-metadata embeddings")
    emb_nm, tgt_nm, classes_nm = extract_embeddings(CKPT_NO_META, VAL_DIR_NO_META)

    print("\n[3/5] Extracting with-metadata embeddings")
    emb_wm, tgt_wm, classes_wm = extract_embeddings(CKPT_WITH_META, VAL_DIR_WITH_META)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("\n[4/5] t-SNE projections")
    xy_nm = tsne_project(emb_nm)
    plot_tsne(xy_nm, tgt_nm, classes_nm,
              "Embeddings — no metadata (micro-acc 74.3%)",
              OUT_DIR / "tsne_no_meta.png")

    xy_wm = tsne_project(emb_wm)
    plot_tsne(xy_wm, tgt_wm, classes_wm,
              "Embeddings — with metadata (micro-acc 75.8%)",
              OUT_DIR / "tsne_with_meta.png")

    # ── confusion detail for focus classes ───────────────────────────────────
    print("\n[5/5] Confusion detail plots")
    preds_nm = get_predictions(CKPT_NO_META, VAL_DIR_NO_META)
    preds_wm = get_predictions(CKPT_WITH_META, VAL_DIR_WITH_META)

    focus_classes = [
        "Somateria mollissima",  # Common Eider
        "Morus bassanus",        # Northern Gannet
        "Larus fuscus",          # big regression
        "Fratercula arctica",    # big regression
    ]
    for cls in focus_classes:
        safe = cls.replace(" ", "_")
        plot_confusion_for_class(emb_nm, tgt_nm, preds_nm, classes_nm, cls,
                                 "No metadata", OUT_DIR / f"confusion_no_meta_{safe}.png")
        plot_confusion_for_class(emb_wm, tgt_wm, preds_wm, classes_wm, cls,
                                 "With metadata", OUT_DIR / f"confusion_with_meta_{safe}.png")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
