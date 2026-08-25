"""Score the flat CropModel and H-CAST against each other, and as an ensemble, on one val split.

Both models are run over the SAME crop PNGs -- the ones `scripts/classifier_confusion.py`
already scored, so the flat side is read from its saved logits rather than recomputed. That
matters: the two training paths build crops slightly differently (DeepForest's
expand_bbox_to_square vs src.hierarchical's), and a comparison that let each model see its own
crop geometry would price that difference as model quality.

Sanity check built in: H-CAST's Species@1 here should reproduce the ~76.7 its training run
logged on this split. If it does not, the crop geometry is wrong and nothing else printed is
trustworthy.

Reports species/genus/family accuracy per model, three ensembles, the head-to-head on the
crops where they disagree, and the Delphinidae/Laridae confusion that motivated the run.

Usage (one GPU):
  python scripts/compare_flat_vs_hcast.py --comet-id a3dc30a085f5442393736ecd96b564c5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.taxonomy_hier import load_taxonomy  # noqa: E402
from src import hierarchical  # noqa: E402
from src.classification import TURTLE_CLASS  # noqa: E402

BOEM = "/blue/ewhite/b.weinstein/BOEM"
CETACEAN_FAMILIES = {"Delphinidae", "Balaenopteridae", "Balaenidae", "Physeteridae", "Phocoenidae"}


class CropPngDataset(Dataset):
    """Reads the crop PNGs the flat model was scored on, with H-CAST's eval transform."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img)


def _rollup(probs: np.ndarray, species: list[str], mapping: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Sum species probabilities into their parent rank. Returns (probs, ordered parent names)."""
    parents = sorted({mapping[s] for s in species if s in mapping})
    idx = {p: i for i, p in enumerate(parents)}
    out = np.zeros((probs.shape[0], len(parents)), dtype=probs.dtype)
    for j, s in enumerate(species):
        if s in mapping:
            out[:, idx[mapping[s]]] += probs[:, j]
    return out, parents


def _acc(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comet-id", default="a3dc30a085f5442393736ecd96b564c5")
    p.add_argument("--hcast-checkpoint", default="output/usgs_hier/best_checkpoint.pth")
    p.add_argument("--hcast-label-csv", default="output/usgs_hier/species.csv")
    p.add_argument("--taxonomy", default="taxonomy.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    short = args.comet_id[:8]
    pred = pd.read_csv(f"{BOEM}/classifier_confusion_{short}_predictions.csv")
    flat_logits = np.load(f"{BOEM}/classifier_confusion_{short}_logits.npy")
    flat_species = [c for c in open(f"{BOEM}/classifier_confusion_{short}_classes.txt").read().split("\n") if c]
    assert len(pred) == len(flat_logits) == flat_logits.shape[0]
    assert len(flat_species) == flat_logits.shape[1]

    z = flat_logits - flat_logits.max(1, keepdims=True)
    flat_probs = np.exp(z)
    flat_probs /= flat_probs.sum(1, keepdims=True)

    triples, _ = load_taxonomy(args.taxonomy)
    sp2genus = {s: g for (_, g, s) in triples}
    sp2family = {s: f for (f, _, s) in triples}
    sp2genus[TURTLE_CLASS] = "Chelonioidea"
    sp2family[TURTLE_CLASS] = "Chelonioidea"

    truth_species = pred["true"].to_numpy()
    truth_genus = pred["true"].map(sp2genus).to_numpy()
    truth_family = pred["true"].map(sp2family).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  {len(pred)} val crops", flush=True)
    model = hierarchical.load_hcast_model(
        checkpoint_path=args.hcast_checkpoint, label_csv=args.hcast_label_csv, device=device
    )
    transform = hierarchical._default_transform(image_size=model.image_size, eval_crop_ratio=0.875)
    dl = DataLoader(
        CropPngDataset(pred["path"], transform),
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
    )

    sp_chunks, gn_chunks, fm_chunks = [], [], []
    with torch.no_grad():
        for batch in dl:
            out = model.predict_logits(batch)
            sp_chunks.append(torch.softmax(out[0], 1).cpu().numpy())
            gn_chunks.append(torch.softmax(out[1], 1).cpu().numpy())
            fm_chunks.append(torch.softmax(out[2], 1).cpu().numpy())
    hc_sp, hc_gn, hc_fm = np.vstack(sp_chunks), np.vstack(gn_chunks), np.vstack(fm_chunks)

    # HCastWrapper stores labels prefixed ("species_Alca torda"); classify_dataframe strips
    # the prefix in a nested helper, so do the same here or every name misses the taxonomy.
    def _names(numeric_to_label, n, prefix):
        out = []
        for i in range(n):
            key = numeric_to_label.get(i, f"{prefix}_{i}")
            out.append(key[len(prefix) + 1:] if key.startswith(f"{prefix}_") else key)
        return out

    hc_species = _names(model.species_numeric_to_label, hc_sp.shape[1], "species")
    hc_genus = _names(model.genus_numeric_to_label, hc_gn.shape[1], "genus")
    hc_family = _names(model.family_numeric_to_label, hc_fm.shape[1], "family")
    overlap = len(set(hc_species) & set(flat_species))
    if overlap == 0:
        raise SystemExit(
            f"H-CAST and flat share no species names (H-CAST e.g. {hc_species[:3]}, "
            f"flat e.g. {flat_species[:3]}) -- label mapping is wrong, refusing to report."
        )

    # ---- per-model accuracy -------------------------------------------------
    flat_sp_pred = np.array(flat_species)[flat_probs.argmax(1)]
    hc_sp_pred = np.array(hc_species)[hc_sp.argmax(1)]

    flat_gn_probs, flat_gn_names = _rollup(flat_probs, flat_species, sp2genus)
    flat_fm_probs, flat_fm_names = _rollup(flat_probs, flat_species, sp2family)
    flat_gn_pred = np.array(flat_gn_names)[flat_gn_probs.argmax(1)]
    flat_fm_pred = np.array(flat_fm_names)[flat_fm_probs.argmax(1)]
    hc_gn_pred = np.array(hc_genus)[hc_gn.argmax(1)]
    hc_fm_pred = np.array(hc_family)[hc_fm.argmax(1)]

    print("\n=== accuracy on the shared val split ===")
    print(f"{'model':28s} {'Species@1':>10s} {'Genus@1':>9s} {'Family@1':>9s}")
    print(f"{'flat CropModel':28s} {_acc(flat_sp_pred, truth_species)*100:9.2f}% "
          f"{_acc(flat_gn_pred, truth_genus)*100:8.2f}% {_acc(flat_fm_pred, truth_family)*100:8.2f}%"
          "   (genus/family are rollups of the species softmax)")
    print(f"{'H-CAST':28s} {_acc(hc_sp_pred, truth_species)*100:9.2f}% "
          f"{_acc(hc_gn_pred, truth_genus)*100:8.2f}% {_acc(hc_fm_pred, truth_family)*100:8.2f}%"
          "   (native heads)")

    # ---- ensembles ----------------------------------------------------------
    shared = [s for s in flat_species if s in set(hc_species)]
    fi = [flat_species.index(s) for s in shared]
    hi = [hc_species.index(s) for s in shared]
    f_sub = flat_probs[:, fi] / flat_probs[:, fi].sum(1, keepdims=True)
    h_sub = hc_sp[:, hi] / hc_sp[:, hi].sum(1, keepdims=True)
    shared_arr = np.array(shared)
    in_shared = np.isin(truth_species, shared_arr)
    print(f"\nshared species vocabulary: {len(shared)} of {len(flat_species)} flat / {len(hc_species)} H-CAST"
          f"  ({in_shared.sum()}/{len(pred)} val crops have a truth label in it)")

    ens = {
        "mean of probabilities": (f_sub + h_sub) / 2,
        "product (log-average)": np.exp((np.log(f_sub + 1e-12) + np.log(h_sub + 1e-12)) / 2),
        "H-CAST family x flat species": None,
    }
    print("\n=== ensembles, on the crops both models can express ===")
    print(f"{'method':32s} {'Species@1':>10s} {'Family@1':>9s}")
    base_f = _acc(np.array(shared)[f_sub[in_shared].argmax(1)], truth_species[in_shared])
    base_h = _acc(np.array(shared)[h_sub[in_shared].argmax(1)], truth_species[in_shared])
    fam_of_shared = {s: sp2family[s] for s in shared if s in sp2family}
    for name, probs in ens.items():
        if probs is None:
            # Re-weight the flat species softmax by H-CAST's family posterior, then argmax.
            w = np.zeros_like(f_sub)
            fam_idx = {f: i for i, f in enumerate(hc_family)}
            for j, s in enumerate(shared):
                f = fam_of_shared.get(s)
                w[:, j] = hc_fm[:, fam_idx[f]] if f in fam_idx else 0.0
            probs = f_sub * w
            probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
        sp = np.array(shared)[probs.argmax(1)]
        fm, fm_names = _rollup(probs, shared, sp2family)
        fmp = np.array(fm_names)[fm.argmax(1)]
        print(f"{name:32s} {_acc(sp[in_shared], truth_species[in_shared])*100:9.2f}% "
              f"{_acc(fmp[in_shared], truth_family[in_shared])*100:8.2f}%")
    print(f"{'(flat alone, same subset)':32s} {base_f*100:9.2f}%")
    print(f"{'(H-CAST alone, same subset)':32s} {base_h*100:9.2f}%")

    # ---- head to head -------------------------------------------------------
    disagree = flat_sp_pred != hc_sp_pred
    both_wrong = disagree & (flat_sp_pred != truth_species) & (hc_sp_pred != truth_species)
    print(f"\n=== head to head at species rank ===")
    print(f"  agree            : {(~disagree).sum():5d}  ({_acc(flat_sp_pred[~disagree], truth_species[~disagree])*100:.1f}% correct when they agree)")
    print(f"  disagree         : {disagree.sum():5d}")
    print(f"    flat right     : {(disagree & (flat_sp_pred == truth_species)).sum():5d}")
    print(f"    H-CAST right   : {(disagree & (hc_sp_pred == truth_species)).sum():5d}")
    print(f"    both wrong     : {both_wrong.sum():5d}")

    # Disagreement as a review trigger: what share of the flat model's errors does it catch,
    # and at what review cost? This is the number that decides whether disagreement is worth
    # routing to humans rather than just flagging "unresolved" and keeping the flat label.
    flat_wrong = flat_sp_pred != truth_species
    print(f"\n  flat accuracy when they agree   : {_acc(flat_sp_pred[~disagree], truth_species[~disagree])*100:.1f}%")
    print(f"  flat accuracy when they disagree: {_acc(flat_sp_pred[disagree], truth_species[disagree])*100:.1f}%")
    print(f"  share of ALL flat errors sitting in the disagree bucket: "
          f"{(flat_wrong & disagree).sum() / max(1, flat_wrong.sum())*100:.0f}% "
          f"({int((flat_wrong & disagree).sum())}/{int(flat_wrong.sum())}), "
          f"for {disagree.sum()/len(pred)*100:.0f}% of crops reviewed")

    # ---- the cetacean / gull question --------------------------------------
    print("\n=== Delphinidae vs Laridae ===")
    is_dolphin = np.isin(truth_family, list(CETACEAN_FAMILIES))
    is_gull = truth_family == "Laridae"
    ens_probs = np.exp((np.log(f_sub + 1e-12) + np.log(h_sub + 1e-12)) / 2)
    ens_fm, ens_fm_names = _rollup(ens_probs, shared, sp2family)
    ens_fm_pred = np.array(ens_fm_names)[ens_fm.argmax(1)]
    for name, fm_pred in [("flat (rollup)", flat_fm_pred), ("H-CAST (family head)", hc_fm_pred),
                          ("ensemble (product)", ens_fm_pred)]:
        d_ok = (fm_pred[is_dolphin] == truth_family[is_dolphin]).sum()
        d_as_gull = (fm_pred[is_dolphin] == "Laridae").sum()
        g_ok = (fm_pred[is_gull] == truth_family[is_gull]).sum()
        g_as_cet = np.isin(fm_pred[is_gull], list(CETACEAN_FAMILIES)).sum()
        print(f"  {name:22s} cetacean crops {int(is_dolphin.sum()):4d}: {int(d_ok):3d} right, {int(d_as_gull):3d} -> Laridae"
              f"   |  Laridae crops {int(is_gull.sum()):4d}: {int(g_ok):4d} right, {int(g_as_cet):3d} -> cetacean")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pd.DataFrame({
            "path": pred["path"], "true_species": truth_species, "true_family": truth_family,
            "flat_species": flat_sp_pred, "flat_family": flat_fm_pred,
            "hcast_species": hc_sp_pred, "hcast_family": hc_fm_pred,
        }).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
