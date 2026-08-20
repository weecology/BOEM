"""Measure how H-CAST inference accuracy depends on crop geometry.

H-CAST is applied after the detection gate on boxes the CropModel also sees, but
its inference path (src/hierarchical.InferenceCropDataset) historically took the
raw detection box, with no expand buffer, no squaring, and a straight squash to
224x224 -- while scripts/USGS_hierarchical.py trains on boxes padded by
--expand-pixels, squared, then resized at eval_crop_ratio 0.875. This sweeps the
inference geometry over a fixed checkpoint and val split so the mismatch can be
priced rather than assumed.

The training-matched row (expand 30 / square / 0.875) should reproduce the
Species@1 the training run itself logged; that is the harness's sanity check.
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.taxonomy_hier import load_taxonomy, load_taxonomy_restricted_to_species  # noqa: E402
from src import hierarchical  # noqa: E402
from src.classification import TURTLE_CLASS  # noqa: E402

# (expand_pixels, square, eval_crop_ratio, note)
#
# Rows 1-8 are a clean 4 (expand) x 2 (square) factorial at the training resize,
# so the expand effect and the squaring effect can be read separately rather than
# confounded. That separation matters because the two knobs overlap: on this val
# split the median raw box is 33x33 px with a long/short aspect of 1.46, but after
# padding 30 px per side the median aspect falls to 1.13 and only 2.5% of boxes
# still exceed 1.5. Expanding already does most of what squaring would do, so the
# square effect should be small at expand 30 and large at expand 0.
GRID = [
    (0, False, 0.875, ""),
    (0, True, 0.875, "training transform, no expand"),
    (15, False, 0.875, ""),
    (15, True, 0.875, ""),
    (30, False, 0.875, "expand, no squaring"),
    (30, True, 0.875, "training-matched (--expand-pixels 30)"),
    (60, False, 0.875, ""),
    (60, True, 0.875, ""),
    # Resize policy, held at the best-guess expand.
    (0, False, None, "pipeline default before this change"),
    (30, False, None, "expand, no squaring, squash resize"),
    (30, True, None, "expand+square, squash resize"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--val-split-csv", required=True)
    parser.add_argument("--train-split-csv", required=True,
                        help="Needed to rebuild the exact label vocabulary of the run")
    parser.add_argument("--image-dir", default="/blue/ewhite/b.weinstein/BOEM/training/crops")
    parser.add_argument("--taxonomy", default="taxonomy.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="output/usgs_hier/expand_sweep.csv")
    parser.add_argument(
        "--restrict-to-label-csv", action="store_true",
        help="Keep only val rows whose species the checkpoint can actually predict. "
             "Needed when the checkpoint's vocabulary is narrower than the split "
             "(e.g. the 37-class Dec-2025 model against a 68-class split), otherwise "
             "accuracy is capped by unpredictable classes and the geometry effect is diluted.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_split_csv, low_memory=False)
    val_df = pd.read_csv(args.val_split_csv, low_memory=False)
    unique_labels = set(train_df["label"].unique()) | set(val_df["label"].unique())
    nb_classes, name_to_ids = load_taxonomy_restricted_to_species(
        args.taxonomy, unique_labels, include_ancestor_labels=True
    )
    if TURTLE_CLASS in unique_labels and TURTLE_CLASS not in name_to_ids:
        sid, gid, fid = nb_classes[0], nb_classes[1], nb_classes[2]
        name_to_ids[TURTLE_CLASS] = (fid, gid, sid)
        nb_classes = [n + 1 for n in nb_classes]

    # Same filter the training val loader applies, so the row count is comparable
    # to the Species@1 printed during training.
    val_df = val_df[val_df["label"].isin(name_to_ids)].copy()
    label_df = pd.read_csv(args.label_csv)
    # Ground truth comes from the taxonomy, not from the checkpoint's label CSV --
    # the second arm runs a checkpoint whose vocabulary differs from this split, so
    # its CSV indices do not address these species.
    triples, _ = load_taxonomy(args.taxonomy)
    species_to_genus_name = {s: g for (_, g, s) in triples}
    species_to_family_name = {s: f for (f, _, s) in triples}
    species_to_genus_name[TURTLE_CLASS] = "Chelonioidea"
    species_to_family_name[TURTLE_CLASS] = "Chelonioidea"
    val_df["true_species"] = val_df["label"]
    val_df["true_genus"] = val_df["label"].map(species_to_genus_name)
    val_df["true_family"] = val_df["label"].map(species_to_family_name)
    if args.restrict_to_label_csv:
        known = set(label_df["species"].dropna())
        before = len(val_df)
        val_df = val_df[val_df["label"].isin(known)].copy()
        print(f"Restricted to the checkpoint's {len(known)} species: {before} -> {len(val_df)} crops")
    print(f"Val crops: {len(val_df)}  species classes present: {val_df['label'].nunique()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = hierarchical.load_hcast_model(
        checkpoint_path=args.checkpoint, label_csv=args.label_csv, device=device
    )

    rows = []
    for expand, square, ratio, note in GRID:
        preds = hierarchical.classify_dataframe(
            val_df, image_dir=args.image_dir, model=model,
            batch_size=args.batch_size, num_workers=args.workers,
            expand_pixels=expand, square=square, eval_crop_ratio=ratio,
        )
        rec = {
            "expand": expand,
            "square": square,
            "eval_crop_ratio": ratio,
            "species_acc1": (preds["hcast_species"] == preds["true_species"]).mean(),
            "genus_acc1": (preds["hcast_genus"] == preds["true_genus"]).mean(),
            "family_acc1": (preds["hcast_family"] == preds["true_family"]).mean(),
            "mean_species_score": preds["hcast_species_score"].mean(),
            "note": note,
        }
        rows.append(rec)
        print(
            f"expand={expand:>3} square={str(square):>5} ratio={str(ratio):>5} | "
            f"Species@1 {rec['species_acc1'] * 100:6.2f}  Genus@1 {rec['genus_acc1'] * 100:6.2f}  "
            f"Family@1 {rec['family_acc1'] * 100:6.2f}  conf {rec['mean_species_score']:.3f}  {note}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {os.path.abspath(args.out)}")
    best = out.loc[out["species_acc1"].idxmax()]
    print(f"Best Species@1: expand={best['expand']} square={best['square']} "
          f"ratio={best['eval_crop_ratio']} -> {best['species_acc1'] * 100:.2f}")


if __name__ == "__main__":
    sys.exit(main())
