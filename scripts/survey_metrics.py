"""Abundance-weighted classification accuracy for the report table.

Two metrics that weight per-class accuracy by how much of the survey each class
actually accounts for, instead of treating all 70 classes as equal:

  Rank 10 abundance : accuracy on the 10 most abundant species.
  90% abundance     : accuracy on the species that together make up 90% of all
                      labelled individuals (22 species for a3dc30a0).

Abundance is recovered from the per-image annotation CSVs with the SAME filters
scripts/USGS_classification.py applies before it splits and class-balances --
the train/val splits themselves are capped (1896 train / 100 val per class) and
so carry no abundance signal at all.

Usage:  python scripts/survey_metrics.py [comet_id]
        python scripts/survey_metrics.py --rebuild-abundance   # rescan 72k CSVs
"""
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classification import map_turtle_labels

ROOT = "/blue/ewhite/b.weinstein/BOEM"
OUTPUT = Path(__file__).resolve().parent.parent / "output"
CROPS = f"{ROOT}/training/crops"
DEFAULT_ID = "a3dc30a085f5442393736ecd96b564c5"
ABUNDANCE_CSV = OUTPUT / "corpus_abundance.csv"

# Mirrors UBFAI_CROPS_EXCLUDE_* in scripts/USGS_classification.py.
EXCLUDE_CSV = {"train.csv", "test.csv", "zero_shot.csv"}
EXCLUDE_PREFIX = ("train_max_empty_", "NEAQ")


def build_abundance():
    """Pre-balance class abundance from the per-image annotation CSVs."""
    files = [
        f for f in glob.glob(os.path.join(CROPS, "*.csv"))
        if os.path.basename(f) not in EXCLUDE_CSV
        and not os.path.basename(f).startswith(EXCLUDE_PREFIX)
    ]
    print(f"[abundance] reading {len(files)} per-image CSVs")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df = df.loc[~df.duplicated(
        subset=["image_path", "xmin", "ymin", "xmax", "ymax", "label"], keep="first")]
    df["label"] = map_turtle_labels(df["label"])
    df = df.groupby("label").filter(lambda x: len(x) > 25)
    df = df[df["label"].str.contains(" ", na=False)]
    df = df[~df.label.isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile",
                            "Turtle", "Mammal", "Artificial"])]
    df["label"] = df["label"].apply(lambda s: " ".join(str(s).split()[:2]))
    df = df[df["label"].apply(lambda x: len(x.split()) == 2)]

    counts = df.label.value_counts().rename_axis("species").rename("abundance")
    counts.to_csv(ABUNDANCE_CSV)
    print(f"[abundance] {len(df):,} crops, {len(counts)} classes -> {ABUNDANCE_CSV}")
    return counts


def load_table(comet_id):
    """Per-class val accuracy joined to corpus abundance, sorted most abundant first."""
    pred = pd.read_csv(f"{ROOT}/classifier_confusion_{comet_id[:8]}_predictions.csv")
    acc = pred.groupby("true").agg(val_n=("ok", "size"), val_correct=("ok", "sum"))
    acc["accuracy"] = acc.val_correct / acc.val_n

    abundance = (build_abundance() if not ABUNDANCE_CSV.exists()
                 else pd.read_csv(ABUNDANCE_CSV).set_index("species")["abundance"])

    df = acc.join(abundance)
    missing = df[df.abundance.isna()]
    if not missing.empty:
        # Ambiguous slash-classes ("Calonectris/Puffinus diomedea/gravis") normalise to a
        # model class name that never appears verbatim in the corpus. Both are tiny.
        print(f"[warn] no corpus abundance for {list(missing.index)} -- excluded")
    df = df[df.abundance.notna()].sort_values("abundance", ascending=False)
    df["cum_share"] = df.abundance.cumsum() / df.abundance.sum()
    return pred, df


def report(sub, title, total_abundance):
    print(f"\n=== {title} ===")
    print(sub[["abundance", "cum_share", "val_n", "accuracy"]]
          .to_string(float_format=lambda v: f"{v:,.4f}"))
    weighted = (sub.abundance * sub.accuracy).sum() / sub.abundance.sum()
    print(f"\n  {len(sub)} species = {sub.abundance.sum() / total_abundance:.1%} "
          f"of individuals, {int(sub.val_n.sum()):,} val crops")
    print(f"  abundance-weighted accuracy : {weighted:.4f}")
    print(f"  unweighted (macro)          : {sub.accuracy.mean():.4f}")
    print(f"  pooled crops (micro)        : {sub.val_correct.sum() / sub.val_n.sum():.4f}")
    return weighted


def main():
    if "--rebuild-abundance" in sys.argv:
        build_abundance()
        sys.argv.remove("--rebuild-abundance")
    comet_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID

    pred, df = load_table(comet_id)
    total = df.abundance.sum()
    print(f"\ncheckpoint {comet_id[:8]}  |  {len(df)} classes  |  "
          f"{int(total):,} labelled individuals  |  "
          f"overall val accuracy {pred.ok.mean():.4f}")

    report(df.head(10), "RANK 10 ABUNDANCE", total)

    n90 = int((df.cum_share < 0.90).sum()) + 1
    report(df.head(n90), "90% ABUNDANCE", total)

    tail = df.tail(len(df) - n90)
    print(f"\n[tail] the other {len(tail)} species are "
          f"{tail.abundance.sum() / total:.1%} of individuals, "
          f"macro accuracy {tail.accuracy.mean():.4f}")


if __name__ == "__main__":
    main()
