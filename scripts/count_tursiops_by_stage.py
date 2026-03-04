"""Count Tursiops truncatus at each pipeline stage (annotations → detection crops → UBFAI → classification train/val).

Run from repo root: uv run python count_tursiops_by_stage.py
"""

import glob
import os

import numpy as np
import pandas as pd

ANNOTATIONS_BASE = "/blue/ewhite/b.weinstein/BOEM/annotations"
DETECTION_CROPS_BASE = "/blue/ewhite/b.weinstein/BOEM/detection/crops"
UBFAI_CROPS = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
LABEL = "Tursiops truncatus"


def _count_label(df: pd.DataFrame, label: str) -> int:
    if "label" not in df.columns:
        return 0
    return int((df["label"].astype(str).str.strip().str.lower() == label.lower()).sum())


def main():
    print("Tursiops truncatus counts by stage\n" + "=" * 60)
    steps = []

    # 1) Annotations (train / validation / review) – Label Studio source
    n_ann = 0
    for sub in ("train", "validation", "review"):
        parent = os.path.join(ANNOTATIONS_BASE, sub)
        if not os.path.isdir(parent):
            continue
        for path in glob.glob(os.path.join(parent, "*", "*.csv")):
            try:
                df = pd.read_csv(path)
                n_ann += _count_label(df, LABEL)
            except Exception:
                pass
    steps.append(("Annotations (train/validation/review)", n_ann))
    print(f"1) Annotations (train/validation/review): {n_ann}")

    # 2) Detection crops – output of prepare_USGS --generate-detection-crops
    n_det = 0
    for path in glob.glob(os.path.join(DETECTION_CROPS_BASE, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path, low_memory=False)
            n_det += _count_label(df, LABEL)
        except Exception:
            pass
    steps.append(("Detection crops (detection/crops/**/*.csv)", n_det))
    print(f"2) Detection crops (detection/crops/**/*.csv): {n_det}")
    if n_ann > 0 and n_det < n_ann:
        print(f"   >>> BOTTLENECK: {n_ann - n_det} rows lost here. Detection crop CSVs are")
        print("       not regenerated when they already exist. Run prepare_USGS.py")
        print("       --generate-detection-crops; remove existing detection/crops CSVs")
        print("       for affected flights to refresh from annotations.")

    # 3) UBFAI crops – after prepare_USGS Stage 2 (collect_workflow_annotations)
    crop_csvs = [
        p
        for p in glob.glob(os.path.join(UBFAI_CROPS, "*.csv"))
        if os.path.basename(p) not in ("train.csv", "test.csv", "zero_shot.csv")
    ]
    if not crop_csvs:
        print("3) UBFAI crops (per-image CSVs): no CSVs found")
        n_ubfai = 0
        steps.append(("UBFAI crops (per-image CSVs)", 0))
    else:
        ubfai_all = pd.concat(
            [pd.read_csv(p, low_memory=False) for p in crop_csvs], ignore_index=True
        )
        n_ubfai = _count_label(ubfai_all, LABEL)
        steps.append(("UBFAI crops (per-image CSVs)", n_ubfai))
        print(f"3) UBFAI crops (per-image CSVs, n={len(crop_csvs)} files): {n_ubfai}")

    # 4) After USGS_classification filters (same order as in USGS_classification.py)
    if not crop_csvs:
        print("4) After filters / train|val: skipping (no UBFAI CSVs)")
        return
    combined = pd.concat(
        [pd.read_csv(x, low_memory=False) for x in crop_csvs], ignore_index=True
    )

    # Apply same filters as USGS_classification.py
    combined = combined.groupby("label").filter(lambda x: len(x) > 25)
    combined = combined[combined["label"].astype(str).str.contains(" ", na=False)]
    combined = combined[~combined["label"].isin([0, "0", "FalsePositive", "Object", "Bird", "Reptile", "Turtle", "Mammal", "Artificial"])]

    def normalize_label(l):
        if pd.isna(l):
            return l
        s = str(l).strip().replace("/", " ")
        return " ".join(s.split()[:2])

    combined["label"] = combined["label"].apply(normalize_label)
    combined = combined[combined["label"].str.split().str.len() == 2]
    combined = combined[
        (combined["xmin"] != 0) & (combined["ymin"] != 0)
        & (combined["xmax"] != 0) & (combined["ymax"] != 0)
        & (combined["xmin"] >= 0) & (combined["ymin"] >= 0)
        & (combined["xmax"] >= 0) & (combined["ymax"] >= 0)
    ]
    n_after_filters = _count_label(combined, LABEL)
    steps.append(("After filters (>25/class, 2-word, valid boxes)", n_after_filters))
    print(f"4) After filters (>25/class, 2-word, no empty/neg boxes): {n_after_filters}")

    # Class balance (gentle_class_balance)
    counts = combined.groupby("label").size()
    median_count = int(counts.median())
    cap = max(median_count, int(median_count * 3.0))
    balanced = []
    for label in combined["label"].unique():
        class_df = combined[combined["label"] == label]
        if len(class_df) <= cap:
            balanced.append(class_df)
        else:
            balanced.append(class_df.sample(n=cap, random_state=42))
    combined_bal = pd.concat(balanced, ignore_index=True)
    n_after_balance = _count_label(combined_bal, LABEL)
    steps.append(("After class balance (cap 3x median)", n_after_balance))
    print(f"5) After class balance (cap 3x median): {n_after_balance}")

    # 6) Split by parent image (same as train_test_split_by_image)
    def crop_path_to_parent_stem(crop_path):
        import re
        basename = os.path.basename(str(crop_path))
        m = re.match(r"^(.+)_\d+\.(png|PNG|jpg|JPG|jpeg|JPEG)$", basename)
        return basename if m is None else m.group(1)

    combined_bal = combined_bal.copy()
    combined_bal["parent_image"] = combined_bal["image_path"].map(crop_path_to_parent_stem)
    unique_parents = combined_bal["parent_image"].unique()
    rng = np.random.default_rng(42)
    n_test_parents = max(1, int(len(unique_parents) * 0.1))
    test_parents = set(rng.choice(unique_parents, size=n_test_parents, replace=False))
    train_parents = set(unique_parents) - test_parents
    train_df = combined_bal[combined_bal["parent_image"].isin(train_parents)].drop(columns=["parent_image"])
    test_df = combined_bal[combined_bal["parent_image"].isin(test_parents)].drop(columns=["parent_image"])
    # Drop classes with < 5 test or < 1 train
    kept = []
    for label in combined_bal["label"].unique():
        train_count = (train_df["label"] == label).sum()
        test_count = (test_df["label"] == label).sum()
        if test_count >= 5 and train_count >= 1:
            kept.append(label)
    train_df = train_df[train_df["label"].isin(kept)]
    test_df = test_df[test_df["label"].isin(kept)]
    n_train = _count_label(train_df, LABEL)
    n_val = _count_label(test_df, LABEL)
    steps.append(("Classification train (final)", n_train))
    steps.append(("Classification validation (final)", n_val))
    print(f"6) After split-by-image (kept classes with ≥5 val, ≥1 train):")
    print(f"   Train: {n_train}  |  Validation: {n_val}")
    if LABEL not in kept:
        print(f"   Tursiops truncatus was DROPPED (insufficient train or test parents).")
    else:
        print(f"   Tursiops truncatus was KEPT.")

    print("=" * 60)
    print("Summary (Tursiops truncatus):")
    for name, count in steps:
        print(f"  {count:5d}  {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
