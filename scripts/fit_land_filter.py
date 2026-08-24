"""Download land/water annotations and fit the logistic regression land filter.

    uv run python scripts/fit_land_filter.py

The repo's `download_completed_tasks` parses bounding boxes, so it cannot read this
project's whole-frame Choices annotations; this pulls them directly instead. Labels
are joined back to the mined features by frame basename.

Errors here are asymmetric -- dropping a water frame can lose an animal, while letting
a land frame through only preserves the status quo -- so the operating point is chosen
by recall at a small water-loss budget, not by accuracy. It is deliberately not "lose
no water frame at all": that criterion is a max() over the water set, so one hard
negative sets it, and on the 2026-08-24 labels it collapsed land recall to 12%.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import label_studio as ls_mod
from src.label_studio import get_api_key
from src.land_filter import (MAX_BG_FRAC, MAX_EDGE_RATIO, MIN_CHROMA,
                             MIN_STRUCT)

OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")
PROJECT_NAME = "Bureau of Ocean Energy Management - Land Screen"
FEATURES = ["struct", "fine_edge", "chroma", "bg_frac"]

# Share of water frames we are willing to drop. Non-zero on purpose -- see the
# threshold selection in main() for why zero is not a usable target.
WATER_LOSS_BUDGET = 0.03


def fetch_labels(project):
    """One row per annotated task: basename + chosen class."""
    rows = []
    for task in project.get_tasks():
        for ann in task.get("annotations", []):
            choices = [r["value"]["choices"][0]
                       for r in ann.get("result", []) if r.get("from_name") == "surface"]
            if choices:
                rows.append({"image": os.path.basename(task["data"]["image"]),
                             "label": choices[0]})
    return pd.DataFrame(rows).drop_duplicates("image")


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")

    project = ls_mod.connect_to_label_studio(
        url=cfg.annotation.label_studio.url, project_name=PROJECT_NAME)
    labels = fetch_labels(project)
    labels.to_csv(OUT_DIR / "annotations.csv", index=False)
    print(f"{len(labels)} annotated frames\n{labels.label.value_counts().to_string()}\n")

    feats = pd.read_csv(OUT_DIR / "manifest.csv")
    d = feats.merge(labels, on="image")
    # Mixed frames contain land and so still generate false positives, but they also
    # contain water worth searching. Keep them out of the fit and decide separately.
    d = d[d.label.isin(["Land", "Water"])]
    X, y = d[FEATURES].values, (d.label == "Land").astype(int).values
    print(f"fitting on {len(d)} frames ({y.sum()} land, {len(y) - y.sum()} water)")

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    print(f"cross-validated ROC-AUC {roc_auc_score(y, prob):.3f}")

    # Highest-recall threshold whose water loss stays inside the budget. An earlier
    # version demanded ZERO water loss, which is a max() over the water frames and so
    # is set by a single frame: the turbid shallows in JPG_20260710_155800, where dark
    # seagrass patches on a bright bottom are genuinely land-like. That one frame
    # dragged the operating point to 0.93 and land recall to 12%, i.e. the filter
    # stopped doing anything. A small budget buys back most of the recall.
    grid = np.arange(0.01, 1.0, 0.005)
    affordable = [t for t in grid if (prob[y == 0] > t).mean() <= WATER_LOSS_BUDGET]
    thresh = float(min(affordable)) if affordable else 1.0
    print(f"operating point {thresh:.3f} at a {WATER_LOSS_BUDGET:.0%} water-loss budget: "
          f"land recall {(prob[y == 1] > thresh).mean():.1%}, "
          f"water lost {(prob[y == 0] > thresh).sum()}/{(y == 0).sum()}")

    print("\n  thresh  land recall   water lost")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"  {t:6.2f}  {(prob[y == 1] > t).mean():10.1%}   "
              f"{(prob[y == 0] > t).mean():5.1%} ({(prob[y == 0] > t).sum()}/{(y == 0).sum()})")

    # The hand-tuned conjunction this model replaces, scored on the same frames.
    rule = ((d.struct > MIN_STRUCT) & (d.fine_edge < MAX_EDGE_RATIO * d.struct)
            & (d.chroma > MIN_CHROMA) & (d.bg_frac < MAX_BG_FRAC)).values
    print(f"\n  hand-tuned rule baseline: land recall {rule[y == 1].mean():.1%}, "
          f"water lost {rule[y == 0].mean():.1%} ({rule[y == 0].sum()}/{(y == 0).sum()})")

    model.fit(X, y)
    lr = model.named_steps["logisticregression"]
    coefs = dict(zip(FEATURES, lr.coef_[0].round(3)))
    out = {"features": FEATURES, "coef": lr.coef_[0].tolist(),
           "intercept": float(lr.intercept_[0]),
           "scaler_mean": model.named_steps["standardscaler"].mean_.tolist(),
           "scaler_scale": model.named_steps["standardscaler"].scale_.tolist(),
           "threshold": thresh, "water_loss_budget": WATER_LOSS_BUDGET,
           "cv_auc": float(roc_auc_score(y, prob)),
           "cv_land_recall": float((prob[y == 1] > thresh).mean()),
           "n_train": int(len(d))}
    (OUT_DIR / "land_model.json").write_text(json.dumps(out, indent=2))
    print(f"\nstandardised coefficients: {coefs}")
    print(f"wrote {OUT_DIR / 'land_model.json'}")


if __name__ == "__main__":
    main()
