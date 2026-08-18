"""Download land/water annotations and fit the logistic regression land filter.

    uv run python scripts/fit_land_filter.py

The repo's `download_completed_tasks` parses bounding boxes, so it cannot read this
project's whole-frame Choices annotations; this pulls them directly instead. Labels
are joined back to the mined features by frame basename.

Errors here are asymmetric -- dropping a water frame can lose an animal, while letting
a land frame through only preserves the status quo -- so the reported operating point
is the highest-recall threshold that loses no water frame in cross-validation, not the
one that maximises accuracy.
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import label_studio as ls_mod
from src.label_studio import get_api_key

OUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen")
PROJECT_NAME = "Bureau of Ocean Energy Management - Land Screen"
FEATURES = ["struct", "fine_edge", "chroma", "bg_frac"]


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

    # Highest-recall threshold that misclassifies no water frame.
    water_max = prob[y == 0].max()
    safe = float(np.nextafter(water_max, 1.0))
    print(f"\ncross-validated: land recall {(prob[y == 1] > safe).mean():.1%} "
          f"at zero water loss (threshold {safe:.3f})")
    for t in (0.5, 0.7, 0.9):
        print(f"  threshold {t:.2f}: land recall {(prob[y == 1] > t).mean():.1%}, "
              f"water lost {(prob[y == 0] > t).mean():.1%}")

    model.fit(X, y)
    lr = model.named_steps["logisticregression"]
    coefs = dict(zip(FEATURES, lr.coef_[0].round(3)))
    out = {"features": FEATURES, "coef": lr.coef_[0].tolist(),
           "intercept": float(lr.intercept_[0]),
           "scaler_mean": model.named_steps["standardscaler"].mean_.tolist(),
           "scaler_scale": model.named_steps["standardscaler"].scale_.tolist(),
           "threshold": safe, "n_train": int(len(d))}
    (OUT_DIR / "land_model.json").write_text(json.dumps(out, indent=2))
    print(f"\nstandardised coefficients: {coefs}")
    print(f"wrote {OUT_DIR / 'land_model.json'}")


if __name__ == "__main__":
    main()
