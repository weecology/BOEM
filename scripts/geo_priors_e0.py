"""E0: frame location index + coverage of a classifier's train/val crops by class x region.

Usage: uv run python scripts/geo_priors_e0.py [comet_id]
Writes to /blue/ewhite/b.weinstein/BOEM/geo_priors/.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geo_priors import SURVEY_REGION, build_frame_index, crop_parent, frame_date

OUT = Path("/blue/ewhite/b.weinstein/BOEM/geo_priors")
CROPS = Path("/blue/ewhite/b.weinstein/BOEM/training/classification/crops")
ANNOTATION_CSVS = Path("/blue/ewhite/b.weinstein/BOEM/training/crops")
comet_id = sys.argv[1] if len(sys.argv) > 1 else "a3dc30a085f5442393736ecd96b564c5"

OUT.mkdir(exist_ok=True)
index_path = OUT / "frame_index.csv"
if index_path.exists():
    index = pd.read_csv(index_path, parse_dates=["date"])
else:
    index = build_frame_index()
    index.to_csv(index_path, index=False)
print(f"frame index: {len(index):,} frames", index.source.value_counts().to_dict())
print(index.groupby(["region", "subregion"]).size().to_string())

crops = pd.DataFrame(
    [(split, p.parent.name, p.name)
     for split in ("train", "val")
     for p in (CROPS / split / "buffer_30" / comet_id).glob("*/*.png")],
    columns=["split", "label", "filename"],
)
crops["basename"] = crops.filename.map(crop_parent)
crops = crops.merge(index.drop(columns="date"), on="basename", how="left")
crops["date"] = crops.basename.map(frame_date)

# Old-format frames: region from the survey named in the annotation job_name.
unmatched = crops.loc[crops.lat.isna(), "basename"].unique()
def _job_name(basename: str) -> str | None:
    """job_name of the frame's annotation CSV; some CSVs (non-UBFAI sources) lack the column."""
    path = ANNOTATION_CSVS / f"{basename}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, nrows=1).get("job_name", pd.Series([None])).iloc[0]


surveys = {b: _job_name(b) for b in unmatched}
crops["survey"] = crops.basename.map(surveys).str.extract(r"-(nyserda|boemy\d)-")[0]
for survey, (region, sub) in SURVEY_REGION.items():
    hit = crops.lat.isna() & (crops.survey == survey)
    crops.loc[hit, ["region", "subregion", "source"]] = [region, sub, f"survey:{survey}"]
crops["source"] = crops.source.fillna(
    "survey:" + crops.survey.where(crops.survey.notna(), "unknown")
)

crops.to_csv(OUT / f"e0_crops_{comet_id[:8]}.csv", index=False)
print("\ncrops by split x source")
print(pd.crosstab(crops.source, crops.split, margins=True).to_string())
ct = pd.crosstab(crops.label, crops.region.fillna("none"))
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
ct.to_csv(OUT / f"e0_class_by_region_{comet_id[:8]}.csv")
print("\nclass x region (train+val)")
print(ct.to_string())
print("\nregion x month (crops with coordinates)")
print(pd.crosstab(crops.region, pd.to_datetime(crops.date).dt.month).to_string())
