"""Frame-level location index and coarse region assignment for species geographic priors.

See SPECIES_GEO_PRIORS_PLAN.md. Two frame-naming schemes exist:

* New (2020+): ``C{cam}_L{line}_F{frame}_T{YYYYMMDD_HHMMSS}_{ms}``. Coordinates come
  from the per-flight captures CSVs (``Basename`` column), falling back to the UBFAI
  ``metadata_aws`` exports (``bname_parent``, X=lon, Y=lat).
* Old (2016-2020 USGS/UBFAI): ``{n}-{MMDDYYHHMMSS}{d}-CAM{k}``. No coordinates exist
  anywhere on disk. Date is parsed from the name; region comes from the survey in the
  annotation ``job_name`` (``nyserda`` = New York Bight) where known.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

CAPTURES_DIR = Path("/blue/ewhite/b.weinstein/BOEM/metadata_aflight_csvs")
AWS_DIR = Path("/blue/ewhite/b.weinstein/BOEM/UBFAI Data Collection/metadata_aws")

_NEW_FRAME_RE = re.compile(r"^C\d+_L\d+_F\d+_T(\d{8})_\d{6}")
_OLD_FRAME_RE = re.compile(r"^\d+-(\d{2})(\d{2})(\d{2})\d+-CAM\d+$")
# Training crops are "{frame}_{box_index}_{global_index}.png".
_CROP_SUFFIX_RE = re.compile(r"(_\d+){2}$")

# Surveys whose old-format frames have a known region but no coordinates.
SURVEY_REGION = {"nyserda": ("Atlantic", "NewYorkBight")}


def crop_parent(crop_name: str) -> str:
    """Frame basename for a training crop filename."""
    return _CROP_SUFFIX_RE.sub("", Path(crop_name).stem)


def frame_date(basename: str) -> pd.Timestamp | None:
    """Capture date encoded in either frame-naming scheme."""
    if m := _NEW_FRAME_RE.match(basename):
        return pd.Timestamp(m.group(1))
    if m := _OLD_FRAME_RE.match(basename):
        mm, dd, yy = m.groups()
        return pd.Timestamp(f"20{yy}-{mm}-{dd}")
    return None


def assign_region(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    """Coarse region and sub-region from coordinates (decimal degrees).

    Deliberately broad: the prior is meant to separate oceans and coasts, not
    10 km cells. Sub-regions are the finer level of the same partition.
    """
    gulf = (lat < 31) & (lon < -81)
    west = lon < -100
    lakes = (lat > 41) & (lon < -82) & ~west
    region = np.select([west, lakes, gulf], ["InlandWest", "GreatLakes", "Gulf"], "Atlantic")
    sub = np.select(
        [region == "Gulf", region == "Atlantic"],
        [np.where(lon < -89, "GulfWest", "GulfEast"),
         np.select([lat >= 41, lat >= 35], ["AtlanticNorth", "MidAtlantic"], "SouthAtlantic")],
        region,
    )
    return pd.DataFrame({"region": region, "subregion": sub}, index=lat.index)


def _captures_index() -> pd.DataFrame:
    frames = []
    for path in sorted(CAPTURES_DIR.glob("*_captures.csv")):
        flight = path.name.removesuffix("_captures.csv")
        flights_csv = CAPTURES_DIR / f"{flight}_flights.csv"
        asset = pd.read_csv(flights_csv, usecols=["Asset"]).Asset.iloc[0] if flights_csv.exists() else None
        df = pd.read_csv(path, usecols=["Basename", "Lat", "Lon"]).dropna()
        frames.append(df.assign(flight=flight, asset=asset))
    df = pd.concat(frames).rename(columns={"Basename": "basename", "Lat": "lat", "Lon": "lon"})
    return df.assign(source="captures")


def _aws_index() -> pd.DataFrame:
    df = pd.concat(
        pd.read_csv(p, usecols=["bname_parent", "X", "Y"]) for p in sorted(AWS_DIR.glob("*_metadata.csv"))
    )
    df = df.rename(columns={"bname_parent": "basename", "X": "lon", "Y": "lat"}).dropna()
    return df.assign(flight=None, asset=None, source="metadata_aws")


def build_frame_index() -> pd.DataFrame:
    """One row per frame basename: lat, lon, date, flight, asset, region, subregion, source.

    Captures CSVs win over metadata_aws when both have a frame. Frames at (0, 0) are
    dropped: that is a missing-GPS sentinel, not a location.
    """
    df = pd.concat([_captures_index(), _aws_index()])
    df["basename"] = df.basename.astype(str)
    df = df[(df.lat != 0) | (df.lon != 0)].drop_duplicates("basename", keep="first")
    df["date"] = pd.to_datetime(df.basename.str.extract(r"_T(\d{8})_")[0], format="%Y%m%d")
    return pd.concat([df.reset_index(drop=True), assign_region(df.lat, df.lon).reset_index(drop=True)], axis=1)
