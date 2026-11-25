"""E2 input: GBIF record counts per class x sub-region x month (a coarse presence prior).

One facet request per (class, sub-region); counts are records 2000+ with coordinates
inside the sub-region's bounding box. Writes geo_priors/gbif_counts.csv. Needs network.
Boxes only approximate E0's `assign_region` partition (land inside the box is counted).
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("/blue/ewhite/b.weinstein/BOEM/geo_priors")
# (lat_min, lat_max, lon_min, lon_max)
BOXES = {
    "AtlanticNorth": (41, 47, -76, -60),
    "MidAtlantic": (35, 41, -80, -65),
    "SouthAtlantic": (24, 35, -81, -70),
    "GulfEast": (24, 31, -89, -81),
    "GulfWest": (24, 31, -100, -89),
}
classes = (OUT / "eval_classes_0139af8c.txt").read_text().split("\n") if (OUT / "eval_classes_0139af8c.txt").exists() \
    else Path("/blue/ewhite/b.weinstein/BOEM/classifier_confusion_0139af8c_classes.txt").read_text().split("\n")
session = requests.Session()


def facet_months(**params) -> dict[int, int]:
    while True:  # GBIF rate-limits with 429; back off and retry
        r = session.get("https://api.gbif.org/v1/occurrence/search", timeout=60, params={
            "limit": 0, "facet": "month", "facetLimit": 12, "year": "2000,2026",
            "hasCoordinate": "true", "hasGeospatialIssue": "false", **params})
        if r.status_code != 429:
            break
        time.sleep(10)
    r.raise_for_status()
    time.sleep(0.5)
    return {int(c["name"]): c["count"] for c in r.json()["facets"][0]["counts"]}


rows = []
for name in classes + ["__effort__"]:
    taxon = 44 if name == "__effort__" else session.get(
        "https://api.gbif.org/v1/species/match", timeout=60,
        params={"name": "Cheloniidae" if name.endswith(" sp") else name}).json().get("usageKey")
    key = {"taxonKey": taxon}
    if key["taxonKey"] is None:
        print("no GBIF match", name)
        continue
    for region, (la0, la1, lo0, lo1) in BOXES.items():
        counts = facet_months(**key, decimalLatitude=f"{la0},{la1}", decimalLongitude=f"{lo0},{lo1}")
        rows += [(name, region, month, counts.get(month, 0)) for month in range(1, 13)]
    print(name, key, flush=True)
pd.DataFrame(rows, columns=["label", "subregion", "month", "n"]).to_csv(OUT / "gbif_counts.csv", index=False)
