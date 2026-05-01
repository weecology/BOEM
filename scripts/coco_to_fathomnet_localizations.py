"""Convert COCO detection JSON (bbox xywh) to FathomNet localization CSV.

FathomNet expects columns: concept, image (public URL), x, y, width, height
with origin top-left (+Y down). COCO ``bbox`` uses the same convention.

Imagery from the USGS ScienceBase release (DOI: 10.5066/P9CBZQV1) is served
from UF Orange public web space:

  /orange/ewhite/web/public/BOEM/usgs/images/

which maps to:
  https://data.rc.ufl.edu/pub/ewhite/BOEM/usgs/images/

Annotations are extracted from 03_Annotations.zip into:
  data/usgs_P9CBZQV1/annotations_extracted/annotations/{train,eval,test}/

See: https://www.fathomnet.org/post/how-to-submit-localized-image-annotations-to-the-fathomnet-database

Example (10-image sample from test split):
  uv run python scripts/coco_to_fathomnet_localizations.py \\
    --coco-json data/usgs_P9CBZQV1/annotations_extracted/annotations/test/test_gt.json \\
    --image-base-url https://data.rc.ufl.edu/pub/ewhite/BOEM/usgs/images/ \\
    --output-csv output/usgs_fathomnet_sample.csv \\
    --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument(
        "--image-base-url",
        required=True,
        help="Base URL ending with /; file_name from COCO is appended (urljoin).",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N unique images (useful for sampling).",
    )
    args = parser.parse_args()

    coco_path: Path = args.coco_json.expanduser().resolve()
    out_csv: Path = args.output_csv.expanduser().resolve()
    base: str = args.image_base_url
    if not base.endswith("/"):
        base = base + "/"

    data = load_coco(coco_path)
    id_to_name = {int(c["id"]): str(c.get("name", "")) for c in data.get("categories", [])}
    all_images = data.get("images", [])
    if args.limit is not None:
        kept_ids = {int(im["id"]) for im in all_images[: args.limit]}
    else:
        kept_ids = None
    id_to_image = {int(im["id"]): im for im in all_images}
    rows: list[dict[str, Any]] = []
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        if kept_ids is not None and image_id not in kept_ids:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        image = id_to_image.get(image_id)
        if image is None:
            continue
        file_name = str(image["file_name"])
        url = urljoin(base, file_name.split("/")[-1])
        cat_id = int(ann.get("category_id", 0))
        concept = id_to_name.get(cat_id, str(cat_id))
        rows.append(
            {
                "concept": concept,
                "image": url,
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(w)),
                "height": int(round(h)),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concept", "image", "x", "y", "width", "height"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
