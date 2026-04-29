"""Convert COCO detection JSON (bbox xywh) to FathomNet localization CSV.

FathomNet expects columns including: concept, image (public URL), x, y, width, height
with origin top-left (+Y down). COCO ``bbox`` uses the same convention.

Use on HiPerGator (or any host) after imagery is reachable at a stable HTTPS URL, for
example under UF Orange public web space:

  /orange/ewhite/web/public/...

so each patch URL looks like:
  https://<your-public-host>/.../0400AGL_P1_20170224_102133_631_136760_56257_0.png

See: https://www.fathomnet.org/post/how-to-submit-localized-image-annotations-to-the-fathomnet-database

Example:
  uv run python scripts/coco_to_fathomnet_localizations.py \\
    --coco-json ./data/usgs_P9CBZQV1/annotations_extracted/train/train.json \\
    --image-base-url https://example.rc.ufl.edu/public/boem/usgs_patches/ \\
    --output-csv ./fathomnet_localizations.csv
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
    args = parser.parse_args()

    coco_path: Path = args.coco_json.expanduser().resolve()
    out_csv: Path = args.output_csv.expanduser().resolve()
    base: str = args.image_base_url
    if not base.endswith("/"):
        base = base + "/"

    data = load_coco(coco_path)
    id_to_name = {int(c["id"]): str(c.get("name", "")) for c in data.get("categories", [])}
    id_to_image = {int(im["id"]): im for im in data.get("images", [])}
    rows: list[dict[str, Any]] = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        image_id = int(ann["image_id"])
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
