"""Download assets from the USGS ScienceBase data release for Ke et al. aerial wildlife ML data.

Catalog page: https://data.usgs.gov/datacatalog/data/USGS:658da7c1d34e3265ab14bb00
DOI: https://doi.org/10.5066/P9CBZQV1

The release bundles Lake Michigan and Nantucket Shoals (Atlantic) imagery and COCO-format
annotations. Place keywords on the catalog include Cape Cod; the bundled FGDC metadata
describes geographic coverage as "Lake Michigan and Nantucket Shoals". COCO ``file_name``
values do not encode region, so splitting "Cape Cod only" requires external flight metadata
from the data provider if you need a strict geographic subset.

Large imagery (``02_Imagery.zip``, ~18 GB) is served via ScienceBase/S3 and may require
using the request-download flow in a browser when direct ``curl`` receives HTML instead
of bytes. This script prints the request URL when you pass ``--include-imagery``.

Examples:
  uv run python scripts/download_usgs_aerial_wildlife_release.py --out-dir ./data/usgs_P9CBZQV1
  uv run python scripts/download_usgs_aerial_wildlife_release.py --out-dir ./data/usgs_P9CBZQV1 --peek-schema
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

SCIENCEBASE_ITEM_ID = "658da7c1d34e3265ab14bb00"
ITEM_JSON_URL = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM_ID}?format=json"


def fetch_item_json() -> dict[str, Any]:
    with urllib.request.urlopen(ITEM_JSON_URL, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def file_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    return list(item.get("files") or [])


def pick_download_uri(entry: dict[str, Any]) -> str:
    uri = entry.get("downloadUri") or entry.get("url")
    if not uri:
        raise ValueError(f"No download URI for file: {entry.get('name')}")
    return str(uri)


def human_size(n: int | None) -> str:
    if n is None:
        return "unknown"
    units = ("B", "KB", "MB", "GB", "TB")
    v = float(n)
    u = 0
    while v >= 1024 and u < len(units) - 1:
        v /= 1024.0
        u += 1
    if u == 0:
        return f"{int(n)} {units[u]}"
    return f"{v:.1f} {units[u]}"


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as resp:
        body = resp.read()
    if len(body) < 5000 and body.lstrip().startswith(b"<"):
        raise RuntimeError(
            f"Download from {url} returned HTML ({len(body)} bytes), not a file. "
            "Try the ScienceBase request-download page for S3-backed assets."
        )
    dest.write_bytes(body)


def peek_coco_schema(annotations_zip: Path) -> None:
    with zipfile.ZipFile(annotations_zip) as zf:
        names = [n for n in zf.namelist() if n.endswith("train/train.json")]
        if not names:
            print("No annotations/train/train.json inside zip; members:", zf.namelist()[:30])
            return
        inner = names[0]
        with zf.open(inner) as f:
            data = json.load(f)
    imgs = data.get("images") or []
    anns = data.get("annotations") or []
    cats = data.get("categories") or []
    print("COCO keys:", sorted(data.keys()))
    print("n_images:", len(imgs), "n_annotations:", len(anns), "n_categories:", len(cats))
    if imgs:
        print("sample image:", {k: imgs[0].get(k) for k in ("id", "file_name", "width", "height")})
    if anns:
        a0 = anns[0]
        print(
            "sample annotation keys:",
            sorted(a0.keys()),
        )
        print("sample bbox (xywh):", a0.get("bbox"), "image_id:", a0.get("image_id"))
    if cats:
        print("sample category:", cats[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/usgs_P9CBZQV1"),
        help="Directory to write downloaded zips and metadata.",
    )
    parser.add_argument(
        "--include-codebase",
        action="store_true",
        help="Also download 01_CodeBase.zip (~8 MB).",
    )
    parser.add_argument(
        "--include-imagery",
        action="store_true",
        help="Attempt 02_Imagery.zip (~18 GB). Often requires ScienceBase browser download.",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Do not download 03_Annotations.zip.",
    )
    parser.add_argument(
        "--peek-schema",
        action="store_true",
        help="After annotations zip exists, print COCO schema summary from train/train.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List resolved URLs and sizes without downloading.",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.expanduser().resolve()

    item = fetch_item_json()
    by_name = {str(e.get("name")): e for e in file_entries(item)}

    plan: list[tuple[str, Path]] = []
    if not args.skip_annotations:
        e = by_name.get("03_Annotations.zip")
        if not e:
            print("Item JSON missing 03_Annotations.zip", file=sys.stderr)
            sys.exit(1)
        plan.append(("03_Annotations.zip", out_dir / "03_Annotations.zip"))
    if args.include_codebase:
        e = by_name.get("01_CodeBase.zip")
        if e:
            plan.append(("01_CodeBase.zip", out_dir / "01_CodeBase.zip"))
    if args.include_imagery:
        e = by_name.get("02_Imagery.zip")
        if e:
            plan.append(("02_Imagery.zip", out_dir / "02_Imagery.zip"))
            s3_page = e.get("s3DownloadRequestPageUri")
            if s3_page:
                print("If direct download fails, request the large imagery zip via:", s3_page)

    meta = by_name.get("Metadata for Deep Learning Wildlife Detection Model.xml")
    if meta:
        plan.append(
            (
                "Metadata for Deep Learning Wildlife Detection Model.xml",
                out_dir / "metadata_fgdc.xml",
            )
        )

    for logical, dest in plan:
        entry = by_name.get(logical) or next(
            (e for e in file_entries(item) if e.get("name") == logical),
            None,
        )
        if entry is None and logical == "metadata_fgdc.xml":
            continue
        name = str(entry.get("name")) if entry else logical
        size = int(entry.get("size") or 0) if entry else 0
        url = pick_download_uri(entry) if entry else ""
        print(f"{name} -> {dest} ({human_size(size)})")
        print(f"  {url}")
        if args.dry_run:
            continue
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip (exists): {dest}")
            continue
        if name == "02_Imagery.zip":
            print("  Warning: very large file; ensure disk space and stable network.")
        download_url(url, dest)
        print(f"  wrote {dest} ({human_size(dest.stat().st_size)})")

    ann_zip = out_dir / "03_Annotations.zip"
    if args.peek_schema:
        if not ann_zip.is_file():
            print("--peek-schema: annotations zip not found at", ann_zip, file=sys.stderr)
            sys.exit(1)
        peek_coco_schema(ann_zip)


if __name__ == "__main__":
    main()
