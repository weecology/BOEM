#!/usr/bin/env python3
"""Download completed Label Studio annotations without running the pipeline.

Writes one timestamped CSV per flight under each instance's csv_dir, e.g.
/blue/ewhite/b.weinstein/BOEM/annotations/validation/<flight_name>/<timestamp>.csv

Tasks are left in place on the Label Studio server; unlike the pipeline path
(src/label_studio.check_for_new_annotations) this never calls
delete_completed_tasks, so a rerun re-downloads the same tasks.

Usage:
    uv run python scripts/download_annotations.py
    uv run python scripts/download_annotations.py --instances train validation
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from src.label_studio import connect_to_label_studio, download_completed_tasks, get_api_key

CONFIG_PATH = Path(__file__).resolve().parent.parent / "boem_conf" / "annotation" / "label_studio.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["train", "validation", "review"])
    args = parser.parse_args()

    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    cfg = OmegaConf.load(CONFIG_PATH).label_studio

    for instance_name in args.instances:
        instance = cfg.instances[instance_name]
        print(f"\n=== {instance_name}: {instance.project_name} -> {instance.csv_dir}")
        project = connect_to_label_studio(url=cfg.url, project_name=instance.project_name)
        annotations = download_completed_tasks(label_studio_project=project, csv_dir=instance.csv_dir)
        if annotations is not None:
            print(f"{instance_name}: {annotations.shape[0]} rows across "
                  f"{annotations.image_path.nunique()} images, "
                  f"{annotations.flight_name.nunique()} flights")


if __name__ == "__main__":
    main()
