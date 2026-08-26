#!/usr/bin/env python3
"""Delete unlabeled Label Studio tasks belonging to given flights.

Unlabeled tasks are the pending annotation queue: nobody has looked at them, so
deleting one destroys no human work, only the queue entry. Use this to retire a
dataset the project has stopped annotating. Completed tasks are never touched —
`scripts/delete_completed_tasks.py` owns those, and it downloads before deleting
because a completed task holds the only server-side copy of its annotation.

Flights are matched case-insensitively against each task's `data.flight_name`,
because the same NEAQ camera directory is spelled several ways upstream
("Whale camera edited" vs "Whale Camera Edited").

The images themselves are left alone, on both /blue and the Label Studio server's
input folder, so a retired dataset can be re-uploaded later.

Usage:
    uv run python scripts/delete_unlabeled_tasks.py --flights "whale camera edited" --dry-run
    uv run python scripts/delete_unlabeled_tasks.py --flights "belly camera edited" "whale camera edited"
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from src.label_studio import connect_to_label_studio, get_api_key

CONFIG_PATH = Path(__file__).resolve().parent.parent / "boem_conf" / "annotation" / "label_studio.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", nargs="+", required=True,
                        help="flight_name values to retire, matched case-insensitively")
    parser.add_argument("--instances", nargs="+", default=["train", "validation", "review"])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted.")
    args = parser.parse_args()

    targets = {f.lower() for f in args.flights}
    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    cfg = OmegaConf.load(CONFIG_PATH).label_studio

    for instance_name in args.instances:
        instance = cfg.instances[instance_name]
        print(f"\n=== {instance_name}: {instance.project_name}")
        project = connect_to_label_studio(url=cfg.url, project_name=instance.project_name)

        # Re-check `annotations` rather than trusting get_unlabeled_tasks alone: a task
        # labeled since the listing must survive, since deleting it would drop the
        # annotation without ever downloading it.
        doomed = [t for t in project.get_unlabeled_tasks()
                  if t["data"].get("flight_name", "").lower() in targets and not t.get("annotations")]

        for flight, n in sorted(Counter(t["data"]["flight_name"] for t in doomed).items()):
            print(f"  {n:>6}  {flight}")
        if not doomed:
            print("  no matching unlabeled tasks")
            continue

        if args.dry_run:
            print(f"  --dry-run: would delete {len(doomed)} tasks")
            continue

        for task in doomed:
            project.delete_task(task["id"])
        print(f"  deleted {len(doomed)} tasks")


if __name__ == "__main__":
    main()
