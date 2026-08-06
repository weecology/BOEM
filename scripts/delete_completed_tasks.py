#!/usr/bin/env python3
"""Download completed Label Studio tasks, then delete exactly those tasks.

This is the destructive counterpart to scripts/download_annotations.py. Deleting a task
removes the only server-side copy of its annotation, so the order here matters:

1. Snapshot the ids of every labeled task that actually carries an annotation.
2. Download completed tasks to CSV (a superset of the snapshot — a task labeled during
   the run is downloaded but not deleted, so it simply survives to the next round).
3. Mirror the CSVs into annotations_backup/ and refuse to go further if any drift
   remains, because after step 4 that mirror is the backup.
4. Delete the snapshotted ids only.

Unlabeled tasks are never touched — they are the pending annotation queue.

Usage:
    uv run python scripts/delete_completed_tasks.py --dry-run   # counts only
    uv run python scripts/delete_completed_tasks.py
    uv run python scripts/delete_completed_tasks.py --instances review
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from src.label_studio import connect_to_label_studio, download_completed_tasks, get_api_key

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "boem_conf" / "annotation" / "label_studio.yaml"
BACKUP_SCRIPT = REPO_ROOT / "scripts" / "backup_annotations.py"


def sync_backup():
    """Mirror the annotation tree into annotations_backup/ and prove it is in sync."""
    subprocess.run([sys.executable, str(BACKUP_SCRIPT)], check=True)
    subprocess.run([sys.executable, str(BACKUP_SCRIPT), "--check"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["train", "validation", "review"])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted.")
    args = parser.parse_args()

    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    cfg = OmegaConf.load(CONFIG_PATH).label_studio

    pending = {}
    for instance_name in args.instances:
        instance = cfg.instances[instance_name]
        print(f"\n=== {instance_name}: {instance.project_name}")
        project = connect_to_label_studio(url=cfg.url, project_name=instance.project_name)

        tasks = project.get_labeled_tasks()
        task_ids = [t["id"] for t in tasks if t.get("annotations")]
        print(f"{instance_name}: {len(task_ids)} labeled tasks with annotations "
              f"({len(tasks) - len(task_ids)} labeled-but-empty, left in place)")

        annotations = download_completed_tasks(label_studio_project=project, csv_dir=instance.csv_dir)
        if annotations is None:
            print(f"{instance_name}: nothing downloaded, skipping delete")
            continue
        print(f"{instance_name}: downloaded {annotations.shape[0]} rows across "
              f"{annotations.image_path.nunique()} images")
        pending[instance_name] = (project, task_ids)

    if args.dry_run:
        print("\n--dry-run: deleting nothing.")
        return

    print("\nSyncing annotations_backup/ before deleting...")
    sync_backup()

    for instance_name, (project, task_ids) in pending.items():
        for task_id in task_ids:
            project.delete_task(task_id)
        print(f"{instance_name}: deleted {len(task_ids)} tasks")

    print("\nDeleted from the server. Commit annotations_backup/ now — it is the backup.")


if __name__ == "__main__":
    main()
