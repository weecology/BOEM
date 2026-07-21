#!/usr/bin/env python3
"""
Copy all tasks (with annotations) from one Label Studio project to another.

Source:  BOEM - Full Flight - JPG_20260202_141900
Target:  Bureau of Ocean Energy Management - Review

The source project is NOT modified. Tasks are exported in Label Studio JSON format
and imported into the target project.
"""
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from label_studio_sdk import Client

LABEL_STUDIO_URL = "https://labelstudio.naturecast.org/"
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY")
if not API_KEY:
    # Fall back to config file
    config_path = os.path.join(os.path.dirname(__file__), "..", ".label_studio.config")
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if line.startswith("api_key"):
                    API_KEY = line.split("=", 1)[1].strip()
                    break

if not API_KEY:
    print("ERROR: Could not find LABEL_STUDIO_API_KEY")
    sys.exit(1)

SOURCE_PROJECT_NAME = "BOEM - Full Flight - JPG_20260202_141900"
TARGET_PROJECT_NAME = "Bureau of Ocean Energy Management - Review"

def main():
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    print("Connected to Label Studio.")

    projects = ls.list_projects()
    project_map = {p.get_params()["title"]: p for p in projects}

    # Locate source project
    if SOURCE_PROJECT_NAME not in project_map:
        print(f"ERROR: Source project not found: '{SOURCE_PROJECT_NAME}'")
        print("Available projects:")
        for name in project_map:
            print(f"  {name}")
        sys.exit(1)

    # Locate target project
    if TARGET_PROJECT_NAME not in project_map:
        print(f"ERROR: Target project not found: '{TARGET_PROJECT_NAME}'")
        print("Available projects:")
        for name in project_map:
            print(f"  {name}")
        sys.exit(1)

    source_project = project_map[SOURCE_PROJECT_NAME]
    target_project = project_map[TARGET_PROJECT_NAME]

    source_id = source_project.get_params()["id"]
    target_id = target_project.get_params()["id"]
    print(f"Source project: '{SOURCE_PROJECT_NAME}' (id={source_id})")
    print(f"Target project: '{TARGET_PROJECT_NAME}' (id={target_id})")

    # Export all tasks from the source project in Label Studio JSON format
    print("Exporting tasks from source project...")
    tasks = source_project.export_tasks(export_type="JSON")
    print(f"  Exported {len(tasks)} tasks.")

    if not tasks:
        print("No tasks to copy. Exiting.")
        return

    # Strip internal IDs so Label Studio treats these as new tasks on import
    for task in tasks:
        task.pop("id", None)
        task.pop("project", None)
        for ann in task.get("annotations", []):
            ann.pop("id", None)
            ann.pop("task", None)
        for pred in task.get("predictions", []):
            if isinstance(pred, dict):
                pred.pop("id", None)
                pred.pop("task", None)

    # Import tasks into the target project in batches to avoid 413
    BATCH_SIZE = 50
    print(f"Importing {len(tasks)} tasks into target project (batch size={BATCH_SIZE})...")
    imported = 0
    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i : i + BATCH_SIZE]
        target_project.import_tasks(batch)
        imported += len(batch)
        print(f"  Imported {imported}/{len(tasks)} tasks...")
    print(f"  Done. Imported {imported} tasks into '{TARGET_PROJECT_NAME}'.")


if __name__ == "__main__":
    main()
