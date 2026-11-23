#!/usr/bin/env python3
"""Pull completed Label Studio annotations, review them, then delete exactly those tasks.

Deleting a task removes the only server-side copy of its annotation, and /blue has no
snapshots, so this runs in separate steps with a manifest between them:

    status            read-only: annotated / total tasks per project
    pull              snapshot annotated tasks, download exactly that snapshot to CSV,
                      mirror into annotations_backup/, write a manifest
    review MANIFEST   summarize what was pulled (labels, empties, oddities)
    delete MANIFEST   delete the manifest's tasks, only if every pulled CSV is
                      byte-identical in a pushed origin/* branch and the task has not
                      been edited since the pull

Between pull and delete: review, commit annotations_backup/, and push.
Unlabeled tasks are never touched — they are the pending annotation queue.

Usage:
    uv run python scripts/pull_annotations.py status
    uv run python scripts/pull_annotations.py pull [--instances review]
    uv run python scripts/pull_annotations.py review <manifest.json>
    uv run python scripts/pull_annotations.py delete <manifest.json> [--dry-run]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from label_studio_sdk import Client

from src.label_studio import (
    download_completed_tasks,
    get_annotated_tasks,
    get_api_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "boem_conf" / "annotation" / "label_studio.yaml"
BACKUP_SCRIPT = REPO_ROOT / "scripts" / "backup_annotations.py"
TAXONOMY_PATH = REPO_ROOT / "transformed_taxonomy.json"
SOURCE = Path("/blue/ewhite/b.weinstein/BOEM/annotations")
MANIFEST_DIR = SOURCE / "pulls"
BACKUP_PREFIX = "annotations_backup"


def load_cfg():
    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    return OmegaConf.load(CONFIG_PATH).label_studio


def find_project(url: str, title: str):
    """Look a project up by title. Never creates one (unlike connect_to_label_studio)."""
    client = Client(url=url, api_key=os.environ["LABEL_STUDIO_API_KEY"])
    matches = [p for p in client.list_projects() if p.get_params()["title"] == title]
    if not matches:
        print(f"  not on the server: {title!r}")
    return matches[0] if matches else None


def fingerprint(task: dict) -> list:
    """Annotation ids and edit times; any change after the pull means unsaved work."""
    return sorted([a["id"], a.get("updated_at")] for a in task["annotations"])


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          check=True, capture_output=True, text=True).stdout


def status(args):
    cfg = load_cfg()
    for name in args.instances:
        inst = cfg.instances[name]
        project = find_project(cfg.url, inst.project_name)
        if project is None:
            continue
        tasks = project.get_tasks()
        done = sum(bool(t.get("annotations")) for t in tasks)
        print(f"{name:15s} {done:>5} annotated / {len(tasks):>5} tasks  ({inst.project_name})")


def pull(args):
    cfg = load_cfg()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"created": stamp, "deleted_at": None, "instances": {}}

    for name in args.instances:
        inst = cfg.instances[name]
        print(f"\n=== {name}: {inst.project_name}")
        project = find_project(cfg.url, inst.project_name)
        if project is None:
            continue
        tasks = get_annotated_tasks(project)
        if not tasks:
            print(f"{name}: no annotated tasks")
            continue

        before = set(Path(inst.csv_dir).rglob("*.csv"))
        annotations = download_completed_tasks(project, inst.csv_dir, tasks=tasks)
        csvs = sorted(set(Path(inst.csv_dir).rglob("*.csv")) - before)
        print(f"{name}: {len(tasks)} tasks -> {annotations.shape[0]} rows in {len(csvs)} CSVs")
        manifest["instances"][name] = {
            "project_name": inst.project_name,
            "csvs": [str(p) for p in csvs],
            "tasks": {str(t["id"]): fingerprint(t) for t in tasks},
        }

    if not manifest["instances"]:
        print("\nNothing to pull.")
        return

    MANIFEST_DIR.mkdir(exist_ok=True)
    path = MANIFEST_DIR / f"{stamp}.json"
    path.write_text(json.dumps(manifest, indent=1))
    subprocess.run([sys.executable, str(BACKUP_SCRIPT)], check=True)
    print(f"\nmanifest: {path}")
    print("Next: review, commit + push annotations_backup/, then delete.")


def taxonomy_names() -> set[str]:
    names = set()

    def visit(node):
        names.update(filter(None, (node.get("alias"), node.get("value"))))
        for child in node.get("children", []):
            visit(child)

    for item in json.loads(TAXONOMY_PATH.read_text())["items"]:
        visit(item)
    return names | {"Object", "FalsePositive", "0"}


def review(args):
    manifest = json.loads(Path(args.manifest).read_text())
    known = taxonomy_names()
    for name, entry in manifest["instances"].items():
        csvs = [Path(p) for p in entry["csvs"]]
        df = pd.concat([pd.read_csv(p) for p in csvs]).assign(label=lambda d: d.label.astype(str))
        empty = df.label == "0"
        boxes = df[~empty]
        degenerate = boxes[(boxes.xmax <= boxes.xmin) | (boxes.ymax <= boxes.ymin)]
        unknown = boxes.label[~boxes.label.isin(known)].value_counts()
        multi = sum(len(fp) > 1 for fp in entry["tasks"].values())

        # Images already present in earlier CSVs of this split = re-downloads / duplicates.
        split_dir = csvs[0].parents[1]
        earlier = set(split_dir.rglob("*.csv")) - set(csvs)
        prior_images = set(pd.concat([pd.read_csv(p, usecols=["image_path"]) for p in earlier]).image_path) if earlier else set()
        repeats = df.image_path[df.image_path.isin(prior_images)].nunique()

        print(f"\n=== {name}: {entry['project_name']}")
        print(f"tasks {len(entry['tasks'])} | images {df.image_path.nunique()} | "
              f"empty images {empty.sum()} | boxes {len(boxes)} | flights {df.flight_name.nunique()}")
        print(f"annotators: {df.annotator.str.strip().value_counts().to_dict()}")
        print(f"FalsePositive boxes: {(boxes.label == 'FalsePositive').sum()}  "
              f"'Object' (no species): {(boxes.label == 'Object').sum()}")
        if "model_label" in boxes and boxes.model_label.notna().any():
            scored = boxes[boxes.model_label.notna() & (boxes.label != "FalsePositive")]
            print(f"model_label agreement: {(scored.label == scored.model_label).mean():.1%} "
                  f"of {len(scored)} pre-filled boxes kept as real")
        print("top labels:\n" + boxes.label.value_counts().head(20).to_string())
        flags = {
            "tasks with >1 annotation (only the first is saved)": multi,
            "degenerate boxes": len(degenerate),
            "images already in earlier CSVs of this split": repeats,
            "labels not in taxonomy": unknown.to_dict() or 0,
        }
        print("FLAGS:")
        for k, v in flags.items():
            print(f"  {'!!' if v else 'ok'} {k}: {v}")


def pushed_blobs() -> dict[str, set[str]]:
    """{backup path: blob hashes} across every origin/* branch, after a fetch."""
    git("fetch", "-q", "origin")
    blobs: dict[str, set[str]] = {}
    for ref in git("for-each-ref", "--format=%(refname)", "refs/remotes/origin").split():
        for line in git("ls-tree", "-r", ref, "--", BACKUP_PREFIX).splitlines():
            meta, path = line.split("\t")
            blobs.setdefault(path, set()).add(meta.split()[2])
    return blobs


def delete(args):
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["deleted_at"]:
        sys.exit(f"Already deleted at {manifest['deleted_at']}.")

    blobs = pushed_blobs()
    unpushed = []
    for entry in manifest["instances"].values():
        for csv in map(Path, entry["csvs"]):
            rel = f"{BACKUP_PREFIX}/{csv.relative_to(SOURCE)}"
            if git("hash-object", str(csv)).strip() not in blobs.get(rel, set()):
                unpushed.append(rel)
    if unpushed:
        print("Refusing to delete: these CSVs are not byte-identical on any pushed origin/* branch:")
        print("\n".join(f"  {p}" for p in unpushed))
        sys.exit("Commit annotations_backup/ and push, then rerun.")
    print("All pulled CSVs are backed up on GitHub.")

    cfg = load_cfg()
    report = {}
    for name, entry in manifest["instances"].items():
        project = find_project(cfg.url, entry["project_name"])
        live = {str(t["id"]): t for t in project.get_tasks()} if project else {}
        gone = [i for i in entry["tasks"] if i not in live]
        edited = [i for i, fp in entry["tasks"].items()
                  if i in live and fingerprint(live[i]) != fp]
        ready = [i for i in entry["tasks"] if i in live and i not in edited]
        print(f"{name}: delete {len(ready)} | edited since pull, kept {len(edited)} | "
              f"already gone {len(gone)}")
        if edited:
            print(f"  edited task ids (will be re-pulled next time): {edited[:20]}")
        if not args.dry_run:
            for task_id in ready:
                project.delete_task(int(task_id))
        report[name] = {"deleted": len(ready), "edited_kept": edited, "already_gone": len(gone)}

    if args.dry_run:
        print("\n--dry-run: deleted nothing.")
        return
    manifest["deleted_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["delete_report"] = report
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nDone. Manifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    all_instances = list(OmegaConf.load(CONFIG_PATH).label_studio.instances)
    for name, fn in [("status", status), ("pull", pull)]:
        p = sub.add_parser(name)
        p.add_argument("--instances", nargs="+", default=all_instances)
        p.set_defaults(fn=fn)
    p = sub.add_parser("review")
    p.add_argument("manifest")
    p.set_defaults(fn=review)
    p = sub.add_parser("delete")
    p.add_argument("manifest")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=delete)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
