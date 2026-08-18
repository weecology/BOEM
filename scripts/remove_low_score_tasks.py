#!/usr/bin/env python3
"""Delete unlabeled Label Studio tasks whose only boxes are low-confidence.

A task is doomed when it has at least one predicted box (rectanglelabels
result) and every box's score is below --threshold. Tasks with zero boxes are
left alone — that is a separate empty-screening concern, not this one. Tasks
with any box at or above the threshold are left alone too, since an annotator
still has real work to do there.

Only unlabeled tasks are touched (matching scripts/delete_unlabeled_tasks.py's
pattern): get_unlabeled_tasks() is re-filtered on `annotations` so a task
labeled since the listing was fetched survives.

Usage:
    uv run python scripts/remove_low_score_tasks.py --dry-run
    uv run python scripts/remove_low_score_tasks.py --threshold 0.6
    uv run python scripts/remove_low_score_tasks.py --instances train validation
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


def box_scores(task):
    """All rectanglelabels prediction scores for a task, across all prediction objects."""
    return [
        result["score"]
        for prediction in (task.get("predictions") or [])
        for result in (prediction.get("result") or [])
        if result.get("type") == "rectanglelabels" and result.get("score") is not None
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Delete tasks where every box scores below this (default 0.6).")
    parser.add_argument("--instances", nargs="+", default=["train", "validation", "review"])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted.")
    args = parser.parse_args()

    os.environ["LABEL_STUDIO_API_KEY"] = get_api_key()
    cfg = OmegaConf.load(CONFIG_PATH).label_studio

    for instance_name in args.instances:
        instance = cfg.instances[instance_name]
        print(f"\n=== {instance_name}: {instance.project_name}")
        project = connect_to_label_studio(url=cfg.url, project_name=instance.project_name)

        unlabeled = [t for t in project.get_unlabeled_tasks() if not t.get("annotations")]

        doomed = []
        n_no_boxes = 0
        for task in unlabeled:
            scores = box_scores(task)
            if not scores:
                n_no_boxes += 1
                continue
            if max(scores) < args.threshold:
                doomed.append((task, scores))

        print(f"  {len(unlabeled)} unlabeled tasks ({n_no_boxes} with no boxes, skipped)")
        print(f"  {len(doomed)} tasks with all boxes < {args.threshold}")

        if doomed:
            n_boxes = Counter(len(s) for _, s in doomed)
            for n, count in sorted(n_boxes.items()):
                print(f"    {count:>6} tasks with {n} low-score box(es)")

        if not doomed:
            continue

        if args.dry_run:
            print(f"  --dry-run: would delete {len(doomed)} tasks")
            continue

        for task, _ in doomed:
            project.delete_task(task["id"])
        print(f"  deleted {len(doomed)} tasks")


if __name__ == "__main__":
    main()
