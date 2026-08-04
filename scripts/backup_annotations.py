#!/usr/bin/env python3
"""Mirror Label Studio annotation CSVs into the repo so git history is their backup.

The CSVs under /blue/ewhite/b.weinstein/BOEM/annotations are the only surviving copy
of every human annotation: `src/label_studio.check_for_new_annotations` deletes
completed tasks from the Label Studio server after downloading them, and /blue is
Lustre with no snapshots. Losing that tree loses the labeling effort outright.

This copies the tree into `annotations_backup/` at the repo root, preserving the
<split>/<flight>/<timestamp>.csv layout. Downloads are append-only (each run writes a
new timestamped file), so this is normally pure addition; a changed file is reported
loudly because it means an existing annotation was rewritten in place.

Usage:
    uv run python scripts/backup_annotations.py           # sync, then commit the result
    uv run python scripts/backup_annotations.py --check   # report drift, write nothing
"""

import filecmp
import shutil
import sys
from pathlib import Path

SOURCE = Path("/blue/ewhite/b.weinstein/BOEM/annotations")
DEST = Path(__file__).resolve().parents[1] / "annotations_backup"
SPLITS = ("train", "validation", "review")


def relative_csvs(split: str) -> set[Path]:
    """Paths of every annotation CSV in one split, relative to SOURCE."""
    return {p.relative_to(SOURCE) for p in (SOURCE / split).rglob("*.csv")}


def main():
    check_only = "--check" in sys.argv

    new, changed, unchanged = [], [], []
    for split in SPLITS:
        for rel in sorted(relative_csvs(split)):
            src, dst = SOURCE / rel, DEST / rel
            if not dst.exists():
                new.append(rel)
            elif filecmp.cmp(src, dst, shallow=False):
                unchanged.append(rel)
                continue
            else:
                changed.append(rel)

            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # Files in the backup with no counterpart upstream: either an upstream deletion
    # (the case this whole script exists to survive) or a renamed flight dir.
    live = {rel for split in SPLITS for rel in relative_csvs(split)}
    orphans = sorted(
        p.relative_to(DEST) for p in DEST.rglob("*.csv")
        if p.relative_to(DEST) not in live
    )

    verb = "would copy" if check_only else "copied"
    print(f"source: {SOURCE}")
    print(f"backup: {DEST}\n")
    print(f"  new        {len(new):>5}  ({verb})")
    print(f"  changed    {len(changed):>5}  ({verb})")
    print(f"  unchanged  {len(unchanged):>5}")
    print(f"  total      {len(new) + len(changed) + len(unchanged):>5}")

    if changed:
        print("\nWARNING: these already-backed-up CSVs changed upstream — an existing")
        print("annotation was rewritten, not appended. Check the diff before committing:")
        for rel in changed[:20]:
            print(f"  {rel}")
        if len(changed) > 20:
            print(f"  ... and {len(changed) - 20} more")

    if orphans:
        print(f"\nNOTE: {len(orphans)} CSV(s) exist in the backup but not upstream —")
        print("deleted or renamed at the source. The backup is retaining them:")
        for rel in orphans[:20]:
            print(f"  {rel}")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")

    if check_only and (new or changed):
        sys.exit(1)


if __name__ == "__main__":
    main()
