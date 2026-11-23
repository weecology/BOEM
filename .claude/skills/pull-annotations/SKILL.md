---
name: pull-annotations
description: Pull the latest completed BOEM annotations from Label Studio, review them, back them up to git, and only then delete those tasks from the server. Use when asked to download/pull/sync/harvest Label Studio annotations, clear completed tasks, or check what annotators have finished.
---

# Pull BOEM annotations from Label Studio

The downloaded CSVs under `/blue/ewhite/b.weinstein/BOEM/annotations/<split>/<flight>/<timestamp>.csv`
become the **only** copy of each annotation once its task is deleted, and /blue has no
snapshots. So deleting happens last, and only after the backup is on GitHub. Everything goes through
`/blue/ewhite/b.weinstein/src/BOEM/scripts/pull_annotations.py`. Run all commands from the repo root.

Projects (from `boem_conf/annotation/label_studio.yaml`): `train`, `validation`, `review`,
`dolphin_review`. Default is all of them; pass `--instances review` etc. to narrow, or to
follow what the user asked for.

## 1. Status (read-only)

```
uv run python scripts/pull_annotations.py status
```

Report annotated/total per project. If nothing is annotated, stop here. Annotators work
live, so always use the live numbers, never numbers from an earlier pull or from memory.

## 2. Pull

```
uv run python scripts/pull_annotations.py pull
```

This snapshots the annotated tasks, downloads exactly that snapshot, syncs
`annotations_backup/`, and prints a manifest path
(`/blue/ewhite/b.weinstein/BOEM/annotations/pulls/<timestamp>.json`). It deletes nothing.
Do not run `pull` twice before deleting, because each pull writes new timestamped CSVs for
the same tasks (duplicates).

Read the backup script's output: any **changed** (not new) file is a red flag, because
downloads are append-only. Stop and show the user before going further.

## 3. Review

```
uv run python scripts/pull_annotations.py review <manifest>
```

Summarize for the user, per project: tasks, images, empty images, boxes, annotators, top
labels, FalsePositive / bare `Object` counts, and model_label agreement. Then go through
every `!!` FLAG:

- **tasks with >1 annotation**: `download_completed_tasks` keeps only the first. The
  others would be lost on delete. Ask the user before deleting.
- **labels not in taxonomy**: typos or retired names. Report them. They do not block the
  delete, since the CSV keeps them as typed.
- **images already in earlier CSVs**: re-downloads or re-annotation. Say how many. A large
  number usually means an earlier pull was never followed by a delete.
- **degenerate boxes**: zero or negative area. Report them.

Also note anything the numbers make suspicious, such as one annotator labeling everything
`Object` or a sudden all-empty flight. Use the per-flight CSVs listed in the manifest to
dig in. Only you and the user decide whether to proceed. The script does not.

## 4. Commit and push the backup

On the current branch (minor data commit, no PR needed):

```
git add annotations_backup
git commit -m "Back up <split> annotations pulled <YYYY-MM-DD> (<rows> rows, <images> images)"
git push -u origin <current-branch>
```

Retry the push with backoff on network errors. The delete step will refuse until these
CSVs are byte-identical on some `origin/*` branch.

## 5. Delete (confirm with the user first)

Deleting is irreversible, so show the dry run and get an explicit go-ahead:

```
uv run python scripts/pull_annotations.py delete <manifest> --dry-run
uv run python scripts/pull_annotations.py delete <manifest>
```

Delete removes only the manifest's task ids. Tasks edited after the pull are kept and
listed, and the next pull picks them up. Unlabeled tasks are never touched. The manifest
records `deleted_at` and a per-project report, and refuses to run twice.

## 6. Wrap up

Report the absolute manifest path, the commit hash and branch, and the deleted/kept counts
per project. Finish with `status` again to show what remains in the queue.
