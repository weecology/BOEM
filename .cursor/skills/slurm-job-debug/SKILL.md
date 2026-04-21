---
name: slurm-job-debug
description: >-
  Debugs failed SLURM jobs: fetches the job log from the cluster over SSH,
  identifies the error, fixes code in the local workspace, then re-submits using
  the slurm-remote-submit workflow. Use when a batch job failed, the user
  mentions slurm-*.out errors, sacct/squeue failures, or wants an
  edit-resubmit loop for cluster runs.
---

# SLURM job debug loop

## When to use

- A submitted job exited non-zero, timed out, or the user sees errors in the Slurm output file.
- The user provides a **JOBID**, a **log path on the cluster**, or agrees to discover logs under the repo directory.

## Inputs

Ask or infer:

- `SLURM_SSH` (default **`hpg`** for UF HiPerGator), `SLURM_REMOTE_ROOT`, `SLURM_REPO_DIR` — same as [slurm-remote-submit](../slurm-remote-submit/SKILL.md).
- **JOBID** or explicit log path: `REMOTE_LOG` (e.g. `$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR/slurm-12345.out`).

## Loop (repeat until fixed or blocked)

### 1) Fetch the log

Prefer **tail** for large files; use **full file** if small or the error is near the top.

```bash
ssh "$SLURM_SSH" "tail -n 250 '<REMOTE_LOG>'"
```

If the log path is unknown:

```bash
ssh "$SLURM_SSH" "ls -lt '$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR'/slurm-${JOBID}.out 2>/dev/null || \
  find '$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR' -name 'slurm-${JOBID}.out' 2>/dev/null"
```

Optional status:

```bash
ssh "$SLURM_SSH" "sacct -j '$JOBID' --format=JobID,State,ExitCode,Elapsed,MaxRSS"
```

### 2) Diagnose

- Read from the **bottom** of the log first (Python tracebacks, CUDA OOM, file not found, module load errors).
- Map paths: cluster paths (e.g. `/blue/...`) ↔ local workspace paths.
- Align with project conventions (see repo `AGENTS.md` / launch/debug args if relevant).

### 3) Fix locally

- Edit the **local** repo; keep changes minimal and consistent with existing style.
- Run quick local checks if appropriate (e.g. `uv run pytest`, lint, or a dry-run the project supports).

### 4) Re-submit

Follow the **slurm-remote-submit** skill end-to-end:

1. Confirm branch (`git branch --show-current`) and push if needed (`git push -u origin '<BRANCH>'`).
2. On the cluster: `git fetch`, checkout branch, `git pull --ff-only`.
3. Run the same `sbatch` line as before (or the corrected script/args).

Report the new **JOBID** and what changed.

### 5) Stop conditions

- **Stop** when the job should succeed and logs show completion, or the user stops the loop.
- **Escalate** (do not guess) when the failure is site-specific (quota, module versions, node hardware): tell the user exactly what the cluster reported.

## Relationship to slurm-remote-submit

This skill **does not** duplicate the full submit procedure. After each fix, apply [slurm-remote-submit](../slurm-remote-submit/SKILL.md) so SSH, branch sync, and `sbatch` stay consistent.
