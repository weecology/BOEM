---
name: slurm-remote-submit
description: >-
  Submits SLURM jobs on a remote cluster from the local workspace via SSH: sync
  the current git branch on the server, then sbatch a chosen script. Use when
  the user wants to run jobs on the cluster, sbatch from local context, deploy
  the branch they are on, or submit pipeline/cluster work remotely.
---

# SLURM remote submit

## Assumptions

- SSH access to the cluster works from this machine (keys, `~/.ssh/config` host alias if used).
- The user handles interactive login, 2FA, or session tickets **before** or outside this workflow so non-interactive `ssh` commands succeed.
- Paths, partition, and account in sbatch scripts match the cluster; this skill does not guess SLURM directives.

## Configuration (once per user or session)

Define and reuse these (environment variables, a small shell snippet, or the user states them in chat):

| Variable | Meaning |
|----------|---------|
| `SLURM_SSH` | SSH target: use **`hpg`** (UF HiPerGator) unless the user specifies another `Host` or `user@host` |
| `SLURM_REMOTE_ROOT` | Parent directory on the cluster for git checkouts (e.g. under `/blue/ewhite/...`; set per user) |
| `SLURM_REPO_DIR` | Directory name for this repo on the server (often `BOEM`) |

**Default for this project:** `export SLURM_SSH=hpg` after `hpg` is defined in `~/.ssh/config` (see [reference.md](reference.md)).

Optional: `SLURM_SBATCH_SCRIPT` — path relative to repo root on the cluster, e.g. `submit_upload_full_flight.sh`.

## Workflow

### 1) Local: record branch and remote

From the workspace root:

```bash
git rev-parse --show-toplevel
git branch --show-current
git remote get-url origin
```

Use the **current branch** and **origin URL** for the remote clone/update.

### 2) SSH: ensure repo at the requested branch

Pick **one** pattern (prefer updating an existing clone if it already exists).

**Fresh clone:**

```bash
ssh "$SLURM_SSH" "mkdir -p '$SLURM_REMOTE_ROOT' && cd '$SLURM_REMOTE_ROOT' && \
  git clone --branch '<BRANCH>' --single-branch '<ORIGIN_URL>' '$SLURM_REPO_DIR'"
```

**Existing clone (typical):**

```bash
ssh "$SLURM_SSH" "cd '$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR' && \
  git fetch origin && git checkout '<BRANCH>' && git pull --ff-only origin '<BRANCH>'"
```

If the branch only exists locally, push it first: `git push -u origin '<BRANCH>'`, then use it in the commands above.

### 3) SSH: submit the job

Run `sbatch` from the repo root on the cluster (adjust script and args to match the project’s scripts, e.g. `submit_flythrough.sh`, `submit_upload_full_flight.sh`):

```bash
ssh "$SLURM_SSH" "cd '$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR' && sbatch <SCRIPT_RELATIVE_PATH> [ARGS...]"
```

Capture the job id from stdout (`Submitted batch job <JOBID>`).

### 4) Report back

Tell the user: SSH target, branch, commit if useful (`git rev-parse HEAD` locally vs on remote), full `sbatch` command, and **JOBID**. Mention standard SLURM log naming (e.g. `slurm-<JOBID>.out` in the submission directory) so they can debug.

## Non-interactive SSH

Open a **master** session once (`ssh hpg` in a terminal, complete 2FA if prompted). With **ControlMaster** + **ControlPersist** (see [reference.md](reference.md)), follow-up `ssh hpg ...` from the agent reuse that connection for ~8h without re-authenticating.

## Optional details

For the `hpg` `ssh_config` block and log path conventions, see [reference.md](reference.md).
