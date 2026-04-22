# SLURM remote submit — reference

## UF HiPerGator: `~/.ssh/config`

Use host alias **`hpg`** so `SLURM_SSH=hpg` matches this project’s submit/debug skills:

```
Host hpg
    HostName hpg.rc.ufl.edu
    User b.weinstein
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

After one interactive `ssh hpg` (2FA if required), multiplexed sessions reuse the master for **ControlPersist** duration.

## SSH multiplexing elsewhere

For other clusters, copy the same `ControlMaster` / `ControlPath` / `ControlPersist` pattern and set `SLURM_SSH` to that `Host` name.

## Log files

Default Slurm stdout/stderr often looks like `slurm-<JOBID>.out` in the working directory where `sbatch` ran (`chdir` in script or `#SBATCH -D`). Confirm with:

```bash
ssh "$SLURM_SSH" "ls -lt '$SLURM_REMOTE_ROOT/$SLURM_REPO_DIR'/slurm-*.out | head"
```

## Repo sync without push (advanced)

If the branch must not be on `origin`, use `git bundle` or `rsync` of the repo; that is outside the default clone/fetch workflow.
