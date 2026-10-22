# Explicit local experiment-data cleanup

The user explicitly authorized deleting all job data on October 22, superseding
the earlier instruction to retain the last 18 partial runs. This scope is the
known GoMARL project's local experiment directories, not arbitrary account or
group storage, code, environments or W&B cloud runs. The command makes no
backup and does not certify that all originals/media exist in W&B.

Entry point: `scripts/ozstar_clear_all_job_data.py` (preview by default).

It clears contents of the following ordinary directories when present:

- `/home/kyang/gomarl-runtime/gomarl-dual-branch/{results,wandb,figures,grouping_backfill,qcore_upload,td_audit_upload,wandb_cache,wandb_data}`
- `/home/kyang/code/gomarl-dual-branch/{results,wandb,figures}`

It preserves directory roots and shared lock inodes. All `ozstar_logs` including
submission manifests and cleanup audits are preserved: the fresh rerun launcher
still needs the original receipts/accounting. Conda and code are out of scope.

## Execution

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clear_all_job_data.py --apply
```

This is irreversible local deletion, including raw Sacred curves, offline W&B
records, checkpoints, videos, importance/heatmap files, local plots/CSVs and
analysis-run copies. Remove `--apply` for a preview. No extra confirmation is
needed for the already authorized scope; unrelated storage is never included.

Before applying, the tool requires an empty user Slurm queue, successful process
checks on farnarkle1/farnarkle2 and exclusive acquisition of all three GoMARL
sync locks in each existing W&B data root. Active loops, upload/render processes,
failed checks and busy locks abort; it never terminates processes/cancels jobs.
Known writers on either node must be absent. It rechecks both nodes by data
category, and the queue and unchanged target snapshot before each removal.

Symlink roots, different owners/devices, special files and audit-write failures
stop deletion. Descendant symlinks are unlinked rather than followed. A later
failure can leave a partially cleaned set: inspect preceding `DELETED` lines
and the fsynced JSONL audit, rather than assuming everything was cleared.

The output prints before/after user+group quota and the persistent audit path.
Directory-byte estimates do not imply exact quota recovery (hardlinks/open files
can prevent immediate recovery). The tool never restarts uploads or training.
The local agent prepares/tests/deploys the tool but has no authenticated server
execution channel; only actual server output confirms deletion.
