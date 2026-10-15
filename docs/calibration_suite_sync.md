# Separate live sync from ended-job backfill

## Periodic tmux upload: RUNNING jobs only

Use the existing `ozstar_wandb_sync_tmux.sh` launcher and
`ozstar_sync_running_counter_once.sh` uploader. Despite the historical Counter
filename, it queries this user's RUNNING jobs across this repository, checks
each Slurm WorkDir, and reads actual StdOut/StdErr paths from scontrol.
The original exact-path discovery and run-name fallback are retained. No
experiment manifest is required for this live path. It never calls ended-job
backfill, plot generation or cleanup.

On farnarkle2, where the user last confirmed the existing session, switch that
session back to the original shell uploader (same session/log name):

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SESSION_NAME=gomarl-calibration-sync \
SYNC_LOG=/home/kyang/gomarl-runtime/gomarl-dual-branch/ozstar_logs/gomarl-calibration-sync.log \
bash scripts/ozstar_wandb_sync_tmux.sh restart
```

For normal starts use `start` instead of `restart`. The interval remains 600s;
slow rounds do not overlap. This replaces only the named sync session, not
training jobs. Do not start a second uploader on another login node or delete
locks. If an old upload child retains the shared once lock, a new round skips
and retries later; inspect that exact child if it persists.

```bash
tail -n 30 -f /home/kyang/gomarl-runtime/gomarl-dual-branch/ozstar_logs/gomarl-calibration-sync.log
tmux list-panes -t '=gomarl-calibration-sync' -F '#{pane_current_command} dead=#{pane_dead}'
```

Expected output contains `syncing job=...` and
`running repository sync result: matched=... uploaded=... failed=...`.
An empty RUNNING queue is a valid idle round. New RUNNING jobs are picked up
later; ended jobs do not get automatic final uploads in this loop. Starting
the window does not prove upload success or cloud completeness.

## Historical/final upload: explicit single pass

When requested, separately run:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_sync_ended_once.py
```

This entry point makes one pass, then exits. It selects only known terminal
jobs from the configured calibration/ablation manifests, validates accounting
and repository WorkDir, and indexes only those jobs' logs. RUNNING, PENDING and
unknown jobs are excluded. Failed/timeout jobs may have partial data; inclusion
is not a claim that training completed its intended budget.

It appends to original W&B IDs, retains local data and uses the shared upload
lock. Unchanged successful attempts are skipped by the signature ledger, not
independent cloud-history verification. A busy lock means no pass was performed;
rerun separately when free, do not start another periodic loop. `NO LOCAL RUN`
means no matching local source was found; deleted records cannot be recovered
by uploading. The live script never calls this entry point.

`ozstar_calibration_suite_sync.py` retains its older compatibility CLI, but is
no longer the recommended tmux launcher. Do not use its `--all-states` loop
for the user's normal live updates.

## Offline checks

```bash
python scripts/smoke_test_counter_sync_tmux.py
python scripts/smoke_test_calibration_suite_sync.py
python scripts/smoke_test_sync_ended_once.py
```

These use synthetic local data and mocked Slurm/W&B/tmux tools; they do not
verify actual server paths or cloud history.
