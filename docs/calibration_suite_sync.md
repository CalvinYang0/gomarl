# Separate live sync from ended-job backfill

## Periodic tmux upload: RUNNING jobs only

Use the existing `ozstar_wandb_sync_tmux.sh` launcher and
`ozstar_sync_running_counter_once.sh` uploader. Despite the historical Counter
filename, it queries this user's RUNNING jobs across this repository, checks
each Slurm WorkDir, and reads actual StdOut/StdErr paths from scontrol.
The original exact-path discovery and run-name fallback are retained. No
experiment manifest is required for this live path. It never calls ended-job
backfill, plot generation or cleanup.

On the node that owns the existing uploader (latest user snapshot: farnarkle1),
update and start the original shell uploader (same session/log name):

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SESSION_NAME=gomarl-calibration-sync \
SYNC_LOG=/home/kyang/gomarl-runtime/gomarl-dual-branch/ozstar_logs/gomarl-calibration-sync.log \
bash scripts/ozstar_wandb_sync_tmux.sh start
```

`start` preserves live panes, and replaces an entirely dead legacy session even
if its window was called `sync` instead of `wandb-sync`. Use `restart` only when
explicitly changing an already-live loop. The interval remains 600s;
slow rounds do not overlap. This replaces only the named sync session, not
training jobs. Do not start a second uploader on another login node or delete
locks. If an old upload child retains the shared once lock, a new round skips
and retries later; inspect that exact child if it persists.

```bash
tail -n 30 -f /home/kyang/gomarl-runtime/gomarl-dual-branch/ozstar_logs/gomarl-calibration-sync.log
tmux list-panes -t '=gomarl-calibration-sync' -F '#{pane_current_command} dead=#{pane_dead}'
```

Expected output contains `syncing job=...`, `VERIFIED cloud=...` and
`running repository sync result: matched=... uploaded=... failed=...`.
An empty RUNNING queue is a valid idle round. New RUNNING jobs are picked up
later; ended jobs do not get automatic final uploads in this loop. Starting
the window does not prove upload success or cloud completeness.

### W&B 0.18.7 null-schema repair (2026-10-16)

The server log failed in `upsert_run -> server_settings_introspection`, where
`ServerSettingsType: null` triggered `NoneType.get`. Legacy `wandb sync` ignored
the run-result error and could still print a URL and `done`. Both the original
RUNNING uploader and the manifest-based one-shot path now invoke
`ozstar_wandb_sync_verified.py` instead of bare `python -m wandb sync`.

This helper repairs only 0.18.7's nullable settings probe inside the short-lived
uploader process. It does not edit site-packages, upgrade Conda, alter training,
disable authentication, invent a new run ID or delete local data. Valid schema
responses retain their normal behavior; permission/network failures still fail.
Run-result errors and worker exceptions become nonzero uploader failures.

Before upload, the helper reads the original binary ID/destination and the
latest complete scalar step. After upload, fresh API reads must find that exact
run and a history step at least as large as the pre-upload snapshot. Only then
does the shell count `uploaded`, or the backfill ledger advance. Three bounded
verification attempts tolerate short indexing delays; otherwise the next round
retries. This is a scalar-progress check, NOT a full per-metric/media audit and
NOT proof that an active training run has completed its budget.

The shell's RUNNING selection, Slurm-log discovery, locks, run IDs and 600-second
cadence are unchanged. To run one immediate RUNNING-only round after pulling:

```bash
bash scripts/ozstar_sync_running_counter_once.sh
```

A busy shared lock skips the pass safely; do not bypass/delete the lock.

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
lock. Unchanged verified snapshots are skipped by the signature ledger, not
re-audited for all media on every pass. Old exit-code-only signatures have no
verification version and are retried automatically. A busy lock means no pass was performed;
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
python scripts/smoke_test_wandb_sync_verified.py
```

These use synthetic local data and mocked Slurm/cloud/tmux tools; they do not
verify actual server paths or cloud history. The last regression requires the
affected W&B 0.18.7 SDK and reproduces the original null-response failure before
checking the repair, error propagation and cloud-progress acceptance rules.
