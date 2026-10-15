# Manifest-scoped status, RUNNING-only tmux updates and explicit backfill

## Current default: only jobs running now

`start`, `restart` and `loop` now default to `--running-only`. Every round
intersects exact manifest IDs with this user's live `squeue` RUNNING rows, then
retains the accounting/WorkDir check. Pending, ended, unknown and unrelated jobs
are not uploaded. Only the selected jobs' local logs are read for discovery.
The live queue is checked again immediately before each upload; a CLI upload
already in progress is not cancelled if training finishes during that upload.
No RUNNING jobs means an idle round, without historical run discovery.

On the node hosting the existing `gomarl-calibration-sync` session (currently
farnarkle2), pull the updated code and explicitly switch the existing loop:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py restart --running-only
```

The new log round says `mode=--running-only`. `start` alone leaves an already
live process and its old scope unchanged. The default interval is still 600s.
Do not start a second uploader on another login node or delete lock files to
force this switch. If an old upload child retains the shared lock, inspect it
and wait for it to exit rather than launching duplicate uploads.

Strict RUNNING-only operation deliberately does **not** guarantee a final
upload after a job exits. A new job entering RUNNING in a later round is picked
up automatically, provided it belongs to the configured manifests.

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py status --running-only
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py once --running-only
```

## Optional historical/final backfill (not the tmux default)

RUNNING-only upload windows can lose a final upload when a job leaves squeue.
The optional backfill mode reads actual submitted/retained
IDs from `qmix_vdn_calibration_gate_10m_3seeds_fresh20261014` manifests, then queries
Slurm accounting for State/ExitCode/Elapsed and checks the repository WorkDir.
It also reads `5m6m_score_gradient_factorial_10m_6jobs_20261014` manifests for
the missing score/gradient comparison cells. Use `restart` after pulling this
update so an existing uploader process loads both cohorts.
No inferred contiguous job-ID range is used. Missing accounting remains UNKNOWN.
COMPLETED means Slurm success, not independently verified 10M training coverage.

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py status
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py once --ended-only
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py start --all-states
```

`start` creates/revives the dedicated `gomarl-calibration-sync:sync` tmux pane.
With explicit `--all-states`, the first loop syncs backfill and live runs; subsequent
rounds default to 600 seconds. A round never overlaps itself. Failed uploads
retry without killing the loop. Each upload times out after 180 seconds by
default (`SYNC_TIMEOUT` override). Other uploaders and cleanup share the existing
W&B exclusive lock; a busy lock means skip/retry. Only this session's dead pane
is revived; no unrelated tmux sessions or training jobs are stopped.

View progress:

```bash
tail -f /home/kyang/gomarl-runtime/gomarl-dual-branch/ozstar_logs/gomarl-calibration-sync.log
tmux list-panes -t gomarl-calibration-sync -F '#{pane_current_command} dead=#{pane_dead}'
```

Upload discovery requires exact manifest job IDs in local job logs and checks
the W&B config identity if present. Explicit backfill includes ended/failed runs' available data,
not just RUNNING ones. No data are deleted or marked synced. Upload appends to
original W&B IDs; a local signature ledger avoids resending unchanged successful
sources. Source changes or terminal-state changes force another upload. This
ledger confirms CLI success, not remote-history completeness. Since the ledger
is new, the first round conservatively uploads every matching ended source once,
even if an old uploader previously uploaded it. Later rounds skip unchanged data.

No source files means `NO LOCAL RUN`, not a claim that cloud data were updated.
No manifest IDs means abort rather than guess the missing jobs.

Offline tests (no Slurm/W&B writes):

```bash
python scripts/smoke_test_calibration_suite_sync.py
```
