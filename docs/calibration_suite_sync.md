# Exact-cohort status, final backfill and continuous W&B updates

The old login-node upload windows scan RUNNING jobs only, so a job leaving
squeue can lose its final upload. This new tool reads actual submitted/retained
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
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py start
```

`start` creates/revives the dedicated `gomarl-calibration-sync:sync` tmux pane.
The first loop immediately syncs pending backfill and live runs; subsequent
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
the W&B config identity if present. Includes ended/failed runs' available data,
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
