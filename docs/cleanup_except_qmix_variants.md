# Local cleanup: keep three mixer variants and all live jobs

`scripts/ozstar_clean_except_qmix_variants.py` previews by default; `--apply`
permanently deletes eligible local data. No cloud verification/upload is done.
No W&B cloud records, source code, or Slurm jobs are changed.

Always retained, including all seeds and retries:

- `smac_5m6m_qmix_state_static10_*`
- `smac_5m6m_qmix_state_group_qchange_*`
- `smac_5m6m_qmix_state_group_tderror_*`
- Their group `5m6m_qmix_mixer_input_controls_10m_9jobs`.
- The newly requested October 14 cohort group
  `qmix_vdn_calibration_gate_10m_3seeds_fresh20261014`, including its fresh baselines.
- The six factorial-ablation jobs in
  `5m6m_score_gradient_factorial_10m_6jobs_20261014`.

All running/pending/completing jobs are retained. Unknown ownership/state is
retained. Terminal state must be confirmed with Slurm accounting and repository
working-directory identity. A same-name queued retry without logs also protects
its history. Shared results folders containing retained/unknown runs stay.

The exact allowed runtime is `/home/kyang/gomarl-runtime/gomarl-dual-branch`.
Only individual W&B run folders, Sacred run leaves, and attributable results
artifact folders (models, videos, importance maps, etc.) are removed. Sacred
raw curves **are deleted**, not just duplicate caches. Logs and shared caches
stay. Symlinks are not traversed. Recent or changed files stay. Scheduler state
and ownership are checked again before each deletion. Cleanup uses the same
exclusive lock as the W&B uploader. An audit is flushed before each deletion
to `ozstar_logs/cleanup_except_qmix_variants.jsonl`.

On the server:

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_except_qmix_variants.py
# After reviewing the DELETE/KEEP list:
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_clean_except_qmix_variants.py --apply
```

This deliberately does not promise to free the full combined folder sizes:
retained live experiments and unidentifiable shared data still occupy space.

Offline safety tests:

```bash
python scripts/smoke_test_clean_except_qmix_variants.py
```
