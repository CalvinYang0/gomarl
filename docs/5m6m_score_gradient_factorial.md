# Complete the 5m6m score × gradient-routing comparison

Two additional models, seeds 1/2/3, T_MAX=10,050,000, same 10M evaluation budget.
Only the corresponding gradient-routing flag differs from each matched source.
No existing jobs are cancelled, restarted, or renamed.

| Score | Shared agent gradients | Auxiliary agent gradients detached |
| --- | --- | --- |
| Q-change | Existing `qmix_state_group_qchange` | NEW `qmix_state_group_qchange_isolated` |
| Return-error masking gain | NEW `qmix_state_group_calibration_shared` | Existing `qmix_state_group_calibration_isolated` |

New names end `ablate20261014`, distinct from completed sources. New group:
`5m6m_score_gradient_factorial_10m_6jobs_20261014`.

Shared mode: two 0.5-weight TD losses update the same ordinary agent, a shared
optimizer and combined gradient clipping (historical behavior). Isolated mode:
same loss weights, but auxiliary Q is detached; separate optimizers/clipping
keep auxiliary gradients out of the actor/important mixer. Optimizer separation
is part of the isolation treatment, not a separate independently controlled factor.
All other score/input/partition/target/training/media settings stay matched.

Submit only the missing six jobs:

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_score_gradient_ablation_6jobs.py
```

Omit SUBMIT=YES for preview. At least 2 GiB free quota and the installed 5m6m
map are checked. All smoke tests and Slurm preflights run before submission.
Re-running retains exact-name active/completed runs. Failed runs are not
automatically treated as completed. Submission manifest is flushed after each ID.

The cleanup whitelist preserves this group. Status/sync now reads both the
original 18-job and additional 6-job manifests. Restart only its own uploader
to load the new scope (no training jobs affected):

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py restart
```

The same two-map plot command automatically includes the new six names,
bringing inventory to 33 runs and readable sections to 8 models on 5m6m,
3 on 8m9m. Before data appear they are labelled missing, never drawn as zero
performance. Historical Q-change/shared vs fresh Q-change/isolated remain
separate attempts/cohorts; no seed histories are joined across retries.
