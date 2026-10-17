# 5m6m: three clean Q-change grouping ablations

Reference: `smac_5m6m_qmix_state_group_qchange_10m_s{1,2,3}_mixercontrol`.
Each ablation changes exactly one factor relative to this reference, not relative
to the preceding ablation. All use seeds 1/2/3 and the existing 10M setup.

| Model | Mixer condition encoding (5m6m) | Auxiliary TD to agent | Ranking statistic |
| --- | --- | --- | --- |
| Old Q-change control | `[state × group_mask, group_mask]`, 196 dimensions | Shared | Mean absolute Q_tot change |
| A1: Q-change isolated | Same 196 dimensions | Detached | Same Q-change |
| A2: Q-change plain mask | `state × group_mask`, original 98 dimensions | Shared | Same Q-change |
| A3: Q variance | Same 196 dimensions | Shared | Var(original Q_tot) − Var(perturbed Q_tot) |

Only A2 removes the extra membership features. A1 and A3 deliberately keep the
old encoding to isolate their respective changes. The ordinary agent network,
two QMIX branches, losses, target construction, group schedule, EMA and migration
rules otherwise remain unchanged. A1 trains the auxiliary mixer on detached
agent Q with its own optimizer and gradient clipping; the main branch still
updates the agent.

## Identical evaluation protocol

All three variants retain the control's single-field **batch-mean replacement**;
they do not additionally switch to zero replacement. Group masking itself is
still multiplication. This distinction avoids mixing a perturbation change into
the encoding or statistic ablations.

Evaluate only a field's currently owning group, holding agent Q and membership
fixed. Select the same valid replay probes as the control; exclude padding and
invalid post-terminal transitions. A3 computes population variance across those
probes (`unbiased=False`), keeps the signed difference, and uses neither absolute
value, sigmoid nor positive clipping. At least two valid probes are required.
Its EMA ranking retains the control's update rules. This measures Q dispersion,
not error against a theoretical return, and does not by itself establish that a
field improves policy performance.

A2's inactive raw state fields and corresponding input-column gradients are
zero in the current backward pass. This is not an optimizer-state freeze:
historical optimizer momentum can still move a parameter.

## Names and retained jobs

A1 reuses the previously declared names exactly:
`smac_5m6m_qmix_state_group_qchange_isolated_10m_s{1,2,3}_ablate20261014`.
The submitter retains matching active or completed jobs rather than launching
duplicates. A2/A3 have independent names:

- `smac_5m6m_qmix_state_group_qchange_plainmask_10m_s{1,2,3}_ablate20261015`
- `smac_5m6m_qmix_state_group_variance_10m_s{1,2,3}_ablate20261015`

At most nine jobs are submitted; if A1 already exists, only six are new. No old
experiment is overwritten, deleted or cancelled. The startup quota check remains
2 GiB; this is not a guarantee of sufficient storage for the entire batch.

## Diagnostics and integration

All variants log grouping heatmaps and per-field details. Snapshots distinguish
the pre-migration partition used to evaluate scores from the current partition
after migration. A3 records both variances, their raw signed difference, probe
count and EMA ranking. Existing two-map charts, upload/backfill inventory,
suite sync and cleanup preservation include these new models. Dedicated plots
show the old control plus these three variants, with each model's three seeds
in its own 5m6m panel. Historical cohorts are not merged.

Run on OzSTAR from the repository:

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_qchange_three_ablation_9jobs.py
```

Plot available data (before all seeds finish, coverage is reported honestly):

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_5m6m_qchange_three_ablation_3seeds.py --local-only --mean-window 10
```

An already-running sync process needs to reload the updated code before it can
discover the new suite manifest. Backfill uses the existing original run identity
and native media, not new training runs.

## Plot the retained control after raw-log cleanup

The control's raw test-win points may still exist in an earlier plot directory,
even when the original offline/Sacred files are gone. Read its `seed_curves.csv`
with the sibling `seed_inventory.csv`, not a smoothed aggregate. Exact planned
run names, map/model/seed, original run IDs and attempt start times must agree.
Retained snapshots participate in latest-attempt selection; a newer empty run
does not silently borrow an old successful curve. Snapshots are not concatenated.
The source CSV path is recorded in the new inventory.

This command reads the confirmed retained control and current local ablations,
omits models with zero readable seeds, and writes to a separate directory. No
cloud reads, uploads, training or sync changes are performed:

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_5m6m_qchange_three_ablation_3seeds.py \
  --local-only --no-upload --available-only --mean-window 10 \
  --retained-curves /home/kyang/gomarl-runtime/gomarl-dual-branch/figures/5m6m_latest_gate_group_baselines_3seeds/seed_curves.csv \
  --output-dir /home/kyang/gomarl-runtime/gomarl-dual-branch/figures/5m6m_qchange_local_with_control
```

Each seed retains its actual endpoint; 1/3 coverage is not labelled as three
seeds. `--available-only` affects figures, not the expected-run inventory, which
still records missing seeds. Omit only `--no-upload` when uploading is desired;
keep `--local-only` to avoid cloud downloads. Synthetic regression QA:
`python scripts/smoke_test_retained_seed_curves.py`.

## Upload all locally available 5m6m QMIX variants and VDN

For the broader ten-model 5m6m comparison (fresh QMIX/VDN, static gate,
Q-change/TD-error/return-error grouping controls and their ablations), use:

```bash
cd /home/kyang/code/gomarl-dual-branch
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_5m6m_local_available_3seeds.py
```

This entry point always enables `--local-only --available-only`. It reads
immediate `figures/*/seed_curves.csv` snapshots with matching inventories, in
addition to current raw logs. It does not query cloud run metadata or histories.
It uploads figures and saves the Media/5m6m layout by default; `--no-upload` is
available for local QA. The new output directory is
`figures/5m6m_local_available_qmix_variants_3seeds`, excluded from snapshot reads.
Other saved CSVs are not overwritten. Duplicate copies of an original run ID
are checked for agreement and the longest intact copy is selected, never joined
or counted as extra seeds. Unknown schemas/provenance are reported and skipped;
conflicting copies stop the operation. Missing models are omitted from figures
but remain in the expected-run inventory. Synthetic test:
`python scripts/smoke_test_plot_local_available.py` (upload mocked).
