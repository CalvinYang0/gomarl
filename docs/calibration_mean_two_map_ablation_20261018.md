# MC-error ranking: independent gradient/input ablations on two maps

Reference method: the October 18 `calibration_mean_shared` experiment. Keep
its normalized MC-MSE quality ranking, complete-return probe restriction,
batch-mean field replacement (membership held fixed), ordinary actor,
two independent mixers, TD targets/loss scale, group schedule, EMA/migration,
and test/diagnostic settings. No model implementation changes are required.

| New label | Mixer condition | Auxiliary TD -> agent |
| --- | --- | --- |
| `qmix_state_group_calibration_mean_isolated` | `[state * mask, mask]` | Detached; auxiliary mixer still learns with separate optimizer/clipping |
| `qmix_state_group_calibration_mean_plainmask` | `state * mask` | Shared, as in the reference |

These are **independent**, not cumulative: plain-mask does not also detach.
5m6m dimensions: isolated 196, plain-mask 98. 8m9m: isolated 358, plain-mask 179.
Raw out-of-group state gradients/current input-column data gradients are zero;
this does not freeze shared parameters or optimizer momentum.

Seeds 1/2/3, 10M target (`t_max=10050000`), both 5m_vs_6m and 8m_vs_9m:
exactly **12 jobs**, no extra baselines and no repeats of the prior three-job
5m6m shared-reference suite. Names:

`smac_{5m6m,8m9m}_qmix_state_group_calibration_mean_{isolated,plainmask}_10m_s{1,2,3}_ablate20261018`.

The matched shared-gradient/mask-concat reference is already declared on 5m6m.
**No equivalent 8m9m reference is submitted here.** That map can compare the
two new variants and existing QMIX/VDN/gate controls, but the two variants differ
in both encoding and gradient route relative to each other. A later matched
8m9m shared/mask-concat reference would be needed to attribute each factor's
effect independently on that map; do not substitute an older scoring method.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_calibration_mean_ablation_12jobs.py
```

Default is plan-only. Actual submission validates both installed maps, checks
the unchanged 2 GiB startup guard (not a batch capacity guarantee), runs
simulator-free preflight tests, and validates every missing Slurm job before
submitting any. Repeated invocation retains exact active/completed names.
No job cancellations, force reruns, runtime cleanup, or periodic-loop changes.

Native branch-separated heatmaps and readable grouping tables are retained.
This cohort is registered for one-shot ended backfill, historical heatmap
repair, storage preservation, both-map/local-only plots and scalar mean3.
The existing running-only repository sync automatically discovers the jobs
without broadening periodic scope. Plotting/uploads are not started by this
submission command; missing/partial seeds remain explicit later.
