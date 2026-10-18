# Q-change control: MC-error ranking ablation

Reference: `smac_5m6m_qmix_state_group_qchange_10m_s{1,2,3}_mixercontrol`.
New: `smac_5m6m_qmix_state_group_calibration_mean_shared_10m_s{1,2,3}_ablate20261018`.
Seeds 1/2/3, 10M target (`t_max=10050000`), 24G, two-day limit as before.

Keep two independent mixers, ordinary actor, **shared** auxiliary TD gradient,
legacy `[state * group_mask, group_mask]` input (196 dimensions), TD objective,
targets, optimizer/clipping, group schedule, EMA, promotion/demotion rules,
probe budget, evaluation, and diagnostics unchanged. Replace the Q-change
ranking with normalized MC-MSE deletion/replacement benefit and its existing
quality mapping `sigmoid(-benefit)`:

`benefit = (mean((Q-G)^2) - mean((Q_replaced-G)^2)) / max(mean(G^2), 1e-6)`.

Positive means replacement improves calibration and therefore lowers quality.
Negative means replacement harms calibration and increases quality. This is
**not** Var(Q), TD-error scoring, or an additional loss. G is the observed
complete-episode discounted behavior return, not optimal-Q ground truth.

## Preserve the perturbation; disclose the unavoidable probe restriction

The brown `calibration_isolated` and existing `calibration_shared` use legacy
`mixer_calibration_perturbation=mask_zero`: zero the slot and membership bit.
They remain unchanged. The new model explicitly selects `batch_mean`, retaining
Q-change's intervention: replace only the field value by its valid-batch mean,
keep membership fixed, and evaluate only its owning branch with fixed agent Q,
samples, and G. The mean is computed from **all valid TD transitions**, just as
in Q-change, excluding padding and post-terminal transitions.

MC-error scoring needs complete observed returns: scoring probes are restricted
to complete episodes; no bootstrap substitutes for G. If no complete episode
exists, TD training continues but regrouping is skipped. Thus architecture,
gradient routing, and intervention are controlled, while the new score includes
an explicit **complete-return availability restriction** versus Q-change's full
valid pool. Do not describe it as proof about the formula alone independent of
this sampling requirement. Raw probe counts and scopes are logged. With no
complete-return probes the MC plot explicitly says unavailable, rather than
displaying a stale score as a new measurement.

The new selector defaults to legacy mask-zero, is validated, recorded in
snapshots and checkpoints, and a mismatched checkpoint is rejected before load.
Existing experiments/checkpoints retain legacy behavior.

## Submit just this model

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_calibration_mean_shared_3seeds.py
```

No SUBMIT=YES: plan-only. Check 5m6m map and 2 GiB startup headroom, run
simulator-free smoke tests, preflight all missing jobs, then submit at most three.
Exact active/completed names are retained; no old runs are overwritten/cancelled.
No other experiment is launched. Native branch-separated heatmaps are included.
Running-only tmux scope is unchanged; ended backfill, heatmap backfill, storage
preservation, local-only plot and scalar mean3 registries include this cohort.
No data cleaning or cloud plotting is performed by the submitter.
