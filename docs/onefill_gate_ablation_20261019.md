# One-filled mixer gates: five independent ablations

Implementation, not submission evidence. **5m_vs_6m only**, seeds 1/2/3,
five models, **15 jobs**, 10M steps (`t_max=10050000`). Exact cohort:
`5m6m_onefill_gate_10m_15jobs_20261019`. No 8m9m jobs are planned. No server jobs, uploads or tmux
processes are created by writing/testing this code.

## Common contract

- Ordinary fixed-head `qmix_minimal` agent; its local observations and action
  selection do not change. Only global **state** sent to the mixer is gated.
- A learned state-independent logit per state field. Training samples a hard
  Binary-Concrete straight-through mask with temperature 0.5: forward is binary,
  `P(mask=1)=sigmoid(logit)`; backward uses the continuous relaxation. This is a
  biased gradient estimator, not exact Bernoulli differentiation.
- `input = mask * state + (1-mask) * 1`. No mask concatenation. W1, W2, b1 and V
  all use the controlled input in single-branch models; there is no raw-state
  bypass. No validity/dead-unit mask is added to the mixer API.
- No evaluation threshold or threshold sweep. Evaluation diagnostics use p;
  acting never calls the mixer. Target TD uses deterministic p, preserving the
  previous soft-target convention. With a nonlinear mixer, evaluating the
  expected input does not equal expected Q over sampled masks.
- This also changes old continuous forward masks to **hard** forward masks.
  These are new named experiments, NOT a pure fill-value-only ablation against
  historical zero-fill static10. All five new models use identical sampling.
- No periodic field perturbation, ranking, teacher targets or group migration.
  Every 100K environment steps save a probability heatmap and JSON with field
  names and the full contract. These probabilities are not importance scores.
- Historical `mixer_input_learner`, threshold diagnostics, checkpoints and all
  historical submission commands remain untouched. New entry:
  `onefill_gate_learner`; old unused YAML threshold is not read by this learner.

| Label | p0 | Loss/routing change vs one-fill p10 |
| --- | --- | --- |
| qmix_onefill_p10 | 0.1 | TD updates actor, mixer and gate |
| qmix_onefill_p20 | 0.2 | Only initial probability changes |
| qmix_onefill_p10_kl10 | 0.1 | TD + direct Bernoulli KL(p || 0.1) |
| qmix_onefill_p10_mc | 0.1 | TD trains actor/mixer with gate detached; direct return MSE trains only gate |
| qmix_onefill_p10_residual | 0.1 | TD; all-ones base plus centered weight residual |

## KL, not KL80aux

Keep prior **0.1**, not 0.8. Mean per-field Bernoulli KL, gate only. Same
loss-value EMA coefficient mechanism as historical direct KL80:
`lambda=min(100, 0.1 * EMA(TD) / max(EMA(KL),1e-8))`, decay 0.99. At KL<=1e-8
the KL coefficient is zero and EMAs are not advanced. No extra auxiliary TD
branch or extra sampled mask. Scaling uses detached scalar loss values, not
gradient norms; the 10% ratio is approximate, not a per-step guarantee.

KL prefers p near 0.1; it neither forces exactly 10% open nor drives p to zero.
TD may still open useful gates. At initialization p=prior, KL has no restoring
gradient until TD moves p. Log raw KL, effective coefficient and weighted ratio.

## Direct return MSE

Every valid batch (not every 100 updates) reuse executed-action Q_i detached
from the actor and form complete observed discounted replay returns G. Hold
mixer weights fixed but retain input-to-gate derivatives:
`L_gate = mean_complete[(Q_tot(stopgrad(Q_i), gated_state; stopgrad(theta))-G)^2]`.
Coefficient 1. No normalization, KL, BCE or sparsity term. Both objectives are
computed against the same pre-update model, using separate stochastic mask
draws. Independent gate optimizer/clipping with the same optimizer type and LR
as main TD; main parameter optimizer and clipping exclude the gate entirely.

Padding/post-terminal samples are excluded. Incomplete or time-limit-truncated
episodes without a recorded terminal have no MC supervision. Skip the gate
optimizer step (including momentum) when no complete returns exist. Empty TD
batches skip all updates. Save availability/count along with MC loss/grad norm.
G is behavior-policy return, not ground-truth optimal Q. MC may be noisy and
may encourage opening many gates; no automatic sparsity or win-rate guarantee.

## Weight residual, not sum of two Q_tot

One Q aggregation, with separate parameter-generation branches. Base B takes
all ones; R takes one-filled gated state x. For both W1 and W2 use
`W = abs(B(ones) + 0.1 * (R(x)-R(ones)))`. Base b1 and V take all ones; they are
not given residuals. The combined weights remain nonnegative, preserving QMIX
monotonicity with respect to Q_i at fixed state/gate.

Residual output heads start at zero; scale is fixed nonzero 0.1. This exactly
matches the all-ones base at initialization. The first TD step may not train
gate logits until the residual heads move; zero scale would prevent learning.
At fully closed gates x=ones the residual is exactly zero, even after training.
Poor residual performance does NOT automatically close gates; a co-trained base
is not an independently trained all-ones fallback. No base-only extra TD,
pretraining/freeze, learned amplitude or extra sparsity is silently introduced.

## Files, checkpoints and commands

Probability diagnostics: `results/mixer_onefill_gate/<unique_token>/`. Offline
W&B includes `mixer_input/onefill_probability_heatmap` and JSON. Checkpoints
include main/gate optimizer states, EMA values, online and lagged target models,
schedule counters and strict architecture/objective contract validation. This
does not save replay-buffer or random-number-generator state; resuming is not
a bitwise-identical continuation. Legacy checkpoints cannot be loaded here.

Plan-only (no Slurm submission):

```bash
cd /home/kyang/code/gomarl-dual-branch &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_onefill_gate_15jobs.py
```

Only after code is available on the server, explicit submission:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_5m6m_onefill_gate_15jobs.py
```

Existing exact active/completed names are retained. Only the 5m6m installed map
is validated; missing 8m9m assets do not block this suite. Map validation, quota
startup guard (2 GiB, not a capacity guarantee), smoke tests and all missing
`sbatch --test-only` checks precede any real submission. Submission manifests
are included in running-only sync and separate ended-run backfill selection;
no new sync loop is launched here. Local three-seed plots and mean3 inventories
include all five labels on 5m6m only; older rows keep their original cohort.

## Upload this finished cohort

After the jobs finish, upload **only this cohort**, preserving original run IDs
and separately creating five readable seed panels and five selectable mean3
analysis runs:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_5m6m_onefill_results.py
```

This is a one-shot, not a tmux loop. Output/report:
`figures/5m6m_onefill_20261019_3seeds/upload_report.json`. It refuses to guess job
IDs when manifests/accounting do not confirm the fifteen ended jobs. Original
cloud checks verify scalar progress only, not every remote media object. Local
curve reads never query cloud histories; original-run checks use exact IDs.
`--no-upload` makes local-only figures/inventory with no scheduler/cloud calls.
`--skip-original-sync` is for retrying figures/scalars after raw backfill is
confirmed; it must not be mistaken for uploading original training records.
Missing seeds, short histories and partial uploads return a nonzero status with
details. A layout failure does not prevent trying mean3; retry layout using
`model_panel_upload.json`, not another full image upload. No training, cleanup,
tmux change or 8m9m upload is performed.

## Local verification

Verified with Python 3.12 / PyTorch 2.8.0 on CPU and simulator-free production
map shapes (98 and 179 state fields). New modules also pass Python 3.9 syntax
compilation. No SC2 episode, GPU training, real Slurm submission or W&B upload
was performed.

- `smoke_test_onefill_gate.py`: five conditions on both maps and Adam/RMSprop;
  binary sampling frequency/fill semantics, closed-field state derivatives,
  exact KL/direction and EMA scaling; direct MC routing, 1e6-loss-scale main
  update invariance, padding/incomplete-return exclusion, optimizer-momentum
  skip; residual initialization/full-close fallback/monotonicity for one/two
  hypernetwork layers; checkpoint targets/optimizer/EMA and seeded next update;
  incompatible resume rejection before mutation; production value diagnostics,
  native heatmap buffer/JSON, exact 15-name 5m6m-only inventory and mock idempotent Slurm.
- `smoke_test_mixer_input_controls.py` and `smoke_test_gate_score_supervision.py`:
  legacy zero-fill/grouped/periodic-supervision behavior remains compatible.
- `smoke_test_plot_two_map_qmix_variants.py`, `smoke_test_plot_local_available.py`,
  `smoke_test_plot_8m9m_local_available.py`: all new models/maps/seeds included,
  local retained CSV handling, independent readable panels and missing-seed
  reporting, with uploads mocked or disabled.
- `smoke_test_three_seed_scalar_runs.py`, `smoke_test_calibration_suite_sync.py`,
  `smoke_test_clean_except_qmix_variants.py`: aggregate/sync/protection regressions
  pass with mock cloud/scheduler and synthetic temporary data only.
