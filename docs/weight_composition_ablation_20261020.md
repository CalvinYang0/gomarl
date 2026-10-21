# Mixer weight composition: 8m9m + 5m6m, 15 jobs

New implementation and commands, not evidence that jobs have been submitted.
Group: `weight_composition_10m_15jobs_20261020`; each model uses seeds 1/2/3,
10M plotted steps, `t_max=10050000`. Do not resubmit the October 19 cohort.

## Exact models and formulas

For each of the TWO weight layers independently, let `A=B(ones)` be the raw
main branch, `R` the secondary branch, and `x=m*s+(1-m)` the one-filled global
state. `m` is a hard Bernoulli forward sample with biased Binary-Concrete
straight-through derivative, temperature 0.5, initial keep probability 0.1.
These are parameter-generation branches, NOT a sum of independent Q_tot.

| Map | Label | Effective weights |
| --- | --- | --- |
| 8m9m | `qmix_weight_bounded_p10` | `abs(A)*(1+0.1*tanh(R(x)-R(ones)))` |
| 5m6m | `qmix_residual_zerofill_p10` | `abs(A+0.1*(R(m*s)-R(zeros)))` |
| 5m6m | `qmix_weight_sum_p10` | `abs(A+R(x))` |
| 5m6m | `qmix_weight_alpha10_p10` | `abs(alpha*A+(1-alpha)*R(x))` |
| 5m6m | `qmix_weight_sum_bothones` | `abs(A+R(ones))` |

- Sum/alpha models remove BOTH the old centering and fixed 0.1 residual factor.
  Abs is applied AFTER raw addition: `abs(A+B)`, NOT `abs(A)+abs(B)`.
- Bounded is an identity-centered multiplicative adaptation, NOT an exact
  reproduction of INR-GAN FMM. Each effective weight lies between 0.9 and 1.1
  times its current nonnegative main weight. This does not bound cumulative
  training drift or guarantee policy performance.
- One learned GLOBAL alpha is shared by W1/W2. `alpha=sigmoid(logit)` with
  initialization 0.1, so initially main coefficient=0.1, secondary=0.9.
  Clamp only at machine epsilon/1-epsilon to preserve strict float bounds.
  It is an amplitude, not a field gate or a test-time threshold.
- All secondary output heads start at zero, preserving the existing initializer.
  Direct sum/centered/bounded start at the current all-ones main mixer.
  Alpha starts with 0.1 times its main WEIGHTS (for BOTH layers), not the same
  initial Q function as direct sum; b1/V are NOT scaled. This is an explicit
  consequence of the requested coefficient, not a matched-function initializer.
- Do not separately abs a zero secondary head: PyTorch's derivative of abs at
  zero is zero and would prevent that head from learning. The tested raw-sum
  construction permits nonzero secondary gradients from the first update.
- Both-ones ignores state, has no operative gate, freezes unused logits and
  skips mask sampling. No misleading p=0.1 gate heatmap/metric is emitted.

All models retain the SAME ordinary fixed-head agent and its local obs, one
TD-lambda objective, Double-Q selection, optimizer, target schedule and replay
configuration. Main b1 and V always see ones; no secondary bias/V is added.
No mask bits, field probes, extra TD, detach, KL, MC, L1, pretraining or freezing
of an active main/secondary branch is introduced. Acting remains decentralized
agent argmax and never calls the mixer. Targets and value diagnostics retain
the soft-p convention; nonlinear Q at expected input is not expected sampled Q.

The zero-fill name is reused from its standalone three-job submitter, so exact
active/completed names are retained instead of producing a duplicate cohort.
Other names: `smac_<map>_<label>_10m_s<seed>_weights20261020`.

## Weight and alpha displays

Every 100K environment steps, existing greedy evaluation diagnostics now emit
`test_value/mixer_<w1|w_final>_<part>_mean`, `_std`, and `_norm_mean`:

- `raw_main`, `raw_secondary`: signed generated heads BEFORE combination.
- `main_contribution`, `secondary_contribution`: actual weighted contributions;
  bounded secondary is `abs(A)*0.1*tanh(delta)`, not raw `R(x)`.
- `combined`: sum before final abs; `effective`: actual nonnegative mixer W.
- `test_value/mixer_<layer>_secondary_to_main_norm_ratio_mean` compares the
  per-transition Frobenius norms (denominator floor 1e-8).
- Alpha model adds `test_value/mixer_branch_alpha_mean` and training curves
  `mixer_branches/alpha_main` and `mixer_branches/alpha_secondary`.

At the existing 100K learner snapshot interval, add
`mixer_branches/weight_components_heatmap`: W1 and W2 rows, six labeled columns.
Raw heads share color limits; contributions share another scale. Titles include
alpha if present and mean Frobenius norms. JSON preserves per-entry mean/std
and norms. Diagnostic data are first up to 256 VALID replay transitions with
deterministic soft-p inputs; these are NOT importance scores or sampled-mask
expectations. Test scalar data use valid greedy evaluation transitions.

Local images/JSON: `results/mixer_onefill_gate/<unique_token>/`.
These paths and W&B keys do not alter optimizer graphs or acting decisions.
Old default checkpoint contracts/state_dicts remain compatible; new architectures
carry distinct strict contracts including alpha initialization and combination.

## Submit

On the server, after code is pushed:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_weight_composition_15jobs.py
```

Without `SUBMIT=YES` this is plan-only. It retains exact active/completed jobs,
validates both installed maps and the unchanged 2-GiB quota startup guard,
runs synthetic smoke tests, then `sbatch --test-only` for ALL missing jobs
before actual submission. It writes an incremental job-ID manifest under
`ozstar_logs/weight_composition_10m_15jobs_20261020_*.json`.
Quota guard is not a full-batch capacity guarantee; no deletion is performed.

Running-only tmux sync/explicit ended backfill and storage preservation recognize
the new group. No new tmux service is created by this change. General local
three-seed/mean3 plot inventories include all five models under the correct map,
without duplicate zero-fill entries (108 exact names total: 75 on 5m6m, 33 on
8m9m). This is registration, not a claim of uploaded results.

## Verification

`scripts/smoke_test_weight_composition.py` uses synthetic padded episodes with
the actual 5m6m/8m9m dimensions, Adam and RMSprop. It tests formulas, abs order,
strict alpha bounds, secondary first-step gradients, later gate updates,
monotonic Q derivatives, zero-fill input derivatives, bounded relative weights,
closed-input identity after secondary changes, both-ones state/RNG invariance,
old defaults, target/eval diagnostic routing, resume and contract rejection,
native heatmap/JSON/W&B buffers, matched submission differences, inventory and
mocked idempotent Slurm preflight. No simulator training, real scheduler calls
or cloud uploads are made by smoke tests. Python 3.9 syntax compatibility is
checked separately from local Python 3.12/Torch 2.8 CPU execution.

## Upload the completed cohort

The user supplied Slurm accounting for IDs **18904502–18904516** on October 21:
all 15 are `COMPLETED`, exit `0:0`. This establishes scheduler completion, not
10M test-history coverage or cloud upload completion.

On the server, upload ONLY this cohort's original runs, two-map overviews,
four 5m6m model panels plus one 8m9m panel, and five selectable mean3 runs:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_upload_weight_composition_results.py
```

The uploader reuses verified original IDs, requires terminal manifest/accounting
matches, checks exact original cloud scalar progress and backfills incomplete
originals with bounded per-run timeouts. It never scans all cloud histories or
launches training/tmux/cleanup. Scalar-current checks do not certify every media
object. Local figures use raw local Sacred/offline data and retained CSVs;
latest attempts are not stitched. Missing/partial seeds are explicit errors,
not fabricated 10M curves. Mean3 uses unsmoothed data, seeds 1/2/3 on each
model's shared interval; figure smoothing remains window=10 test points.

Report: `figures/weight_composition_20261020_3seeds/upload_report.json` under
the runtime. It includes original audit, inventories/endpoints, image/workspace
IDs and mean3 URLs/errors. If layout fails after image upload, retry only layout
using `model_panel_upload.json`; do not confuse partial success with completion.
`--no-upload` produces local plots/inventory only, without cloud or scheduler
calls. `--skip-original-sync` intentionally skips original backfill and is only
for retrying analysis after original records have been verified.

`smoke_test_upload_weight_results.py` validates both maps, exact 15-run scope,
5 model panels and 5 unsmoothed mean3 runs using synthetic logs and mocked
cloud/scheduler calls. `smoke_test_upload_onefill_results.py` preserves the old
October 19 single-map upload contract. Neither test uploads experiment data.
