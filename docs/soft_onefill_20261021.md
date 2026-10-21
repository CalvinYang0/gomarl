# Deterministic soft one-fill (2026-10-21)

The new `soft_onefill_gate_learner` is an opt-in single-mixer ablation of the
historical `qmix_onefill_p10`, not a replacement for any old run.

For each scalar coordinate of the global mixer state, learn an unconstrained
static logit `a[j]`. All episodes and timesteps share that coordinate's logit.
The exact online, target and diagnostic forward is:

```
p[j] = sigmoid(a[j])
x[j] = p[j] * state[j] + (1 - p[j])
Qtot = QMIX(agent_qs, x)
```

Initialize `a=log(0.1/0.9)`, giving `p=0.1`. No Bernoulli/Binary-Concrete
sampling, straight-through gradient, threshold, temperature, softmax, manual
clipping, KL, MC, field probes or additional gate loss. Joint historical TD
updates actor, complete mixer and logits using ordinary autograd. All four
mixer state paths (W1, W2, b1, b2/value) receive the same controlled input.
The target mixer has its own lagged logits, copied with the other parameters
at the unchanged target-update interval; same formula does not mean identical
online and target parameter values between syncs.

The actor's observations, GRU, and historical **two-layer ELU action head** are
unchanged. This is NOT the separately corrected official PyMARL2 baseline.
No residual or second branch is enabled. Old stochastic one-fill, weight-level
and Qtot-level versions remain unchanged. Checkpoint contracts reject loading
a historical hard-ST run as this new experiment.

Diagnostics use `mixer_input/soft_gate_value_mean`, `soft_gate_value_sum`, and
`soft_onefill_gate_heatmap`; JSONs live under `results/mixer_soft_onefill_gate/`.
These are interpolation coefficients, not sampled binary masks or established
importance scores. Small nonzero coefficients still carry state information;
downstream layers can compensate by increasing weights. There is no sparsity
guarantee and no performance claim from synthetic QA.

## Checks

Run `python scripts/smoke_test_soft_onefill_gate.py` in the project environment.
The test checks the exact formula and derivatives, Qtot finite-difference
gradients, unchanged initialization, deterministic train/target/eval with no
random-mask calls, all four mixer paths and actor/gate TD updates, target sync,
old-checkpoint rejection, exact continued training after checkpoint reload,
diagnostics and three-seed plan identity. It uses synthetic episodes and never
launches StarCraft, uploads W&B data or submits cluster jobs.

2026-10-22 preflight fix: training also saves an earlier gate snapshot. The QA
now selects the requested step/update filename and checks its metadata instead
of taking the first filesystem-dependent glob result. Numeric comparison uses
explicit float32 tolerances. This changes only QA, not training or its losses.

## Optional 5m6m three-seed run

The launcher is plan-only by default. It preserves the old p10 recipe and changes
only the learner, run identity and 24h allocation; the hard-gate temperature
override is removed. Three seeds, approximately 10M environment steps.

```bash
REPO_DIR="$PWD" python scripts/ozstar_submit_5m6m_soft_onefill_3seeds.py
```

Once these files are present in the server checkout, submit explicitly:

```bash
cd /home/kyang/code/gomarl-dual-branch &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u \
  scripts/ozstar_submit_5m6m_soft_onefill_3seeds.py
```

New model name: `qmix_soft_onefill_p10`, batch `soft20261021`. Existing exact
jobs are retained by the shared submitter; no old jobs are cancelled or renamed.
