# Two batches: clean 5m6m ablations and 8m9m gate controls

Six models, each with seeds 1/2/3 and 10M environment steps. The combined command
has 18 exact job names, keeps already active/completed jobs, and submits only
missing runs. No baseline reruns, cancellation, deletion or historical merging.

| Map | Model | Change relative to its matched control |
| --- | --- | --- |
| 5m6m | Q-change isolated | Auxiliary/unimportant TD detached from agent |
| 5m6m | Q-change plain mask | Only remove appended mask features: state × group mask, 98 dimensions |
| 5m6m | Q variance groups | Only change grouping ranking to signed population variance difference |
| 8m9m | QMIX ones | Replace the mixer condition with 179 ones; no gate; agent observations unchanged |
| 8m9m | Static gate + L1 | Add 0.01 × mean(keep probability), no variance loss |
| 8m9m | Static gate + variance | Add 0.01 × variance-weighted keep probability, no L1 |

5m6m variants are independent ablations of the old Q-change/shared-gradient
control, not cumulative. See `5m6m_qchange_three_ablation_20261015.md`. In
particular A1/A3 retain the legacy mask-concat encoding; only A2 fixes it. Their
single-field batch-mean perturbation remains identical to the old control.

8m9m controls clone the fresh October 14 static-gate setup: ordinary fixed agent
head, original 179-dimensional QMIX state condition, no appended masks, p0=0.1,
same stochastic relaxed training gate, deterministic probability for TD targets,
and evaluation mask p>0.1. QMIX's usual state-conditioned weight generators stay;
these changes do not restore an observation-conditioned agent head.

## Gate losses

Let p_j=sigmoid(logit_j). L1 is mean(p_j), not L1 on logits or sampled gates. It
encourages smaller keep probabilities, while TD may oppose closing useful gates.

For the variance-only gate variant, on a fixed valid replay probe set with frozen
executed-action agent Q, compute:

```
d_j = Var(Q_tot(p × state)) − Var(Q_tot(p × state with field j zero))
c_j = stop_gradient(max(d_j, 0))
L_gate_variance = mean_j(p_j × c_j)
L_total = L_TD + 0.01 × L_gate_variance
```

Population variance uses unbiased=False. The probability p used for scoring is
deterministic, not a new noisy mask or hard test mask. Only this new gate-loss
experiment uses zero masking for the variance attribution; it does not change
the preceding 5m6m ablations' perturbation protocol.

Positive variance-removal gain means the field contributes to dispersion and
therefore receives stronger pressure to close, as requested for this experiment.
Negative gains are logged but not rewarded by the regularizer. For the penalty
alone, the logit derivative is c_j*p_j*(1-p_j)/179. Higher c_j therefore gives a
stronger closing gradient at equal probability. The TD gradient can counteract
this: total training does not guarantee monotonically decreasing gates.

The variance coefficients have no gradients into agent or mixer parameters, so
the extra objective cannot directly shrink all agent Q to reduce variance. TD
still trains those networks normally. This is not a claim that variance equals
overestimation or useless information. Useful state variation can also produce
high variance; compare performance and gate coverage rather than assuming the
penalty is necessarily beneficial. Too much penalty can close useful gates.

Start with coefficient 0.01 for both separate penalty variants. Record raw,
weighted and total losses. Variance probes refresh before the first update and
every 100 optimizer updates (up to 256 valid samples); coefficients are cached
between probes. With fewer than two valid samples, clear the coefficients.
Probe update number and probabilities are saved, avoiding confusion with the
later heatmap probabilities. Snapshot JSON also retains raw signed gains, both
variances, positive penalty coefficients and probe counts. Both gated variants
upload probability/test-mask heatmaps to their native W&B run.

## Run both batches

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_qchange_gate_ablation_18jobs.py
```

If the previous 5m6m batch is already present, this launches only nine new 8m9m
jobs. Otherwise it launches missing jobs up to all 18. The already-declared
Q-change/isolated names are reused, not renamed. Submission runs smoke tests and
all Slurm preflights before actual submission. The 2 GiB quota threshold is only
a startup guard, not a batch storage capacity guarantee.

8m9m-only alternative:

```bash
cd /home/kyang/code/gomarl-dual-branch
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_8m9m_ones_gate_regularization_9jobs.py
```

Optional: reload the existing suite-sync uploader to include new manifests. This
restarts only its uploader window, never training jobs:

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_calibration_suite_sync.py restart
```

The two-map plot includes the new models in its individual 5m6m/8m9m workspace
sections. Plot and upload available data:

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_5m6m_8m9m_qmix_variants_3seeds.py --local-only --mean-window 10
```

The combined suite does not submit fresh QMIX/VDN baselines or rerun old gate
controls. They remain reference curves in the plot. Missing seeds are reported,
not filled with older runs. No server jobs are submitted by local code tests.
