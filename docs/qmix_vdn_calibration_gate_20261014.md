# Fresh 10M baselines and return-calibration-based mixer grouping

18 new jobs: seeds 1/2/3 for each model/map below.

| Map | Models | Jobs |
| --- | --- | --- |
| 5m_vs_6m | ordinary-head QMIX and VDN | 6 |
| 8m_vs_9m | ordinary-head QMIX and VDN | 6 |
| 5m_vs_6m | ordinary agent, two mixer groups, masking calibration improvement, isolated auxiliary gradients | 3 |
| 8m_vs_9m | ordinary agent, static mixer state gate p0=.1, no KL | 3 |

New names end `fresh20261014`. Group:
`qmix_vdn_calibration_gate_10m_3seeds_fresh20261014`. Historical names are not
reused, so old completed runs do not prevent these reruns. Repeated invocation
retains active/completed exact new names; never cancels jobs. T_MAX=10,050,000,
test every 10k, value diagnostics every 100k, ten battle videos every 1M, named
gate/group heatmaps and JSON. Ordinary fixed action heads; standard QMIX mixer
hypernetworks remain.

## Corrected metric: prediction error, NOT statistical variance

The user clarified that “variance” means distance from a theoretical value.
True expected/optimal return is unavailable. Use complete replay episodes'
observed discounted return G as a Monte Carlo proxy, without target-Q bootstrap.
Incomplete/time-limit-only episodes are excluded. This estimates calibration
against the behavior policy's sampled returns, not true optimal Q, and noisy
returns/off-policy differences remain possible confounders.

For up to 256 valid transitions at one fixed checkpoint, hold chosen actions,
agent Q and G fixed. Remove a slot from its owning mixer branch's state mask,
including both value and membership bit. Define:

```
G_t = sum_k gamma**k * reward_(t+k), through the natural terminal
error_original_j = mean((Q_tot - G)**2)
error_masked_j = mean((Q_tot_without_slot_j - G)**2)
removal_gain_j = (error_original_j - error_masked_j) / max(mean(G**2), 1e-6)
slot_quality_j = sigmoid(-clip(removal_gain_j, -20, 20))
```

The common return-based denominator preserves direction and does not allow
inflated predicted Q to suppress the score. Raw before/after MSE is also logged.
Positive gain: removing the slot improves predictions; the ORIGINAL slot is
less desirable. Negative gain: removing it makes predictions worse; prefer to
retain it. Gains remain signed, never clipped to zero. No centered statistical
variance is used; a constant prediction bias must contribute to error.

Every 1000 optimizer updates, EMA=.8 of slot quality selects the highest-quality
slot outside the important group. Demote the lowest-quality important slot if
below half the group's highest quality, preserving at least one slot. Initial
important group is empty. Both complementary branches have separate own TD
targets/losses. Important TD updates the ordinary agent and important mixer;
unimportant TD receives detached agent utilities and updates only its mixer.
The loss retains historical 0.5/0.5 weights, but each parameter set has its own
optimizer and gradient clipping, so a large auxiliary gradient cannot rescale
the actor's update through a shared clip norm. Acting and Double-Q
action selection still use that agent, never the mixer. No complete episodes
means skip migration. The score is a calibration heuristic, not proof that a
slot caused overestimation or that removing it improves win rate. Reassignment
changes the branch being probed; reset reassigned slots' EMA initialization.

Isolation is explicit: `mixer_input_detach_unimportant=True`. Historical config
defaults to False so old jobs remain reproducible. New isolated group run names
contain `qmix_state_group_calibration_isolated`, unlike the previous shared-gradient
names; they will not be mistaken for already completed runs. Baseline/gate names
are unchanged. Existing jobs are not cancelled or modified. Checkpoint schedules
record routing mode; resuming across modes is refused before loading parameters.
The auxiliary optimizer is stored in `opt_unimportant.th`. Heatmap JSON records
`unimportant_agent_gradient_detached`.

JSON includes `normalized_removal_gain`, `original_mc_mse`, `masked_mc_mse`,
`ranking_ema`, `migration`, masks and state-slot names. W&B includes
`mixer_input/partition_probability_heatmap`, `mixer_input/calibration_probe_count`,
`mixer_input/normalized_removal_gain_mean`, `mixer_input/important_count`.
The `mixer_input/` scalar allowlist is fixed (previously only direct image logs
were uploaded). Track win rate and `test_value/q_mc_bias_mean`, plus each branch's
calibration diagnostics; magnitude of agent Q alone is not performance.

8m9m gate is static across observations, learned per state field, p0=.1, no KL;
evaluation uses p>.1. The heatmap's second row is the actual evaluation mask.

## Server submission

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_qmix_vdn_calibration_gate_18jobs.py
```

Omit `SUBMIT=YES` for preview. Both installed map definitions/files are checked;
minimum 2 GiB free /home quota is only a startup guard, not sufficient capacity
for 18 video-enabled jobs. Manifest/job IDs go to runtime `ozstar_logs`. The
local cleanup tool also preserves this new cohort after completion.

## 8m9m QMIX and its actually submitted mixer variant

Only ordinary QMIX and static-gate QMIX have been submitted on 8m9m in this
cohort; group-calibration is 5m6m-only. The dedicated plot reads exactly these
six fresh run names, not VDN or old cohorts. Default smoothing is ten test
points, aggregation uses each model's shared available seed interval, and
missing seeds/partial budgets are explicitly labelled.

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_8m9m_qmix_variants_3seeds.py --local-only --no-upload --mean-window 10
```

Outputs under runtime `figures/8m9m_qmix_variants_3seeds`:
`smac_8m9m_three_seed.png/.pdf`, `smac_8m9m_individual_seeds.png`,
`seed_inventory.csv`, raw and aggregate curve CSVs. Remove `--no-upload` to
upload the figure snapshot to W&B; `--local-only` still prevents cloud history
queries. No scheduler-state filtering or synthetic data in production figures.

## Both 5m6m and 8m9m, with W&B figure upload

```bash
cd /home/kyang/code/gomarl-dual-branch
git pull --ff-only origin codex/linear-id-baselines
/home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/plot_5m6m_8m9m_qmix_variants_3seeds.py --local-only --mean-window 10
```

This reads 27 exact runs: 5m6m fresh QMIX/VDN, original static gate/Q-change/TD-error
mixer controls, and the new return-error gradient-isolated group; 8m9m fresh
QMIX/VDN and static gate. Three planned seeds each. Shared-gradient and isolated
group models are separately labelled; no historical VDN, hyper-generated action heads, or
other-map substitutions. Missing local curves stay missing and seed coverage
is explicit. Both maps' mean/std figures, per-model individual-seed images and CSVs go
to one W&B figure snapshot; `Uploaded figures:` prints its run URL. Local output
directory is runtime `figures/5m6m_8m9m_qmix_variants_3seeds`.

### Readable W&B sections, instead of a long montage

The two-map uploader now exports one 8-inch-wide image per model. The three
seed curves stay together in each image; missing seeds and endpoints are still
explicit. It creates a NEW named project workspace, filtered to the new figure
snapshot run ID, with sibling sections:

- `Media`: collapsed by default, contains the two mean/std comparison figures.
- `5m6m`: open, one column, six ordered per-model three-seed images.
- `8m9m`: open, one column, three ordered per-model three-seed images.

No existing workspace is overwritten and no horizontal montage is uploaded.
Use the printed `Model sections workspace:` link, not the old run's Media panel.

Install the official layout-only library once (Python 3.9/W&B 0.18.7 compatible;
`--no-deps` deliberately avoids upgrading the training environment):

```bash
/home/kyang/.conda/envs/marl_cpu/bin/python -m pip install --no-deps wandb-workspaces==0.4.5
```

Then use the same two-map plotting command above. If workspace saving fails
AFTER image upload, `model_panel_upload.json` retains the run ID and image keys;
the error prints a layout-only retry command that does not re-upload data.
Official API: https://github.com/wandb/wandb-workspaces

Offline checks:

```bash
python scripts/smoke_test_plot_two_map_qmix_variants.py
python scripts/smoke_test_wandb_seed_sections.py
```
