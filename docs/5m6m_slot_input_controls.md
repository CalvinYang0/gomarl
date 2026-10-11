# 5m6m slot-input controls — 12 new runs

Scope: map `5m_vs_6m`, seeds 1/2/3, target 10M (`t_max=10050000`),
test every 10K. No historical run reuse, cancellations, KL, sparsity loss,
distillation, or preliminary scientific attribution-validation experiments.
All models retain matched full-Obs/last-action/agent-ID shared policy GRU.
Only the hypernetwork condition input is controlled.

| Exact job prefix (append `_10m_s{1,2,3}_slotcontrol`) | Treatment | Memory |
| --- | --- | --- |
| `smac_5m6m_linear_static10_nokl` | One Linear hyper-head; static shared slot gate, p0=0.1 | 24G |
| `smac_5m6m_linear_group_forward_hierarchy` | Two heads; initially important empty; hierarchical TD | 48G |
| `smac_5m6m_linear_group_forward_dualtd` | Two heads; initially important empty; two equal-weight TD losses | 48G |
| `smac_5m6m_linear_group_reverse_hierarchy` | Two heads; initially important full; one-way pruning, hierarchical TD | 48G |

The requested single reverse variant is implemented on the hierarchical route,
not an additional fifth dual-TD model. Grouped runs have larger memory requests
because two generated heads are differentiated through full replay sequences.

## Gate and partition

Static gate: trainable observation-independent scalar per raw Obs slot, shared
across agents. Binary Concrete temperature 0.5 in training, strict `p > 0.1`
hard mask in evaluation; no force-open warmup. Target gate uses deterministic
soft probabilities. No gate penalty or auxiliary TD objective.

Groups: two independent Linear encoders and four independent generated weight/
bias projections per head, one shared GRU. Complementary fixed-width masks
are broadcast across agents. Membership changes, not parameters, migrate;
there is no optimizer reset or weight transfer. The unimportant head is not
removed during training.

Every 1,000 **optimizer updates**, not environment timesteps, replay up to ten
training episodes; select at most 256 deterministic agent-stratified live
probes (two or more legal actions), excluding padding/post-terminal states.
For each slot, perturb only its owner's hyper-input using the population mean.
Entity health/geometry means are computed over visible entities and applied
only to visible probes; visibility fields themselves can become fractional.
GRU hidden state and legal-action masks stay fixed. Original chosen-action vs
strongest-alternative margin is recomputed, including a changed rival.

Score = mean absolute margin change / common owning-head RMS centered legal-Q
scale over the probe population. Near-zero scales are skipped; scores are
nonnegative. This cancels common additive offsets and positive utility scaling,
but is learned hyper-input dependence, not causal benefit or win-rate loss.
Comparing two normalized heads remains a heuristic. EMA decay 0.8; restart
a slot's EMA on ownership change.

Forward: promote at most the highest nonzero-score unimportant slot; demote at
most the lowest-score important slot if its score is strictly below half the
highest important score. Both use the pre-migration partition. Empty/all-zero
groups do not cause arbitrary promotions. Reverse: only the demotion rule,
no forced re-promotion. Persist before/after membership and selected indices.

## Training and acting

Hierarchical: each small QMIX receives its group's masked Obs for all agents
plus membership bits, and that head's executed-action utilities. It never
receives global state or other-group Obs. Big QMIX receives the two small
values and global state. One final joint TD-lambda loss.

Dual TD: each small QMIX has its own TD-lambda loss against the same reward.
The total optimization loss is their sum (equal weights); both update the
shared GRU, each its own encoder/head/mixer. Do not sum their Q values as an
execution policy. No big mixer on this route.

Behavior (including epsilon-greedy exploration), evaluation and online
Double-Q argmax all use important head zero. Both target heads evaluate those
same selected actions. Target network weights follow matched hard-update
cadence; partition buffers sync immediately after a migration. A hierarchical
joint critic with important-head-only acting does **not** guarantee that the
acting action maximizes the full joint critic; this is the requested convention.

## Required outputs and submission

- Battle videos: every 1M, ten greedy test episodes.
- Full policy/input/parameter importance: every 1M, ten episodes, 64 probes.
- Value diagnostics: every 100K; grouped joint value uses real group outputs.
- Gate probability and exact hard-mask heatmap, or two group-membership
  heatmaps; every 100K and every regroup. Slot scores, flip rates and
  migration JSON saved locally; images and JSON registered with W&B.
- Both 5m6m and multi-map plot inventories include all four exact cohorts.

Legacy ungated-single-head hyper-importance is disabled for these new models;
`slot_input/` is their fixed-hidden hyper-input diagnostic. Full-policy
importance remains enabled. Missing required media fails rather than claiming
coverage. Offline logging does not assert cloud upload completeness.

```bash
REPO_DIR="$PWD" python -u scripts/ozstar_submit_5m6m_slot_input_controls_12jobs.py
SUBMIT=YES REPO_DIR="$PWD" python -u scripts/ozstar_submit_5m6m_slot_input_controls_12jobs.py
```

Submission validates the installed map, config keys, code/media smoke tests,
at least 3 GiB home-quota headroom, and all `sbatch --test-only` plans before
submitting anything. Exact-name active/completed jobs are retained. Real IDs
are durably recorded in the suite manifest after each submission. Simulator-
free smoke tests use synthetic episodes, do not upload, and are not evidence
of scientific effectiveness or successful cluster training.

The user-requested 3 GiB startup threshold replaces the original 5 GiB guard.
Exactly 3 GiB is accepted; below 3 GiB still stops before any job submission.
The submitter prints measured free space and a capacity warning: meeting this
threshold does not guarantee sufficient space for the full media-enabled runs.
