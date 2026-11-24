# Frozen-mask retraining and nuisance-state experiment

## Matrix

Maps: `corridor`, `3s_vs_5z`, `5m_vs_6m`. Each setting uses fresh seeds 1/2/3,
10M evaluation horizon (training guard 10.05M), and a 48-hour allocation.

| Experiment | Model | Jobs |
| --- | --- | ---: |
| Fixed-mask transfer | QMIX with the user-shown binary mask fixed from initialization | 9 |
| Nuisance state | Ordinary QMIX with D extra noise fields | 9 |
| Nuisance state | Continuous projected init1 gate with original faststages8m | 9 |

No actor/checkpoint weights are transferred. Actor observations and action heads
remain unchanged. Fixed-mask runs have no curriculum and no trainable gates.

## Exact masks

`src/config/masks/frozen_heatmaps_seed3_20261124.json` transcribes the three
user-supplied seed-3 heatmaps, with image SHA256 and source step provenance.
All three NEW training seeds on each map use that map's same mask.

| Map | D | Closed indices (zero-based) |
| --- | ---: | --- |
| corridor | 282 | 4,30,33,36,39,42,45,48,51,54,60,63,66,69,72,81,87,90,93,96,99 |
| 3s_vs_5z | 68 | 0,5,15,19,23,27,30,31 |
| 5m_vs_6m | 98 | 4,8,12,16,26,29,35 |

Preserve historical one-fill: `x = mask*state + (1-mask)`. A closed gate replaces
its feature with **1**, not 0. Submission pins the bundle SHA256; changed map
dimensions or a modified bundle fail closed. These masks are hypotheses about
useful mixer conditioning, not validated semantic feature-importance scores.

## Noise protocol

Input is `[state, noise]`, dimension 2D. The tail is iid Uniform[0,1), independent
of state, action and reward. Private RNG streams are deterministic by experiment
noise seed, training seed, worker index and train/eval mode. Evaluation does not
consume training noise RNG. A new sample is generated per simulator timestep,
cached across repeated state reads and stored in replay (including terminal
state). Online and target mixers therefore see the same collected noise.

The noisy QMIX control is identity-input QMIX; the implementation's fixed
all-ones **gate** means `x=state`, NOT the historical `qmix_state_ones` baseline.
Both noisy models use 2D hypernetworks. Their dimension increase itself changes
parameter counts relative to the original D-state runs.

At the user's request, the gated variant preserves the full original 8M
sample-and-hold curriculum on the concatenated state; evaluation is always live.
Thus its comparison with live noisy QMIX measures robustness of **gate plus
curriculum**, not an isolated gate effect. Noise is not resampled inside TD
updates. Extra real/noise gate-mean metrics and noise gate fraction below 0.1
are recorded. Small soft coefficients alone do not prove information removal:
downstream weights may compensate, unlike an exactly closed binary gate.

## Scope and execution

The submitter is plan-only unless `SUBMIT=YES`; it retains quota/inode guards,
asset validation, synthetic learner preflight and existing duplicate checks.
Supports `--experiment mask`, `--experiment noise`, or default `all`.
No upload, deletion, cancellation or checkpoint resume is performed.
Videos are disabled for this batch; the current request does not ask to repeat
the prior video experiment. Real SC2 episodes are not launched by the local
synthetic test; learning success and 10M completion must be checked afterwards.

Fixed-mask success should be assessed against existing full-state and all-one
baselines with matched horizons. It would support mask reuse from a separate
run; a random mask with matched cardinality would be an additional, not yet
requested control for whether the exact selected coordinates matter.
