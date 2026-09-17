# HyperSelect Paper Handoff

Last updated: 2026-09-17

Code snapshot used for this handoff: `0890c04`
Primary implementation profile: `relation_advantage_qvalue_augtd_nomasktd`

Paper architecture figure:
[editable SVG](../figures/hyperselect_architecture.svg) and
[high-resolution PNG](../figures/hyperselect_architecture.png).

This document is the source-of-truth handoff from the implementation and
experiment thread to the paper-writing thread. It records the agreed paper
story, method naming, exact implemented loss paths, experiment lineage,
current evidence, safe claims, and the remaining validation plan.

The document intentionally separates three kinds of statements:

- **Implemented fact**: directly supported by the current code.
- **Observed indication**: seen in the current single-run plots, but not yet a
  multi-seed paper result.
- **Paper hypothesis**: the mechanism we intend to test, not a conclusion that
  can be asserted before the corresponding experiments finish.

## 1. Working title and terminology

### 1.1 Working method name

**HyperSelect**

Working paper title:

> **HyperSelect: Value-Guided Adaptive Observation Masking for
> Hypernetwork-Based Multi-Agent Reinforcement Learning**

Chinese title:

> **HyperSelect：面向超网络多智能体强化学习的价值引导自适应观测掩码**

Do not use `HyperMask` as the method name. That name is already used by a
continual-learning method. Do not use `Q-Mask` either: it collides with prior
terminology and incorrectly makes Q-value maximization appear to be the main
research object. The main research object here is the observation mask.

### 1.2 Module names

| Paper name | Abbreviation | Current implementation role |
|---|---:|---|
| Adaptive Observation Masker | AOM | Observation-conditioned learned entity gate before the hypernetwork condition encoder |
| Q-Guided Mask Evaluation | QME | Gate-only Direct-Q objective evaluated on the detached open-view teacher action |
| Stochastic Mask Exploration | SME | Learned KL80 auxiliary mask stacked with the main mask, plus augmented TD |
| Open-View Value Anchor | OVA | Force-open/no-main-mask TD branch that supplies a trained Q function, teacher action, and confidence |

One-line description:

> **AOM generates masks, SME explores masks, and QME evaluates masks.**

OVA is a supporting branch inside QME rather than a separate headline
contribution.

### 1.3 Loss names

Use the following paper-facing notation:

| Paper notation | Current code/config terminology |
|---|---|
| `L_OVA-TD` | NoMaskTD, `clean_nomask_td_auxiliary_coef=1` |
| `L_SME-TD` | KL80 random-drop augmented TD |
| `L_SME-KL` | Bernoulli KL of the learned auxiliary gate to keep prior 0.8 |
| `L_QME` | `advantage_objective=action_q`, gate-only Direct-Q loss |

For the current full HyperSelect profile:

```text
L_HyperSelect = L_OVA-TD
              + lambda_SME * L_SME-TD
              + beta * L_SME-KL
              + lambda_QME * L_QME
```

There is **no ordinary main-mask TD loss** in this profile.

## 2. Core paper story

### 2.1 Background: why hypernetworks

Fully shared parameters in cooperative MARL improve sample efficiency,
parameter efficiency, and scalability, but can restrict agent specialization.
Independent parameters permit specialization but lose much of the sharing
benefit. Hypernetworks provide a middle ground: all agents share one parameter
generator, while different conditions generate different decision-network
parameters.

The motivating progression is:

```text
agent identity -> agent capability -> dynamic observation context
```

- HyperMARL conditions a hypernetwork on agent identity or agent embedding.
- CASH conditions policy generation on capabilities and local observations to
  adapt to heterogeneous agents and changing contexts.

The paper should acknowledge these as motivation, not claim that HyperSelect is
the first hypernetwork MARL method.

### 2.2 The higher-level problem

When an ordinary policy consumes an observation, irrelevant input changes its
activation. When an observation-conditioned hypernetwork consumes that input,
the input can change the generated decision parameters themselves:

```text
observation -> generated parameters -> decision function -> action values
```

Richer conditions increase adaptivity, but uncontrolled conditioning can make
the generated decision function sensitive to decision-irrelevant or
distracting context.

The central question is therefore:

> **Which observations should be allowed to participate in generating the
> policy itself?**

This is a stronger and more general framing than "observation denoising."

### 2.3 Natural baseline and failure mode

A natural solution is to insert an observation-conditioned gate before the
hypernetwork:

```text
m_t = AOM_phi(o_t)
o_t_masked = m_t * o_t
theta_t = H_psi(o_t_masked)
```

**Observed indication:** when a naive observation gate is trained only through
the task TD objective, its keep probabilities often drift toward an all-open
solution and performance can be poor.

**Paper hypothesis:** TD learning alone provides weak counterfactual evidence
about what happens when information is removed. Keeping an entity is locally
safer than closing it, so the gate may not explore alternative masking
decisions.

Do not present the hypothesized cause as proven until the Gate / Gate+SME /
Gate+QME / full ablation is complete.

### 2.4 Conceptual reformulation

The agreed conceptual contribution is:

```text
mask learning = mask-space exploration + decision-value evaluation
```

This is **not** a sparsity-maximization paper.

- If all observed information is useful, an all-open mask is valid.
- If only some information is useful, the gate should close selectively.
- If most information is unhelpful, the learned mask may become sparse.

The method does not prescribe the final sparsity level. It attempts to select
information according to current state and decision value.

### 2.5 Role of each component

#### Adaptive Observation Masker (AOM)

AOM produces a state-dependent entity mask before hypernetwork conditioning.
It is the main object being learned.

#### Q-Guided Mask Evaluation (QME)

The open-view branch produces a detached greedy teacher action:

```text
a_star = argmax_a Q_open(o, a)
```

It also produces a confidence weight from the normalized gap between the best
and second-best available actions. The masked branch is evaluated on the same
teacher action:

```text
L_QME = - confidence * Q_masked(o * m, a_star)
```

The QME backward path is restricted to observation-gate parameters. The value
network cannot minimize this loss by simply inflating its own Q scale.

Correct interpretation:

> QME provides a task-related evaluation signal for candidate masks.

Incorrect interpretation:

> Maximizing Q directly labels every removed feature as noise.

#### Stochastic Mask Exploration (SME)

SME samples an additional learned Binary-Concrete entity mask whose Bernoulli
probability is KL-regularized toward a keep prior of 0.8. In the full profile,
this auxiliary mask is multiplied on top of AOM and receives an augmented TD
loss.

Correct interpretation:

> SME explores alternative observation subsets and trains the value function
> on masked views, improving the reliability of mask evaluation.

Incorrect interpretations:

- SME is not the final learned mask.
- KL80 does not prescribe that the final AOM mask must keep 80% of entities.
- The current KL prior directly regularizes the auxiliary SME gate, not the
  main AOM gate.
- The method does not assume that a sparser final mask is always better.

## 3. Method equations

Let `o_t` be the structured local observation and let AOM produce entity-level
keep variables:

```text
m_t = AOM_phi(o_t)
o_t^m = m_t * o_t
theta_t^m = H_psi(o_t^m)
```

The open-view value anchor bypasses the main observation gate:

```text
theta_t^open = H_psi(o_t)
Q_open(o_t, .) = Q_{theta_t^open}(o_t, .)
```

OVA is trained with the ordinary detached TD target:

```text
L_OVA-TD = E[(Q_tot^open - y)^2]
```

The detached teacher action and confidence are:

```text
a_t^star = argmax_{a in A_t} Q_open(o_t, a)
c_t = sigmoid((margin_open - tau) / temperature)
```

The implemented QME loss is:

```text
L_QME = -E[c_t * Q_masked(o_t^m, a_t^star)]
```

`a_t^star`, the open Q values, and the confidence are detached. `L_QME`
updates only AOM parameters.

For SME, let `z_t` be the auxiliary learned Concrete mask:

```text
z_t ~ Concrete(p_aux(o_t), temperature)
o_t^aug = (m_t * z_t) * o_t
```

The auxiliary probability is trained with:

```text
L_SME-KL = KL(Bernoulli(p_aux) || Bernoulli(0.8))
```

and the composed masked view is trained using:

```text
L_SME-TD = E[(Q_tot^aug - y)^2]
```

The ordinary AOM-only masked TD path is disabled in the full model.

## 4. Exact implementation facts

### 4.1 Canonical profile

The canonical full model is:

```text
relation_advantage_qvalue_augtd_nomasktd
```

The `relation_` and `advantage_` parts of this label are historical. The
profile does not enable the old mask-parameter relation loss; its gate
objective is raw teacher-action Q (`action_q`), not the older margin objective.

Resolved key settings:

| Setting | Value | Meaning |
|---|---:|---|
| `gate` | true | Enable AOM |
| `aux` | `kl80` | Enable SME |
| `clean_main_td_coef` | 0 | Disable ordinary AOM-only MaskTD |
| `clean_nomask_td_auxiliary_coef` | 1 | Enable OVA-TD |
| `clean_advantage_objective` | `action_q` | Use QME Direct-Q objective |
| `clean_advantage_margin_auxiliary` | true | Build teacher/action-confidence machinery |
| `clean_advantage_margin_teacher_only` | false | Open branch also has NoMaskTD, not teacher-only forward |
| `clean_kl_auxiliary_force_main_open` | false | SME is stacked on top of AOM |
| `clean_kl_auxiliary_prior` | 0.8 | SME keep prior |
| `clean_dual_gate_test` | true | Evaluate masked and forced-open policies |
| `clean_mask_parameter_relation_coef` | 0 | Disable legacy relation loss |

### 4.2 Warmup

- AOM warmup: 250k environment steps.
- SME identity warmup: 250k environment steps.
- QME starts at 250k and ramps from 0 to full coefficient over the next 250k.

This matched warmup must be preserved in fair comparisons. Earlier runs that
did not use the same warmup should not be treated as controlled ablations.

### 4.3 Gradient paths

| Loss | Shared policy / Q / hypernetwork / mixer | Main AOM | Auxiliary SME gate |
|---|---:|---:|---:|
| `L_OVA-TD` | yes | no, because main gate is forced open | no |
| `L_SME-TD` | yes | yes | yes |
| `L_SME-KL` | no meaningful shared-model path | no | yes |
| `L_QME` | no | yes | no |

Important nuance: the current full profile does **not** globally separate every
masked loss from the shared value model. Only QME is explicitly gate-only.
SME-TD still trains the shared value components and both gates.

### 4.4 Evaluation paths

At each scheduled test, the same checkpoint is evaluated twice:

- **HyperSelect-Masked**: normal AOM execution.
- **HyperSelect-Open**: AOM forced fully open.

This paired test is the primary way to determine whether applying the learned
mask helps execution. The comparison is more informative than comparing
different independently trained runs.

## 5. Paper-facing model and ablation names

| Current run/profile | Paper-facing name | Purpose |
|---|---|---|
| naive learned observation gate | HyperSelect-Gate | Demonstrate the natural baseline and all-open failure mode |
| `relation_noadv_augtd_nomasktd` | HyperSelect w/o QME | Same OVA/SME/warmup/evaluation; remove only value-guided gate objective |
| Direct-Q gate without KL80 auxiliary | HyperSelect w/o SME | Test whether mask-space exploration/calibration is necessary; requires a matched profile/run if not already available |
| `relation_advantage_qvalue_augtd_nomasktd` | HyperSelect | Full method |
| full model, normal test | HyperSelect-Masked | Learned-mask execution |
| full model, force-open test | HyperSelect-Open | Same checkpoint without applied AOM |

Historical objective ablations:

| Profile family | Gate objective |
|---|---|
| `relation_advantage_*` default | Teacher-vs-competitor normalized margin hinge |
| `relation_advantage_actionadv_*` | Maximize normalized teacher-action advantage |
| `relation_advantage_qvalue_*` | Maximize raw masked Q of teacher action; current HyperSelect choice |

Historical TD-path controls:

| Profile suffix | Ordinary main MaskTD | NoMaskTD / OVA-TD | SME-TD |
|---|---:|---:|---:|
| `masktd_augtd_teacheronly` | yes | no; open pass only supplies detached teacher | yes |
| `augtd_teacheronly` | no | no; open pass only supplies detached teacher | yes |
| `augtd_nomasktd` | no | yes | yes |

## 6. Experiment lineage

### 6.1 Earlier all-loss and test-open stage

Earlier experiments studied combinations such as:

- `relation_all4`
- `relation_all4_testopen`
- `relation_all4_dualtest`
- `relation_all4_relcoef01`
- `relation_all4_relcoef01_gradsep`
- `relation_kl80aux_kltd_gateonly`

These runs established several useful lessons:

1. Masked and forced-open evaluation must be recorded from the same checkpoint.
2. Adding all available losses creates sampling, gradient-routing, and compute
   confounds.
3. The test-open behavior of a gradient-separated model is not automatically
   identical to a baseline because shared representations can still be trained
   through other paths unless every path is matched.
4. A large auxiliary loss value is not inherently good; its effect must be
   judged through policy performance and matched ablations.

Do not use `all4` as the main paper method. It is part of the experimental
lineage explaining why the final model uses a leaner TD-path design.

### 6.2 Margin, weighting, and gradient-separation stage

The next stage compared:

- `relation_advantage_margin_kl80aux`
- `relation_advantage_margin_kl80aux_gradsep`
- `relation_advantage_weighted_kl80aux`
- `relation_advantage_weighted_kl80aux_gradsep`
- `relation_kl80aux_kltd_gateonly`

Observed indications from the displayed Counter curves:

- Margin variants eventually reached stronger training/test performance than
  several weighted or gate-only controls.
- Gradient separation changed learning dynamics substantially and sometimes
  delayed convergence.
- The mask heatmaps differed sharply even between superficially similar
  `all4` and `all4_testopen` runs, showing that evaluation/training flags and
  gradient paths cannot be inferred from run names alone.

These are exploratory observations, not final multi-seed results.

### 6.3 Objective-by-TD-path matrix

Six matched Counter profiles were introduced to separate three gate objectives
from two TD-path designs:

| Gate objective | MaskTD + SME-TD + teacher-only open pass | SME-TD + OVA-TD |
|---|---|---|
| margin | `relation_advantage_masktd_augtd_teacheronly` | `relation_advantage_augtd_nomasktd` |
| action advantage | `relation_advantage_actionadv_masktd_augtd_teacheronly` | `relation_advantage_actionadv_augtd_nomasktd` |
| raw action Q | `relation_advantage_qvalue_masktd_augtd_teacheronly` | `relation_advantage_qvalue_augtd_nomasktd` |

OzSTAR jobs submitted for this matrix:

| Job ID | Profile |
|---:|---|
| 16599731 | margin + MaskTD + SME-TD + teacher-only |
| 16599732 | action advantage + MaskTD + SME-TD + teacher-only |
| 16599733 | raw Q + MaskTD + SME-TD + teacher-only |
| 16599734 | margin + SME-TD + OVA-TD |
| 16599735 | action advantage + SME-TD + OVA-TD |
| 16599736 | raw Q + SME-TD + OVA-TD; current HyperSelect |

The later queue snapshot showed all six jobs running. Some curves ended around
2M steps at the time of inspection; absence of later points must not be called
a crash without checking Slurm state and logs.

### 6.4 Current leading observation

**Observed indication, seed 1 only:** the orange
`relation_advantage_qvalue_augtd_nomasktd` run rose rapidly and reached very
high test win rate around the 1.5M--2M region compared with the simultaneously
displayed objective variants.

This suggests that:

- OVA-TD may provide a useful value anchor;
- Direct-Q may provide a strong gate-learning signal;
- omitting ordinary MaskTD may reduce conflicting training pressure.

It does **not** yet prove these mechanisms. The curve was shorter than several
comparison curves and was based on one seed. The strict no-QME control and
multi-seed replication are required.

### 6.5 Strict no-QME control

Profile:

```text
relation_noadv_augtd_nomasktd
```

This is the most important immediate control. It matches the current full
model's OVA-TD, SME, warmup, and dual evaluation while disabling only QME.

Paper comparison:

```text
HyperSelect       vs. HyperSelect w/o QME
```

This comparison measures the contribution of value-guided mask evaluation.

### 6.6 SMAC transfer runs

The full Direct-Q + SME-TD + OVA-TD profile has a dedicated SMAC submission
script for:

- `3s5z_vs_3s6z`
- `corridor`

Protected settings in the submission script:

| Setting | Value |
|---|---:|
| training horizon | 10,050,000 environment steps |
| test interval | 10,000 |
| parallel rollout environments | 8 |
| learner batch size | 128 |
| replay buffer | 5,000 episodes |
| learner updates per collect | 1 |
| requested memory | 96 GB by the dedicated submission wrapper |

These settings were explicitly preserved after an earlier corridor run was
accidentally shortened/reconfigured. Do not reduce the horizon, rollout batch,
or memory merely to make the job start sooner. Runtime/resource changes must be
reported as separate experiments.

## 7. Current metrics and how to interpret them

### 7.1 Primary performance metrics

- `game_win_mean`: training/evaluation behavior under the run's standard path.
- `test_game_win_mean`: scheduled test performance with the learned mask.
- forced-open test metric produced by dual-gate evaluation: same checkpoint,
  AOM bypassed.

The paper must state exactly which metric corresponds to masked and open
execution. Do not compare a training win-rate curve from one run to a test
win-rate curve from another.

### 7.2 Direct QME diagnostics

The learner logs paired statistics using the same replay state, recurrent
context, available actions, and detached open-view teacher action:

- `teacher_action_q`
- `masked_action_q`
- `action_q_gain_mean`
- `teacher_action_advantage`
- `masked_action_advantage`
- `action_advantage_gain_mean`
- `teacher_margin`
- `masked_margin`
- `margin_gain_mean`
- `margin_shortfall_mean`
- `margin_improve_rate`
- `margin_harm_rate`
- `margin_target_met_rate`
- `confidence`
- `high_confidence_margin_gain`
- `high_confidence_improve_rate`

Interpretation:

- Positive `action_q_gain_mean` means the mask increased the model-estimated Q
  of the open teacher action on the paired batch.
- It does not by itself prove higher environment return.
- The convincing evidence chain is paired Q gain plus masked/open action
  behavior plus actual masked/open test return.

### 7.3 TD and auxiliary losses

`weighted_loss_nomask_td_auxiliary` is a loss and is not "higher is better."
Very high or increasing TD error usually indicates larger prediction mismatch,
but different weighting/scales make cross-run raw magnitudes non-comparable.

`weighted_loss_mask_parameter_relation_bundle` and historical relation losses
are optimization terms, not direct performance scores. Their absolute values
should not be ranked as if larger or smaller necessarily meant a better policy.

### 7.4 Mask visualizations

Useful plots include:

- learned mask probability heatmaps;
- sampled mask heatmaps;
- SME main/auxiliary/combined masks;
- dynamic gate trajectories;
- paired masked/open parameter PCA trajectories.

Heatmaps are qualitative evidence. They should be combined with quantitative
mask statistics and policy performance, not used alone to claim denoising.

## 8. Required evidence chain for the paper

The paper should be organized around:

```text
naive gate behavior
    -> SME expands mask-space exploration
    -> QME improves the decision quality of learned masks
    -> the learned mask improves real policy performance
```

### 8.1 Minimum ablation matrix

| Model | AOM | SME | QME | OVA-TD | Question answered |
|---|---:|---:|---:|---:|---|
| matched hypernetwork baseline | no | no | no | standard TD | Is masking needed? |
| HyperSelect-Gate | yes | no | no | matched design | Does the naive gate collapse? |
| HyperSelect w/o QME | yes | yes | no | yes | Is stochastic exploration alone sufficient? |
| HyperSelect w/o SME | yes | no | yes | yes | Is value evaluation alone sufficient? |
| HyperSelect | yes | yes | yes | yes | Are exploration and evaluation complementary? |

All rows must use matched training horizon, batch size, update ratio, warmup,
evaluation cadence, architecture capacity, and seed set.

### 8.2 Required mask evidence

At minimum report:

- mean keep probability and its distribution over time;
- fraction of nearly-open mask decisions;
- per-state or per-episode mask diversity;
- masked vs open action agreement;
- `action_q_gain_mean` and confidence-stratified gains;
- masked vs forced-open test win rate from the same checkpoint.

The intended claims are:

1. The naive gate often approaches an all-open solution.
2. SME increases exposure to alternative mask configurations.
3. QME makes learned masks more useful for high-value decisions.
4. The full combination improves environment performance.

Each claim must be dropped or weakened if the corresponding evidence fails.

### 8.3 Environments and seeds

Current intended main environments:

- GRF `academy_counterattack_easy` (Counter);
- SMAC `3s5z_vs_3s6z`;
- SMAC `corridor`.

For a strong paper, use at least five seeds for headline comparisons and report
confidence intervals, final performance, and sample-efficiency/AUC statistics.
One seed is suitable for debugging and hypothesis generation only.

## 9. Baseline positioning

### 9.1 Internal matched hypernetwork baselines

The repository contains matched profiles that share the recurrent encoder,
generated two-layer Q head, and QMIX learner while changing only the
hypernetwork condition:

| Profile | Condition |
|---|---|
| `hyper_hypermarl_id` | agent ID |
| `hyper_cash_obs_type` | observation + agent type/capability proxy |
| `hyper_rpg_relation` | relation representation |

These are paper-relevant matched conditioning baselines. Unless they exactly
reproduce every detail of the published HyperMARL/CASH algorithms, call them
"matched ID-conditioned" and "matched observation/capability-conditioned"
baselines rather than claiming they are exact reproductions.

### 9.2 Closely related external work

- **[HyperMARL](https://arxiv.org/abs/2412.04233)**: agent-conditioned hypernetworks for specialization and reduced
  cross-agent gradient interference.
- **[Kaleidoscope](https://arxiv.org/abs/2410.08540)**: learnable parameter masks for adaptive partial parameter sharing and agent heterogeneity. Its masks select shared network parameters and are differentiated across agents, whereas HyperSelect uses observation-conditioned entity masks to control which current observations participate in hypernetwork parameter generation.
- **[CASH](https://arxiv.org/abs/2501.06058)**: capability- and observation-conditioned shared hypernetworks for
  heterogeneous coordination and generalization.
- **[S2RL](https://arxiv.org/abs/2206.11054)**: dense and sparse entity-attention branches in value-based MARL;
  most directly related to learned entity selection.
- **[REFIL](https://proceedings.mlr.press/v139/iqbal21a.html)**: randomized entity-wise factorization and auxiliary value learning;
  related to stochastic entity masking.
- **[Prioritized Tasks Mining (PTM)](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1615.pdf)**: random entity dropping in Attention-QMIX
  for exploration/robustness in multi-task MARL.
- **[Attention-Privileged RL](https://proceedings.mlr.press/v155/salter21a.html) / [asymmetric RL](https://proceedings.mlr.press/v139/warrington21a.html)**: richer-view teacher and
  restricted-view student; related to OVA/QME, though the open branch here is
  full local observation rather than necessarily privileged global state.
- **[Causal State Distillation / Q-Mask](https://proceedings.mlr.press/v236/lu24a/lu24a.pdf)**: learns state masks related to Q values
  for explanation; related to Q-guided masking but differs in objective,
  domain, and MARL/hypernetwork setting.

Safe novelty statement:

> HyperSelect combines observation-conditioned masking, stochastic mask-space
> exploration, and gate-isolated value evaluation to learn which local
> observation entities participate in hypernetwork parameter generation.

Do not claim "the first masking method in MARL."

## 10. Safe and unsafe paper claims

### 10.1 Safe conceptual claims

- HyperSelect learns which observation entities participate in hypernetwork
  parameter generation.
- It treats mask learning as exploration plus task-related evaluation.
- It does not prescribe a final sparsity level.
- QME updates the main observation gate without directly updating the value
  network through the QME loss.
- SME explores additional masked observations and applies augmented TD.
- Dual-gate testing directly compares masked and open execution for one
  checkpoint.

### 10.2 Claims requiring completed experiments

- Naive TD-trained gates *consistently* collapse to all-open across tasks.
- SME prevents or mitigates collapse.
- QME identifies decision-irrelevant information.
- HyperSelect improves sample efficiency and final performance across SMAC and
  GRF.
- HyperSelect outperforms published HyperMARL or CASH implementations.

### 10.3 Claims to avoid

- "The observation always contains noise."
- "A sparser mask is always better."
- "KL80 forces the main mask to be sparse."
- "Q gain proves return improvement."
- "Open-view evaluation must equal a separately trained baseline."
- "HyperSelect is the first MARL masking method."

Preferred vocabulary:

- decision-irrelevant or distracting information;
- task-adaptive observation mask;
- selective conditioning;
- mask-space exploration;
- decision-value evaluation;
- masked-view value calibration.

## 11. Working abstract

> 在合作式多智能体强化学习中，完全参数共享能够提高样本效率、参数效率和可扩展性，但也可能限制不同智能体学习专门化行为。超网络为完全参数共享与独立参数学习提供了一种中间方案：智能体共享同一个参数生成器，同时利用不同的条件信息生成各自的决策网络参数。现有方法逐渐将超网络的条件从智能体身份扩展到智能体能力和动态观测，以增强策略对智能体差异及环境变化的适应能力。然而，当观测直接参与参数生成时，每个观测实体都可能改变所生成的决策函数，使模型容易受到与当前决策无关或具有干扰性的信息影响。在超网络前引入观测门控是一种自然的解决方案，但我们发现，仅依赖时序差分目标训练的朴素门控容易退化为全开状态，无法形成有效的信息选择。
>
> 为解决这一问题，我们提出 HyperSelect，一种面向超网络多智能体强化学习的价值引导自适应观测掩码框架。HyperSelect 将掩码学习建模为掩码空间中的探索与评价问题。首先，自适应观测掩码器根据当前观测动态决定哪些实体信息参与智能体参数的生成。其次，Q 值引导的掩码评价机制利用经过无掩码时序差分学习的分支提供参考动作和决策置信度，并通过梯度隔离的 Q-value 目标评价和优化掩码，使门控学习保留有利于高价值决策的信息，而不改变价值网络本身。最后，随机掩码探索机制通过 KL 约束的随机实体遮蔽和增强时序差分学习，主动探索不同的观测子集，并提高模型在遮蔽观测下进行价值估计的能力。HyperSelect 不预设最优掩码的稀疏程度，而是根据当前状态和决策需求，自适应地选择参与策略参数生成的条件信息。
>
> 在 StarCraft Multi-Agent Challenge 和 Google Research Football 上的实验表明，HyperSelect 相较于强值分解基线和匹配的超网络条件化方案，在样本效率和策略性能方面取得了稳定提升。进一步的消融实验和掩码行为分析表明，价值引导的掩码评价与随机掩码探索能够缓解朴素门控的全开退化，并在学习有效观测选择的过程中发挥互补作用。

The last paragraph is a target result statement. It must be revised to match
the final multi-seed results rather than retained automatically.

## 12. Recommended paper structure

1. **Introduction**
   - Parameter-sharing/specialization trade-off.
   - Hypernetworks and increasingly rich conditioning.
   - Risk of uncontrolled observation conditioning.
   - Naive gate and all-open failure observation.
   - Mask learning as exploration plus evaluation.
2. **Related Work**
   - Hypernetwork MARL: HyperMARL, CASH.
   - Entity selection/sparse attention: S2RL and related attention methods.
   - Random entity masking: REFIL, PTM.
   - Asymmetric/privileged teacher learning and Q-guided masks.
3. **Method**
   - Hypernetwork condition generation.
   - AOM.
   - OVA and QME with gate-only gradients.
   - SME and augmented TD.
   - Full objective, warmup, and decentralized execution.
4. **Experiments**
   - Tasks, baselines, matched compute, seeds.
   - Main performance.
   - Gate/QME/SME ablations.
   - Mask behavior and masked/open paired evaluation.
   - Efficiency and sensitivity.
5. **Limitations**
   - Dependence on learned Q accuracy.
   - Potential masked-input distribution shift.
   - Additional training forward passes.
   - No guarantee that selected information is causally irrelevant.

## 13. Code pointers

- Profile definitions and paper variants:
  `src/modules/agents/counter_transformer_suite.py`
- QME objective and paired diagnostics:
  `src/learners/clean_learner.py`, method
  `_advantage_margin_gate_loss`
- Default coefficients, warmup, and dual-test settings:
  `src/config/algs/clean_hyper.yaml`
- Six-profile objective/TD-path regression:
  `scripts/smoke_test_advantage_objective_six.py`
- Six-profile Counter submission:
  `scripts/ozstar_submit_counter_advantage_objective_six.py`
- Strict no-QME control:
  `scripts/smoke_test_advantage_noadv_control.py` and
  `scripts/ozstar_submit_counter_advantage_noadv_control.py`
- Direct-Q SMAC regression and submission:
  `scripts/smoke_test_advantage_qvalue_smac.py` and
  `scripts/ozstar_submit_advantage_qvalue_smac.py`

## 14. Prompt for the paper-writing task

Use the following prompt after opening this repository in the paper-writing
task:

> Read `paper/material_package/11_hyperselect_paper_handoff.md` as the primary
> source of truth, then consult `src/modules/agents/counter_transformer_suite.py`,
> `src/learners/clean_learner.py`, and `src/config/algs/clean_hyper.yaml` for
> implementation verification. Draft the HyperSelect paper around selective
> conditioning for hypernetwork MARL. Keep AOM as the central method, describe
> SME as mask-space exploration and masked-view TD calibration, and describe
> QME as gate-only decision-value evaluation. Do not equate sparsity with mask
> quality, do not claim KL80 directly sparsifies the main gate, and distinguish
> current single-seed observations from final multi-seed conclusions.
