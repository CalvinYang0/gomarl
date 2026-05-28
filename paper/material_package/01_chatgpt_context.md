# ChatGPT Context for Paper Drafting

## Project Summary

The project studies whether a multi-agent reinforcement learning policy can benefit from dynamically changing its local decision function according to the currently observed relation pattern among self, allies, and enemies.

The base framework is QMIX under CTDE. Each agent still has a recurrent local policy network and QMIX still mixes per-agent Q-values during centralized training. The main modification is inside the agent's action-value head: instead of using only a fixed MLP head for all relational situations, the model extracts a relation pattern from the current observation and uses it to condition or generate parts of the decision head.

The key intuition is that the same agent hidden state may require different interaction scoring rules under different local coordination relations. For example, an agent surrounded by allies and facing a low-health enemy should not evaluate attack actions in the same way as an isolated agent facing multiple enemies. A fixed head can represent different outputs, but it must encode all decision rules inside one shared function. A relation-conditioned head instead represents a family of decision functions indexed by the relation pattern.

## Current Method

The model splits each SMAC observation into self-related information, ally-related information, and enemy/other-entity information. It encodes these parts separately, then uses first-person cross-attention from the self token to ally tokens and enemy tokens. The ally context, enemy context, and self token form an instant relation pattern. A GRU updates a temporal relation hidden state, and an output encoder maps this hidden state to a relation condition.

The decision maker decomposes action Q-values into self-action Q-values and interaction-action Q-values.

Self actions include movement, stop, and no-op style actions. Interaction actions are attack actions against visible enemies. The self-action branch is generated from the relation condition. The interaction branch has multiple variants:

- `rpg_linear_interaction_hypercond`: relation condition generates a one-layer linear interaction scorer over `[agent hidden, enemy token]`.
- `rpg_fixed_linear_structured_maker`: fixed control version with a matched one-layer interaction scorer. It uses relation condition as input but does not generate scorer parameters.
- `rpg_residual_interaction_hypercond`: fixed interaction scorer plus a gated relation-generated residual scorer.
- `rpg_film_interaction_hypercond`: fixed interaction encoder, relation-generated FiLM modulation, fixed final scorer.
- `rpg_moe_interaction_head`: fixed expert heads, relation-conditioned soft expert selection.
- `rpg_smooth_linear_interaction_hypercond`: same generated linear interaction scorer as `rpg_linear_interaction_hypercond`, plus a smoothness regularizer encouraging nearby relation patterns to produce nearby generated head parameters.

The fixed control is important because it answers the central question: is dynamic parameter generation useful beyond simply feeding relation information into a fixed network?

## Research Question

The main research question is:

Can relation-conditioned dynamic decision heads make value estimation more sample-efficient or more effective than fixed decision heads in CTDE multi-agent coordination?

More specific questions:

- When does dynamic head generation outperform a fixed relation-conditioned head?
- Is the advantage concentrated in interaction-heavy maps where attack target selection and cooperation structure matter?
- Do similar relation patterns produce similar generated MLP heads?
- Can visualization show a meaningful mapping from observation relation patterns to generated decision functions?

## Working Hypotheses

H1: Dynamic decision heads should help more on maps with changing local interaction regimes, such as corridor-like or asymmetric combat scenarios.

H2: On simple maps, a fixed relation-conditioned head may be enough, so the dynamic head may show little or no improvement.

H3: If the hypernetwork is learning a meaningful relation-to-decision-function mapping, then relation-pattern distance and generated-head-parameter distance should be positively aligned.

H4: A residual or FiLM version may be more stable than fully generated heads because it lets the fixed branch learn a robust default rule while the relation-conditioned component learns situation-specific corrections.

H5: Smoothness regularization may improve generalization and interpretability by making nearby relation patterns map to nearby decision functions.

## Current Empirical Observations

These observations come from current W&B screenshots and should be treated as preliminary single-seed evidence unless confirmed by repeated runs.

On `5m6m`, both `rpg_fixed_linear_structured_maker` and `rpg_linear_interaction_hypercond` reach high win rates quickly. The task may be too easy to create a strong gap. The dynamic version appears slightly stronger or more stable in parts of training, but this is not enough for a strong claim.

On `corridor`, `rpg_linear_interaction_hypercond` appears to outperform `rpg_fixed_linear_structured_maker` clearly in early training and reaches high win rate faster. The fixed run in the available screenshot is much worse and also has longer episode lengths, which may partly explain higher wall-clock cost.

On `MMM2`, both compared versions eventually reach near-perfect performance, but the dynamic linear interaction version appears to converge faster in the available screenshot.

Earlier `3s5z`/`3s5z_vs_3s6z`-related experiments showed that relation-pattern and structured-maker variants can converge much faster than a minimal QMIX baseline, but some full generated-head versions were too expensive.

## Claims That Are Currently Safe

It is safe to say:

- The method introduces relation-conditioned decision heads into a QMIX-style CTDE framework.
- The model explicitly separates relation extraction from decision-head conditioning.
- Preliminary experiments suggest dynamic head generation can accelerate convergence on some maps.
- Matched fixed-head controls are necessary because relation information alone may already improve performance.
- Visualization is used to inspect whether relation patterns and generated head parameters form an aligned mapping.

It is not safe yet to say:

- The method is SOTA.
- Hypernetworks always improve performance.
- The dynamic head universally outperforms fixed heads.
- The method avoids gradient interference completely.
- The relation pattern is causally proven to produce better coordination without ablations and statistical evidence.

## Suggested Paper Angle

A strong paper story should avoid overclaiming "hypernetwork is better." The better angle is:

Fixed relation-conditioned policies still use one shared decision function across all relational regimes. In multi-agent combat, however, the local decision rule itself can change with coordination structure. The proposed method treats relation patterns as selectors or generators of local value-estimation rules, producing a relation-conditioned family of decision heads. The paper then tests when this extra flexibility is useful, when it is unnecessary, and how the learned mapping can be visualized.

