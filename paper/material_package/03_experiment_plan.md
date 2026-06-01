# Experiment Plan

## Core Experimental Question

The core comparison is not simply dynamic head versus QMIX. The sharper comparison is:

```text
fixed relation-conditioned head vs relation-generated dynamic head
```

This comparison controls for the fact that relation information itself can be useful. If the dynamic version wins, the result is more likely due to changing the decision function rather than merely adding more relation features.

## Primary Models to Compare

Minimum comparison set:

- `qmix_minimal`: clean minimal QMIX baseline.
- `rpg_fixed_linear_structured_maker`: fixed relation-conditioned structured control.
- `local_linear_interaction_hypercond`: ordinary local-observation condition with the same generated one-layer interaction head.
- `rpg_linear_interaction_hypercond`: current main dynamic interaction-head model.

Improvement variants:

- `rpg_residual_interaction_hypercond`: dynamic residual correction.
- `rpg_film_interaction_hypercond`: dynamic feature modulation.
- `rpg_moe_interaction_head`: relation-conditioned soft expert selection.
- `rpg_smooth_linear_interaction_hypercond`: smooth relation-to-head mapping.

## Map Selection Logic

Avoid spending too much budget on maps where both models reach near-perfect win rate quickly. Those maps are useful sanity checks but weak for showing the value of dynamic parameter generation.

Priority maps:

- `corridor`: currently the clearest gap in preliminary results; likely interaction-heavy and useful for the main story.
- `MMM2`: hard enough to test convergence speed, but current results suggest both models may eventually solve it.
- `3s5z_vs_3s6z`: useful if the paper wants an asymmetric combat benchmark and prior literature reports it.
- `5m6m`: useful as a sanity/easy-map result, but not ideal as the main evidence because both fixed and dynamic versions perform well.

Staged budget strategy:

1. Single-seed screening on all candidate variants using `corridor`.
2. Keep the top 2 dynamic variants plus the fixed control.
3. Run those on `MMM2` and `3s5z_vs_3s6z`.
4. Only after seeing a clear effect, run additional seeds on the best map/model pairs.

## Metrics to Report

Use more than final win rate:

- Final test win rate.
- Sample efficiency: environment steps to reach 50%, 80%, and 90% win rate.
- Area under the win-rate curve up to a fixed budget.
- Wall-clock time and environment steps per second.
- Episode length during testing.
- Stability: variance or drop frequency after convergence.
- Auxiliary visualization metrics: relation-head rank correlation, relation/head trajectory clustering, smoothness loss if used.

## Expected Interpretations

If dynamic head beats fixed on `corridor` but not `5m6m`:

This supports the story that dynamic decision functions matter mainly when interaction regimes vary and the decision rule changes over time.

If fixed matches or beats dynamic everywhere:

Then relation extraction/structured maker is useful, but dynamic parameter generation is not yet justified. The next move should be residual/FiLM/smooth variants rather than bigger hypernetworks.

If residual or FiLM beats full generated linear:

The story becomes stronger: relation-conditioned adaptation is useful, but it should be constrained around a stable base decision rule.

If smooth variant improves visualization but not win rate:

It may still be useful as an interpretability regularizer, but the paper should not claim performance improvement from smoothness unless supported by results.

## Result Table Template

Use this structure when organizing results:

```text
Map | Model | Seed | Final win | Steps to 80% | Steps to 90% | AUC | Wall-clock | Notes
```

For paper-level tables, aggregate only after multiple seeds:

```text
Map | Model | Final win mean/std | AUC mean/std | Steps to 90% mean/std
```

## Ablation Questions

- Does relation pattern alone help? Compare `qmix_minimal` vs `rpg_fixed_linear_structured_maker`.
- Does relation pattern help as the hypernetwork input? Compare `local_linear_interaction_hypercond` vs `rpg_linear_interaction_hypercond`.
- Does generated interaction head help beyond fixed relation conditioning? Compare `rpg_fixed_linear_structured_maker` vs `rpg_linear_interaction_hypercond`.
- Does constrained dynamic adaptation help? Compare linear dynamic vs residual/FiLM/MoE.
- Does relation-head smoothness help? Compare linear dynamic vs smooth linear dynamic.
- Is the benefit in interaction actions? Inspect attack Q behavior and battle trace videos.

## Risk Control

Do not run every variant on every map first. That burns budget and creates noisy evidence. Use `corridor` as the screening map, then expand only the promising variants.

The current single-seed figures are enough for direction finding, not enough for final claims.
