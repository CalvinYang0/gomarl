# Visualization Plan

## Visualization Goal

The visualization should answer a concrete question:

Do similar observed relation patterns lead to similar generated MLP head parameters, and do those patterns change meaningfully over a battle?

This matters because the paper's central mechanism is the mapping:

```text
observation -> relation pattern -> generated decision-head parameters -> Q-values
```

The visualization should make this mapping visible.

## Battle Trace Video

Purpose:

Show the actual SMAC episode so the reader can connect model behavior with combat dynamics.

Each frame should show:

- Ally units and enemy units.
- Ally selected actions.
- Attack arrows or labels when an ally attacks an enemy.
- Visible enemy orders or inferred damage arrows where possible.
- Health/shield changes if available.
- Timestep label and basic summary text.

This is not yet a perfect StarCraft replay renderer. It is a diagnostic visualization extracted from environment snapshots and model outputs.

## Relation Pattern 2D Dynamics

Purpose:

Map high-dimensional relation conditions into a 2D space so we can observe whether agents occupy different relation regimes and how those regimes change over time.

Interpretation:

- Each point is one agent at one timestep.
- Point color indicates agent id.
- The trajectory shows how an agent's relation pattern evolves during the episode.
- If agents cluster together, they are experiencing similar relation patterns.
- If an agent moves away from others, its local coordination situation is becoming different.

Current implementation uses PCA over relation conditions collected from the traced episode.

## Generated MLP Head 2D Dynamics

Purpose:

Map generated MLP head parameters into a 2D space to see whether the generated decision functions change over time and differ across agents.

Interpretation:

- Each point is the generated interaction-head parameter vector for one agent at one timestep.
- If head trajectories mirror relation trajectories, this supports the relation-to-decision-function story.
- If all heads collapse to the same point, the hypernetwork may not be using relation information meaningfully.
- If heads vary wildly while relations are smooth, the generated mapping may be unstable.

## Per-Timestep Similarity Matrices

Purpose:

Show agent-agent similarity at each timestep.

Relation similarity matrix:

- Rows and columns are agent ids.
- Color is cosine similarity between relation conditions.
- High similarity means two agents currently have similar relation patterns.

MLP-head similarity matrix:

- Rows and columns are agent ids.
- Color is cosine similarity between generated head parameter vectors.
- High similarity means two agents currently receive similar generated decision functions.

Useful evidence:

If the relation similarity matrix and head similarity matrix have similar block structure, this supports the hypothesis that relation-similar agents receive similar decision heads.

## Static Alignment Plot

Purpose:

Quantify the relation-to-head mapping over all sampled agent-time pairs.

Plot:

```text
x-axis: relation-pattern distance
y-axis: generated-head-parameter distance
```

If the scatter has positive trend and high rank correlation, it suggests nearby relation patterns generate nearby head parameters.

Caution:

This is correlational. It does not prove that the generated parameters cause better actions. It supports interpretability of the learned mapping.

## Possible Paper Figures

Figure 1: Method diagram.

Show observation split, relation pattern extractor, relation-conditioned head generator, self-action branch, interaction-action branch, and QMIX mixer.

Figure 2: Learning curves.

Show fixed versus dynamic variants on `corridor`, `MMM2`, and possibly `3s5z_vs_3s6z`.

Figure 3: Relation-head mapping.

Show relation 2D trajectory and generated-head 2D trajectory from the same episode.

Figure 4: Alignment summary.

Show relation distance versus generated-head distance, plus rank correlation.

Figure 5: Battle trace snapshot/video frame.

Show that relation/head changes correspond to actual combat phase changes.

## What To Look For

Strong evidence:

- Agents in similar local situations are close in relation space.
- Generated head parameters are close when relation patterns are close.
- The relation/head trajectories shift when the battle phase changes.
- The dynamic model's attack choices look more coherent in interaction-heavy moments.

Weak or negative evidence:

- Relation patterns vary, but generated heads barely change.
- Generated heads vary randomly with no relation to relation patterns.
- Fixed and dynamic models behave similarly in all visualizations and learning curves.

