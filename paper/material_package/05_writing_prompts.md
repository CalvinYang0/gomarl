# Writing Prompts

## General Rule for ChatGPT

Always paste the relevant context first, then use a strict prompt like:

```text
Use only the facts in the context below. Do not invent experiments, numbers, citations, or implementation details. If a claim is not supported, mark it as a hypothesis instead of presenting it as a result.
```

## Introduction Draft Prompt

```text
I am writing a MARL paper based on QMIX/CTDE. The central idea is relation-conditioned dynamic decision heads: the model extracts a relation pattern from self/ally/enemy observations and uses it to generate or modulate the local Q-value head.

Write a concise Introduction section with:
1. Background on CTDE value factorization.
2. The limitation of using one fixed local decision function across changing coordination relations.
3. The motivation for relation-conditioned dynamic head generation.
4. A cautious statement that fixed relation-conditioned heads are strong controls.
5. Contributions based only on the provided context.

Do not claim SOTA. Do not invent numbers.
```

## Method Draft Prompt

```text
Write the Method section for the provided model.

Cover:
1. QMIX-style recurrent agent backbone.
2. Observation split into self, ally, and enemy parts.
3. Relation pattern extractor with cross-attention and temporal GRU.
4. Structured decision maker with self-action and interaction-action branches.
5. Difference between fixed relation-conditioned head and generated dynamic head.
6. The four improvement variants: residual, FiLM, MoE, smoothness.
7. CTDE training and decentralized execution boundary.

Keep equations simple. Use precise terminology. Do not overclaim.
```

## Experiment Plan Prompt

```text
Given the provided model and preliminary results, design an experiment section.

Requirements:
1. Explain why fixed relation-conditioned heads are the key control.
2. Prioritize maps where dynamic relational interaction should matter.
3. Include metrics beyond final win rate: AUC, steps-to-threshold, wall-clock, episode length.
4. Propose an efficient staged evaluation strategy because server budget is limited.
5. Separate preliminary single-seed observations from final multi-seed claims.
```

## Result Analysis Prompt

```text
Analyze these learning curves as preliminary evidence.

Please structure the analysis as:
1. What is clearly shown.
2. What is plausible but not yet proven.
3. What additional runs are needed.
4. How this affects the paper story.

Do not overstate single-seed results. If a map is too easy because both methods solve it, say that directly.
```

## Visualization Explanation Prompt

```text
Explain the relation-head visualization for a paper reader.

The visualization includes:
1. A 2D PCA projection of relation patterns over timesteps.
2. A 2D PCA projection of generated MLP head parameters over timesteps.
3. Per-timestep agent-agent similarity matrices for relation patterns and generated heads.
4. A static relation-distance versus head-distance alignment plot.

Explain what each axis, point, color, and matrix means. Also explain what evidence would support the hypothesis that similar relation patterns produce similar generated decision functions.
```

## Reviewer Critique Prompt

```text
Act as a strict reviewer for a MARL conference paper.

Critique the proposed method and experiments. Focus on:
1. Whether the dynamic hypernetwork is justified beyond a fixed relation-conditioned network.
2. Whether the ablations isolate the claimed contribution.
3. Whether the selected SMAC maps are sufficient.
4. Whether the visualization supports the mechanism or is only decorative.
5. Whether claims are too strong for single-seed evidence.

Give concrete revisions, not generic advice.
```

## Polishing Prompt

```text
Polish the following paragraph into academic English.

Constraints:
1. Keep the technical meaning unchanged.
2. Do not add unsupported claims.
3. Prefer clear, direct sentences.
4. Avoid hype words such as "significantly" unless numerical statistical evidence is provided.
```

## Claim Boundary Prompt

```text
Given the current evidence, classify each claim below as:
Supported, Plausible but unproven, or Unsupported.

For each unsupported or unproven claim, rewrite it into a defensible version.
```

