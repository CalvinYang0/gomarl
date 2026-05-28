# AI Writing Workflow

## Division of Labor

Use Codex for source-of-truth extraction:

- Read code and verify implementation details.
- Generate method notes from actual files.
- Produce runnable commands.
- Check whether claims match code and results.
- Create structured context packages for ChatGPT.

Use ChatGPT for drafting and polishing:

- Turn verified notes into paper sections.
- Improve academic English.
- Generate alternative framings.
- Act as a reviewer and find weak claims.
- Help rewrite paragraphs after you decide the technical content.

Do not use ChatGPT as the source of implementation truth. If ChatGPT says something about the model architecture, verify it against code before putting it into the paper.

## Recommended Loop

1. Ask Codex to update the material package after code or experiment changes.
2. Paste `01_chatgpt_context.md` and the relevant method/experiment file into ChatGPT.
3. Ask ChatGPT to draft one section only, not the whole paper.
4. Bring the draft back to Codex for technical consistency checking.
5. Revise manually.
6. Use ChatGPT again for polishing, with the instruction not to add new claims.

## Best First Sections to Draft

Start with these sections:

- Motivation paragraph.
- Method overview.
- Experiment design and ablation logic.
- Visualization explanation.

Do not start with the abstract. The abstract should be written after the main claim and result pattern are stable.

## Safe Claim Workflow

For every claim, classify it as:

- Code fact: directly supported by implementation.
- Preliminary observation: supported by current screenshots or single-seed runs.
- Hypothesis: plausible but not yet experimentally proven.
- Unsupported: should not be written.

Example:

```text
Unsupported: The proposed hypernetwork significantly improves all SMAC tasks.

Defensible: Preliminary results suggest that relation-conditioned dynamic heads can improve convergence on interaction-heavy maps such as corridor, while fixed relation-conditioned heads remain competitive on easier maps.
```

## How To Use ChatGPT Without Burning Codex Quota

Use Codex to produce compact, verified markdown context. Then use ChatGPT for repeated drafting and polishing because drafting consumes many tokens but does not require direct code access every time.

A good ChatGPT input should contain:

- The relevant material package sections.
- The exact task.
- Claim boundaries.
- Desired style.
- A warning not to invent numbers or citations.

## Example ChatGPT Session

Step 1:

Paste `01_chatgpt_context.md`, then ask:

```text
Based only on this context, write a 3-paragraph motivation section. Do not claim SOTA or statistical significance.
```

Step 2:

Paste the draft back into ChatGPT:

```text
Now critique this as Reviewer 2. Identify overclaims, missing controls, and unclear motivation.
```

Step 3:

Bring the revised draft back to Codex:

```text
Check whether this method description matches the current code. Point out inaccurate sentences.
```

## What Not To Do

Do not ask ChatGPT:

```text
Write my full paper based on this idea.
```

That will produce generic text and hallucinated claims.

Do ask:

```text
Using only the provided context, rewrite the following method paragraph to be clearer and more academic. Keep every technical claim unchanged.
```

## Update Triggers

Regenerate or edit the material package when:

- A new model variant is added.
- A result changes the main claim.
- A visualization output changes.
- A run command changes.
- A model name is renamed.
- You decide which map/model pair will become the main result.

