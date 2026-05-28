# Method and Code Inventory

## Important Files

- `src/modules/agents/clean_hyper_agent.py`: core agent implementation, relation capturer, dynamic/fixed head variants, graph variants, diagnostic tensors.
- `src/learners/clean_learner.py`: QMIX learner, AMP support, auxiliary smoothness loss, optional relation-conditioned mixer gate.
- `src/config/algs/clean_hyper.yaml`: CPU/general configuration and model-type comments.
- `src/config/algs/clean_hyper_gpu.yaml`: GPU-oriented config for higher-memory GPUs.
- `src/config/algs/clean_hyper_gpu_v100.yaml`: V100-16GB profile with AMP and safer memory settings.
- `src/utils/battle_trace.py`: battle trace rendering, relation/head similarity, relation/head dynamics videos.
- `src/utils/logging.py`: W&B media upload keys for battle trace outputs.
- `src/runners/episode_runner.py` and `src/runners/parallel_runner.py`: trace collection during test episodes.
- `src/run/run.py`: periodic trace scheduling and saving/uploading.

## Baseline Agent Flow

The minimal QMIX-style local agent follows:

1. Input at timestep `t`: local observation, previous action one-hot, and agent id depending on configuration.
2. `fc1`: maps input features into recurrent hidden dimension.
3. `GRUCell`: updates hidden state from previous hidden state `h_{t-1}` and current encoded input.
4. Fixed two-layer head in `qmix_minimal`: maps `h_t` to per-action Q-values.
5. QMIX mixer: during training, mixes selected per-agent Q-values into a centralized total Q using global state.

The agents are processed together as tensors with shape roughly `[batch, n_agents, feature]`, but the policy is shared across agents.

## Relation Pattern Construction

For RPG-inspired variants, the observation is split according to SMAC observation layout:

- Movement/local self part.
- Enemy features.
- Ally features.
- Own features.

The relation capturer does:

1. Encode self, allies, and enemies separately.
2. Use self query to attend to ally tokens.
3. Use self query to attend to enemy tokens.
4. Concatenate self token, ally context, and enemy context.
5. Pass through an instant-pattern MLP.
6. Concatenate self token and instant pattern.
7. Update temporal relation hidden state with a GRUCell.
8. Pass relation hidden through an output encoder to produce the final relation condition.

Implementation anchor: `RPGInspiredRelationCapturer` in `src/modules/agents/clean_hyper_agent.py`.

## Structured Decision Maker

The structured maker decomposes Q-values:

- `q_ego`: Q-values for non-attack/self actions.
- `q_attack`: Q-values for attack actions, one score per enemy slot.

The final output is:

```text
Q = concat(q_ego, q_attack)
```

For generated variants, the relation condition generates parameters of part of the head. For fixed controls, the relation condition is concatenated as an ordinary input feature but does not generate parameters.

## Current Main Variants

### `qmix_minimal`

Fixed two-layer MLP head after GRU. No hypernetwork. This is the clean minimal QMIX-style baseline.

### `rpg_fixed_linear_structured_maker`

Fixed structured-maker control. It keeps relation pattern extraction and structured self/interaction decomposition, but the interaction scorer is a fixed one-layer linear layer. This is the most important control for testing whether generated parameters add value beyond relation features.

### `rpg_linear_interaction_hypercond`

Main dynamic-head version. The self branch is generated from relation condition. The interaction branch uses a relation-generated one-layer scorer over `[h_i, enemy_token_j]`. This was introduced to reduce the cost of the earlier full interaction hypernetwork.

### `rpg_residual_interaction_hypercond`

A fixed interaction scorer learns a stable base rule. A relation-generated linear scorer provides a residual correction. A learned gate controls the correction strength. This has a clearer story: dynamic generation should not replace the default rule everywhere; it should correct the default when relation structure requires it.

### `rpg_film_interaction_hypercond`

A fixed encoder extracts interaction features from `[h_i, enemy_token_j]`. The relation condition generates FiLM scale and bias to modulate that feature before a fixed final scorer. This tests whether dynamic modulation is more stable than generating the full scorer.

### `rpg_moe_interaction_head`

Uses several fixed interaction experts. Relation condition produces a soft mixture over experts. This tests whether relation patterns behave like soft regimes or modes.

### `rpg_smooth_linear_interaction_hypercond`

Same structure as `rpg_linear_interaction_hypercond`, with a training auxiliary loss: nearby relation conditions should generate nearby interaction-head parameters. This directly supports the visualization/story that similar relation patterns should correspond to similar decision functions.

### `clean_relation_mixer_gate`

Optional training-only gate in the learner. It uses relation condition to reweight per-agent selected Q-values before QMIX mixing. It remains CTDE because the mixer is not used during decentralized execution. This is a possible extension, but the logic needs to be argued carefully.

## Current Visualization Outputs

W&B keys:

- `battle_trace/overview`: static snapshots of the traced battle.
- `battle_trace/video`: per-timestep battle intent video.
- `battle_trace/relation_head_similarity`: static relation/head similarity heatmap.
- `battle_trace/relation_head_alignment`: static summary of relation-distance/head-distance alignment.
- `battle_trace/relation_head_dynamics`: per-timestep video showing relation 2D projection, relation similarity, generated-head 2D projection, and generated-head similarity.

The most useful visualization for the paper is likely `battle_trace/relation_head_dynamics`, because it directly shows whether the relation pattern and generated head move together over time.

