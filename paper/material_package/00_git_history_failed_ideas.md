# Git History Reconstruction of Failed and Abandoned Ideas

This document reconstructs early, deleted, renamed, or superseded research ideas from Git history. It is not a paper draft. It is a memory map for ChatGPT and the human author.

Evidence tags:

- `[repo-confirmed]`: directly supported by Git commits, historical files, current files, configs, or comments.
- `[conversation-derived]`: from prior project discussion visible in the current conversation context.
- `[inferred]`: inferred from commit names, model/config names, code structure, or deletion/refactor patterns.
- `[needs-human-confirmation]`: important but not recoverable from Git alone.

## 1. Method: Git Sources Checked

- `[repo-confirmed]` Inspected recent and full history with `git log --oneline --all`, `git log --all --stat`, and `git log --all --name-status`.
- `[repo-confirmed]` Inspected deletion/refactor commits, especially `936cbf5 Replace legacy group family with clean hyper baselines`, `f54ba8c Isolate graph experiments from original GoMARL`, and `ce46073 Restore ablation-only GoMARL version`.
- `[repo-confirmed]` Read historical files through `git show`, especially `a688d23:src/config/algs/group.yaml`, `a688d23:src/modules/agents/n_group_agent.py`, `a688d23:src/learners/group_learner.py`, and `f54ba8c:src/config/algs/group_graph.yaml`.
- `[repo-confirmed]` Searched current and historical code for keywords including `full_head_variant`, `graph_better_struct`, `graph_input_fusion`, `distill`, `belief`, `pid`, `gat`, `hetero_enemy`, `smooth`, `film`, `moe`, `residual`, and `relation_mixer_gate`.
- `[repo-confirmed]` The largest cleanup point is commit `936cbf5`, which deleted the legacy group stack and introduced the current `clean_hyper` stack.

## 2. High-Level Timeline From Git

### 2.1 Import, W&B, and Early GoMARL Grouping Baseline

- `[repo-confirmed]` `493b89e Add wandb logging support` and `f972b26 Import GoMARL project and add grouping ablations` mark the early project baseline period.
- `[repo-confirmed]` Early code used `group.yaml`, `group_controller.py`, `group_learner.py`, `n_group_agent.py`, `group.py`, and `group_vdn.py`.
- `[repo-confirmed]` The old learner supported `mixer: group` and `mixer: group_vdn`, suggesting both QMIX-like grouped mixing and VDN-style grouped mixing were considered.
- `[inferred]` The original research line was not yet "RPG relation-conditioned MLP heads"; it was closer to dynamic grouping and group-conditioned hypernetwork control.

### 2.2 Graph Grouping Period

- `[repo-confirmed]` Commits `d9d085a`, `ad3aaf9`, `476f79e`, `1f23928`, `d46c1ef`, `fe81849`, `f54ba8c`, `8f921d8`, `879d8f3`, `d09c450`, and `127104b` form a graph-grouping wave.
- `[repo-confirmed]` This period introduced graph-style grouping baselines, group graph visualization, graph-head grouping updates, pseudo-attention graph grouping, local-subgraph graph regrouping, and local-subgraph fusion regrouping.
- `[repo-confirmed]` `f54ba8c` temporarily isolated graph experiments into a separate `group_graph` stack with `group_graph.yaml`, `graph_group_controller.py`, `graph_group_learner.py`, and `graph_group_agent.py`.
- `[repo-confirmed]` `ce46073 Restore ablation-only GoMARL version` deleted that separate graph stack.
- `[inferred]` The graph-grouping line was attractive because graphs can explicitly represent relations between agents, but it was not kept as the core path because it complicated the original runner/learner stack and did not settle the decision-head question directly.

### 2.3 Learned Group Structure and Graph-Structure Variants

- `[repo-confirmed]` Commits `a511de4`, `d6af4ce`, `60f0e7d`, `fa44a26`, `a2bbed4`, `27c5290`, `2023d63`, `d5dd555`, and `cce5688` added learned grouping, graph-better-structure variants, prototype grouping, direct-structure grouping, hybrid graph-structure grouping, and sparsity/visualization improvements.
- `[repo-confirmed]` Historical `group.yaml` lists variants such as `graph_better_struct`, `graph_better_struct_proto`, `graph_better_struct_repr`, `graph_better_struct_slow`, `graph_better_struct_sparse`, `graph_better_struct_hybrid`, `graph_better_struct_row_sparse`, `graph_better_struct_topk_signature`, and `graph_better_struct_ego_subgraph`.
- `[repo-confirmed]` Historical `group_learner.py` contained group balance, confidence, sparse graph, prototype compactness, prototype separation, and threshold-similarity regularizers.
- `[inferred]` This period tested whether learned group assignments and graph-derived structural features could replace manual or fixed coordination assumptions.

### 2.4 Input-Fusion and Node-Embedding Period

- `[repo-confirmed]` Commits `23a31c0`, `d476288`, `dd84c33`, `220dda5`, `428a254`, `c628785`, `a869c40`, `a07ad0d`, and related fixes added input-fusion and node-embedding variants.
- `[repo-confirmed]` Historical `group.yaml` lists `graph_input_fusion`, `graph_input_fusion_node_embed`, `graph_input_fusion_fixed_group`, `graph_input_fusion_group_only`, `graph_input_fusion_head_only`, `graph_input_fusion_hidden_head`, `graph_input_fusion_node_embed_head`, `graph_input_fusion_struct_feat_head`, and `graph_input_fusion_node_embed_struct_feat_head`.
- `[inferred]` These variants tried to answer whether graph/structure information should enter the recurrent hidden state, the action head, the grouping mechanism, or some combination.

### 2.5 Full Dynamic Head and Hypernetwork Scope Search

- `[repo-confirmed]` Commits `f84bfb8 Add full dynamic model generation mode`, `4e3b9f1 Add decoupled no-group hyper head mode`, `47f528f Add subgraph-based no-group full-head mode`, `5b752e4 Add GCN-based no-group full-head mode`, `a160f28 Add faster decoupled residual head mode`, and `8496a78 Add residual-stabilized full-head hypernet mode` show a search over the scope of dynamic parameter generation.
- `[repo-confirmed]` Historical `group.yaml` lists full-head variants such as `dynamic`, `episode_once_head`, `kstep_head`, `ema_step`, `ema_ep_param_mean`, `ema_ep_struct_mean`, `grad_decouple`, `rf`, `hypermarl_rf`, `id_cond`, `gcn`, `neighbor_agg`, `standard_gcn`, `gat`, `temporal_gnn`, `edge_gnn`, `relation_gnn`, and `hetero_enemy`.
- `[repo-confirmed]` Historical `n_group_agent.py` included a full-model generation path that could dynamically generate more than the final action head, including earlier modules.
- `[inferred]` This is the clearest Git evidence that the project explored a large question before the current model: how much of the local decision function should be generated dynamically?

### 2.6 Distillation, PTDE, Belief, PID, and Stabilization Period

- `[repo-confirmed]` Commits `1725e50`, `43815f3`, `857d951`, `bbc916e`, `cfdef06`, `1bdf373`, `8f8ae08`, `b340b0e`, `5523b2f`, and `1060c19` added PTDE strict distillation, teacher TD, head distillation, teacher-only variants, stop-gradient, mixed teacher-only distillation, two-stage schedules, and RF init support for distillation.
- `[repo-confirmed]` Historical `group.yaml` lists `distill`, `ptde_strict`, `distill_q_teacher_td`, `teacher_td_qdistill`, `teacher_td_featdistill`, `teacher_td_multidistill`, `distill_head`, `distill_head_teacher_td`, `belief_cond`, and `pid_dropout`.
- `[repo-confirmed]` Historical `group_learner.py` computed `teacher_td_loss`, `distill_loss`, and `belief_aux_loss`, and logged them.
- `[inferred]` These variants indicate the full-head line likely suffered from optimization, stability, or train/test mismatch concerns, leading to teacher-student and regularized training alternatives.

### 2.7 Clean Hyper Rewrite

- `[repo-confirmed]` `936cbf5 Replace legacy group family with clean hyper baselines` deleted `group.yaml`, `group_controller.py`, `group_learner.py`, `n_group_agent.py`, `group.py`, `group_vdn.py`, `interval_run.py`, `graph_grouping.py`, `group_viz.py`, and `grouping.py`.
- `[repo-confirmed]` The same commit added `clean_hyper.yaml`, `clean_controller.py`, `clean_learner.py`, `clean_hyper_agent.py`, `qmix.py`, and `vdn.py`.
- `[repo-confirmed]` Commit stats show 625 insertions and 4150 deletions.
- `[inferred]` This was not a small refactor; it was a research reset. The codebase moved away from the broad group/full-head search toward a smaller, cleaner family of baselines and dynamic-head variants.

### 2.8 RPG, Structured Relation Pattern, and Interaction-Head Period

- `[repo-confirmed]` Commits `142f2ff`, `7ea9fcc`, `93ea9cf`, `1d6db18`, `821ed39`, `10fee58`, `da3d412`, `a3fdb2f`, `1c341d0`, `c88cfe4`, `5a9df99`, `5da7429`, and `42fc2fb` build the current clean family.
- `[repo-confirmed]` Current `clean_hyper.yaml` and `clean_hyper_agent.py` include `qmix_minimal`, `rpg_relation_hypercond`, `rpg_structured_hypercond`, `rpg_full_structured_hypercond`, `rpg_readout_structured_hypercond`, `rpg_linear_interaction_hypercond`, `rpg_residual_interaction_hypercond`, `rpg_film_interaction_hypercond`, `rpg_moe_interaction_head`, and `rpg_smooth_linear_interaction_hypercond`.
- `[repo-confirmed]` Current `clean_learner.py` includes `clean_relation_mixer_gate`.
- `[conversation-derived]` Prior discussion established that expensive full structured versions could take much longer, motivating one-layer interaction variants and fixed linear controls.

## 3. Discovered Old, Deleted, Renamed, or Superseded Variants

### 3.1 Legacy Group-Agent Family

1. Variant or idea name: `group` / `n_group` legacy GoMARL group-family agent.
2. Where it appeared: `[repo-confirmed]` `src/config/algs/group.yaml`, `src/controllers/group_controller.py`, `src/learners/group_learner.py`, `src/modules/agents/n_group_agent.py`, `src/modules/mixers/group.py`, and `src/modules/mixers/group_vdn.py`, deleted in `936cbf5`.
3. Status: `[repo-confirmed]` Deleted and superseded by `clean_hyper` in `936cbf5`.
4. Motivation: `[inferred]` Explore group-conditioned coordination and group-aware value factorization before narrowing the research question to relation-conditioned decision heads.
5. Hypothesis: `[inferred]` Group structure can improve coordination by allowing agents with similar roles or relations to share or condition decision rules.
6. Model part changed: `[repo-confirmed]` Controller, learner, agent, mixer, group update logic, and runner.
7. Why it may have failed or been abandoned: `[inferred]` It accumulated too many intertwined axes: grouping, graph induction, dynamic heads, distillation, regularization, and custom runner logic. This made ablations hard to interpret.
8. Lesson: `[inferred]` A clean QMIX-based stack was needed to isolate the effect of dynamic local decision heads.
9. How it shaped the final model: `[inferred]` It motivated the later `clean_hyper` reset and the stricter baseline-vs-dynamic-head comparison.
10. Paper placement: `[inferred]` Mention only indirectly as internal exploratory work, not as a main paper contribution.

### 3.2 Separate `graph_group` Stack

1. Variant or idea name: `graph_group` isolated graph-based GoMARL agent.
2. Where it appeared: `[repo-confirmed]` `f54ba8c Isolate graph experiments from original GoMARL` added `group_graph.yaml`, `graph_group_controller.py`, `graph_group_learner.py`, and `graph_group_agent.py`.
3. Status: `[repo-confirmed]` Deleted in `ce46073 Restore ablation-only GoMARL version`.
4. Motivation: `[inferred]` Test graph-based grouping without polluting the original GoMARL stack.
5. Hypothesis: `[inferred]` A learned graph over agents can provide better grouping or coordination context than scalar contribution-based regrouping.
6. Model part changed: `[repo-confirmed]` Added a graph-head agent, graph-group learner, and graph-specific config. `group_graph.yaml` used `group_update_mode: graph`, `graph_head_hidden_dim: 32`, and `graph_edge_threshold: 0.75`.
7. Why it may have failed or been abandoned: `[repo-confirmed]` Deleted soon after introduction. `[inferred]` Likely too invasive, not central enough, or inconsistent with preserving ablation-only GoMARL.
8. Lesson: `[inferred]` Graph modules are promising for relation reasoning, but separate graph-control stacks make experimental interpretation difficult.
9. How it shaped the final model: `[inferred]` Later graph attempts moved into local relation encoders and GAT controls instead of a separate graph runner/learner.
10. Paper placement: `[inferred]` Appendix or omitted unless discussing abandoned graph directions.

### 3.3 Graph-Head Grouping Updates

1. Variant or idea name: graph-head based grouping update.
2. Where it appeared: `[repo-confirmed]` Commits `d9d085a`, `ad3aaf9`, `476f79e`, `1f23928`, `d46c1ef`, and `fe81849`; historical `group_learner.py` and `graph_grouping.py`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`; earlier separate graph stack deleted in `ce46073`.
4. Motivation: `[inferred]` Replace hand-coded regrouping based on contribution scores with graph-induced relationships.
5. Hypothesis: `[inferred]` Pairwise graph edges can identify coordination groups more directly than scalar contribution thresholds.
6. Model part changed: `[repo-confirmed]` Group update mode and group assignment logic. `graph_grouping.py` contained graph-to-groups utilities.
7. Why it may have failed or been abandoned: `[inferred]` Group assignment is a discrete or semi-discrete control layer, which can add instability and obscure whether gains come from grouping or the Q-head.
8. Lesson: `[inferred]` Relation information should perhaps condition the local decision function directly, instead of only changing group membership.
9. How it shaped the final model: `[inferred]` Helped move the project from group-level coordination to relation-conditioned action scoring.
10. Paper placement: `[inferred]` Discussion/background only.

### 3.4 Pseudo-Attention Graph Regrouping

1. Variant or idea name: `graph_pseudo_attn`.
2. Where it appeared: `[repo-confirmed]` `8f921d8 Add pseudo-attention graph grouping mode`, `879d8f3 Fix pseudo-attention graph grouping defaults`, `389b646 Optimize pseudo-attention regroup path`, and historical `group_learner.py`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Use hidden-feature similarity as an attention-like graph for regrouping.
5. Hypothesis: `[inferred]` Agents with similar learned representations should be grouped or coordinated together.
6. Model part changed: `[repo-confirmed]` `group_adjustment_mode` gained `graph_pseudo_attn`; historical `graph_grouping.py` had `pseudo_attention_graph`.
7. Why it may have failed or been abandoned: `[inferred]` Hidden-state similarity may not correspond to useful tactical relation; it can become circular because the representation is already shaped by the current policy.
8. Lesson: `[inferred]` Similarity alone is too weak unless tied to semantically meaningful relation patterns.
9. How it shaped the final model: `[inferred]` Later smoothness visualization asks a sharper question: do semantically constructed relation patterns map continuously to head parameters?
10. Paper placement: `[inferred]` Usually omit, or mention as a failed internal grouping heuristic.

### 3.5 Local-Subgraph Graph Regrouping

1. Variant or idea name: `graph_local_subgraph`.
2. Where it appeared: `[repo-confirmed]` `d09c450 Add local-subgraph graph regroup mode`; historical `group_adjustment_mode: graph_local_subgraph`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Capture each agent's neighborhood relation rather than global hidden similarity.
5. Hypothesis: `[inferred]` Local relation subgraphs better identify tactical groups in partially observable SMAC.
6. Model part changed: `[repo-confirmed]` Group update logic; historical config included `graph_local_neighbor_topk`.
7. Why it may have failed or been abandoned: `[inferred]` Local-subgraph construction can be computationally costly and still only changes group assignment, not the agent's action-value decision rule.
8. Lesson: `[inferred]` Locality matters, but it should be used in a decision-head condition rather than as a separate regrouping process.
9. How it shaped the final model: `[inferred]` Supports the later self/ally/enemy local relation pattern.
10. Paper placement: `[inferred]` Appendix if describing why graph grouping was abandoned.

### 3.6 Local-Subgraph Fusion Regrouping

1. Variant or idea name: `graph_local_fusion`.
2. Where it appeared: `[repo-confirmed]` `127104b Add local-subgraph fusion regroup mode`; historical `group_adjustment_mode: graph_local_fusion`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Fuse local neighborhood evidence into a more robust global grouping graph.
5. Hypothesis: `[inferred]` A fused graph can retain local tactical information while producing global group decisions.
6. Model part changed: `[repo-confirmed]` Group update procedure in `group_learner.py`.
7. Why it may have failed or been abandoned: `[inferred]` Added another layer of graph construction without solving interpretability or head-adaptation questions.
8. Lesson: `[inferred]` More graph processing is not automatically better; the target of adaptation matters.
9. How it shaped the final model: `[inferred]` Reinforced the later preference for compact relation conditions and lightweight interaction heads.
10. Paper placement: `[inferred]` Omit or appendix.

### 3.7 Graph-Better-Struct Family

1. Variant or idea name: `graph_better_struct` family.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml`; commits `d6af4ce`, `60f0e7d`, `fa44a26`, `27c5290`, `2023d63`, `d5dd555`, `cce5688`, and related fixes.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Learn better structural group features from graph-derived information.
5. Hypothesis: `[inferred]` Better structural embeddings improve role/group assignment and therefore Q-value factorization.
6. Model part changed: `[repo-confirmed]` Group-head mode and structural feature construction. Variants included `proto`, `repr`, `slow`, `sparse`, `hybrid`, `row_sparse`, `topk_signature`, and `ego_subgraph`.
7. Why it may have failed or been abandoned: `[inferred]` The many variants suggest no single graph-structure design was clearly dominant or sufficiently clean for a paper story.
8. Lesson: `[inferred]` Graph structure needs a clear semantic target. Pure grouping/structure learning risks becoming an ablation maze.
9. How it shaped the final model: `[inferred]` Motivated semantic decomposition into self/ally/enemy rather than generic graph-structure embeddings.
10. Paper placement: `[inferred]` Not main paper; possibly mentioned as internal exploration.

### 3.8 Prototype Grouping

1. Variant or idea name: `graph_better_struct_proto`.
2. Where it appeared: `[repo-confirmed]` `d6af4ce Split better-struct and prototype grouping modes`; historical `group_learner.py` had prototype compactness and separation losses.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Encourage interpretable role prototypes or group centers.
5. Hypothesis: `[inferred]` Agents can be assigned to learned prototypes, making roles/group relations more stable and interpretable.
6. Model part changed: `[repo-confirmed]` Group assignment and auxiliary losses: `group_proto_compact_alpha`, `group_proto_sep_alpha`.
7. Why it may have failed or been abandoned: `[inferred]` Prototype grouping may impose an artificial discrete role structure and does not directly address target-specific enemy interactions.
8. Lesson: `[inferred]` Interpretability through prototypes is attractive, but fixed role/group slots can be too rigid.
9. How it shaped the final model: `[inferred]` Later relation-head visualization moved toward continuous relation-pattern/head-parameter geometry instead of discrete prototypes.
10. Paper placement: `[inferred]` Appendix or future-work inspiration.

### 3.9 Sparse / Top-k / Threshold Grouping

1. Variant or idea name: sparse graph grouping and threshold group regularization.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml` options `graph_better_struct_sparse`, `graph_better_struct_row_sparse`, `graph_better_struct_topk_signature`, `graph_input_fusion_node_embed_sharp`, and `graph_input_fusion_node_embed_threshold_group`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Make learned relation/group structure sharper, sparser, and more interpretable.
5. Hypothesis: `[inferred]` Sparse group graphs reduce noise and make coordination structure clearer.
6. Model part changed: `[repo-confirmed]` Group graph regularization and assignment. Historical `group_learner.py` computed sparse graph and threshold-similarity losses.
7. Why it may have failed or been abandoned: `[inferred]` Sparsity may improve interpretability but can hurt optimization or lock the model into brittle group relations.
8. Lesson: `[inferred]` Sharp grouping is a poor substitute for smooth observation-conditioned decision adaptation.
9. How it shaped the final model: `[inferred]` Later smoothness regularization chooses continuity over hard grouping.
10. Paper placement: `[inferred]` Omit unless discussing negative evidence.

### 3.10 Input-Fusion Graph Family

1. Variant or idea name: `graph_input_fusion`.
2. Where it appeared: `[repo-confirmed]` `23a31c0 Add graph input fusion grouping mode`, `d476288 Add fixed-group input fusion head mode`, `dd84c33 Fix input fusion fixed-group module init`, `220dda5 Add input fusion node embedding mode`, and historical `group.yaml`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Inject graph-derived information into the agent's decision pipeline.
5. Hypothesis: `[inferred]` The model needs relation information not just for grouping, but also as part of the head input or hidden representation.
6. Model part changed: `[repo-confirmed]` Graph context and node embeddings were fused into head inputs, hidden-head paths, node-embed heads, struct-feature heads, or fixed-group heads.
7. Why it may have failed or been abandoned: `[inferred]` The family had too many attachment points, making causal claims hard: improvement could come from extra capacity, graph context, grouping, or head conditioning.
8. Lesson: `[inferred]` A cleaner intervention is needed: hold the recurrent backbone fixed and alter only a specific decision-head component.
9. How it shaped the final model: `[inferred]` Leads to the current "fixed baseline versus dynamic interaction head" framing.
10. Paper placement: `[inferred]` Mention only as internal design-space exploration.

### 3.11 Node-Embedding Head Variants

1. Variant or idea name: `graph_input_fusion_node_embed*`.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml`; commits `220dda5`, `428a254`, `c628785`, `a869c40`, and related fixes.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Use learned node embeddings as the intermediate relation representation.
5. Hypothesis: `[inferred]` Node embeddings can capture local coordination context more flexibly than explicit group IDs.
6. Model part changed: `[repo-confirmed]` Node embeddings were used for head generation, structure-only graph heads, no-group-embedding ablations, sharp/threshold grouping, and struct-feature full heads.
7. Why it may have failed or been abandoned: `[inferred]` Node embeddings alone are generic; without a clear self/ally/enemy semantic split, they may be hard to interpret and hard to defend.
8. Lesson: `[inferred]` Relation representation should be semantically grounded in observation structure.
9. How it shaped the final model: `[inferred]` Current relation pattern explicitly separates self, ally, and other/enemy information.
10. Paper placement: `[inferred]` Not main text unless describing the route toward semantic relation patterns.

### 3.12 No-Group Head Input Comparison Modes

1. Variant or idea name: no-group input comparison modes.
2. Where it appeared: `[repo-confirmed]` `a07ad0d Add no-group head input comparison modes`; historical `n_group_agent.py` lists no-group head compare modes.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Determine whether gains come from grouping or simply from richer head inputs.
5. Hypothesis: `[inferred]` A dynamic head may not need explicit group assignment if relation/context features are sufficient.
6. Model part changed: `[repo-confirmed]` Removed grouping from certain head variants while preserving graph/struct/head conditioning.
7. Why it may have failed or been abandoned: `[inferred]` The old stack still contained many entangled variables, even in "no-group" modes.
8. Lesson: `[inferred]` This likely helped reveal that dynamic head conditioning, not grouping itself, was the more promising axis.
9. How it shaped the final model: `[inferred]` Current `clean_hyper` models are not primarily group-assignment methods.
10. Paper placement: `[inferred]` Internal lineage, possibly useful for explaining why the paper is not about grouping.

### 3.13 Full Dynamic Model Generation

1. Variant or idea name: full dynamic model generation.
2. Where it appeared: `[repo-confirmed]` `f84bfb8 Add full dynamic model generation mode`; historical `group_head_mode: graph_input_fusion_node_embed_struct_feat_full_model`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Maximize dynamic adaptation by generating a much larger part of the agent network, not only the final head.
5. Hypothesis: `[inferred]` If dynamic heads help, dynamically generating earlier transformation and recurrent components might help more.
6. Model part changed: `[repo-confirmed]` Historical `n_group_agent.py` had full-model modes that generated more than the final action head.
7. Why it may have failed or been abandoned: `[inferred]` Likely too expensive and unstable; it also weakens the paper claim because improvements could come from huge capacity rather than relation-conditioned decision adaptation.
8. Lesson: `[inferred]` Dynamic generation scope must be constrained.
9. How it shaped the final model: `[inferred]` Current main direction focuses on interaction-action heads instead of the full model.
10. Paper placement: `[inferred]` Appendix/discussion as a negative result if supported by experiment logs.

### 3.14 Full Dynamic Head Scope Comparison

1. Variant or idea name: dynamic head scope comparison modes.
2. Where it appeared: `[repo-confirmed]` `21c16e6 Add dynamic head scope comparison modes`; historical `group.yaml` and `n_group_agent.py`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Compare where parameter generation should happen: head-only, full-head, no-group, struct-conditioned, graph-conditioned, or full-model.
5. Hypothesis: `[inferred]` There is an optimal granularity for dynamic parameter generation.
6. Model part changed: `[repo-confirmed]` Parameter generation scope and input condition.
7. Why it may have failed or been abandoned: `[inferred]` Too broad for a clean contribution; the current project needed a more focused story.
8. Lesson: `[inferred]` "Generate everything" is not automatically better; a targeted adaptive component is easier to train and explain.
9. How it shaped the final model: `[inferred]` Supports the final focus on relation-conditioned interaction-action scorer.
10. Paper placement: `[inferred]` Useful in discussion if explaining why the model is deliberately narrow.

### 3.15 Residual-Stabilized Full-Head Hypernetwork

1. Variant or idea name: `graph_input_fusion_node_embed_struct_feat_full_head_residual`.
2. Where it appeared: `[repo-confirmed]` `8496a78 Add residual-stabilized full-head hypernet mode`; historical `dynamic_residual_scale`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`; concept later reappears in cleaner form as `rpg_residual_interaction_hypercond`.
4. Motivation: `[inferred]` Stabilize dynamic parameter generation by adding a dynamic residual to a fixed base head.
5. Hypothesis: `[inferred]` A fixed base decision rule plus relation-conditioned correction is safer than fully generated parameters.
6. Model part changed: `[repo-confirmed]` Full-head Q output became `q_static + dynamic_residual_scale * q_dynamic` in the historical agent.
7. Why it may have failed or been abandoned: `[inferred]` The old version still lived inside a complex graph-input-fusion/full-head stack.
8. Lesson: `[inferred]` Residual dynamic heads are conceptually promising but need a cleaner implementation and ablation target.
9. How it shaped the final model: `[repo-confirmed]` Current `rpg_residual_interaction_hypercond` keeps the residual idea at the interaction-head level.
10. Paper placement: `[inferred]` The current residual variant can be an ablation; the old full-head residual is lineage/background.

### 3.16 Decoupled Dynamic Head

1. Variant or idea name: `graph_input_fusion_node_embed_struct_feat_decoupled_head`.
2. Where it appeared: `[repo-confirmed]` `4e3b9f1 Add decoupled no-group hyper head mode`; `a160f28 Add faster decoupled residual head mode`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Separate the representation used for the bottleneck/action state from the representation used for generated output parameters.
5. Hypothesis: `[inferred]` Decoupling can reduce interference between state encoding and parameter generation.
6. Model part changed: `[repo-confirmed]` Historical `n_group_agent.py` had `_build_graph_input_fusion_no_group_decoupled`.
7. Why it may have failed or been abandoned: `[inferred]` Still costly and not as conceptually clean as self/interaction branch decomposition.
8. Lesson: `[inferred]` Separating pathways is useful, but the separation should align with action semantics.
9. How it shaped the final model: `[inferred]` Current structured maker separates ego-action and interaction-action branches.
10. Paper placement: `[inferred]` Internal lineage only.

### 3.17 GCN Full-Head Variant

1. Variant or idea name: `gcn` / `graph_input_fusion_node_embed_gcn_full_head`.
2. Where it appeared: `[repo-confirmed]` `5b752e4 Add GCN-based no-group full-head mode`; historical `full_head_variant: gcn`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Use graph convolution over agent/entity representations before generating full-head parameters.
5. Hypothesis: `[inferred]` Aggregating neighbor information through GCN improves relation-conditioned head generation.
6. Model part changed: `[repo-confirmed]` Graph encoder before head generation.
7. Why it may have failed or been abandoned: `[inferred]` GCN-style aggregation may blur relation types and adds graph computation overhead.
8. Lesson: `[inferred]` Generic message passing is less targeted than typed self/ally/enemy relation conditioning.
9. How it shaped the final model: `[conversation-derived]` Later discussions considered GAT but recognized graph cost and fixed-node-slot concerns.
10. Paper placement: `[inferred]` Appendix if graph ablation is reported; otherwise omit.

### 3.18 Standard GCN and Neighbor-Aggregation Variants

1. Variant or idea name: `standard_gcn` and `neighbor_agg`.
2. Where it appeared: `[repo-confirmed]` `61e1e4c Add standard GCN full-head variant`; `ceb19aa Add neighbor-agg alias for full-head graph variant`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Compare learned GCN-style propagation with simpler neighbor aggregation.
5. Hypothesis: `[inferred]` Some performance gains may come from basic neighbor averaging rather than a more complex graph encoder.
6. Model part changed: `[repo-confirmed]` Full-head conditioning encoder.
7. Why it may have failed or been abandoned: `[inferred]` If simple neighbor aggregation matches GCN, the graph encoder may be unnecessary; if both underperform, graph conditioning may be the wrong target.
8. Lesson: `[inferred]` Need lightweight, interpretable relation conditions.
9. How it shaped the final model: `[inferred]` Current relation pattern uses attention and compact encoders rather than full graph propagation.
10. Paper placement: `[inferred]` Not central.

### 3.19 GAT Full-Head Variant

1. Variant or idea name: `gat`.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml` lists `full_head_variant: gat`; historical `n_group_agent.py` had `full_head_gat_q`, `full_head_gat_k`, `full_head_gat_v`, and `full_head_gat_encoder`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`; later related GAT controls appear in `1d6db18` and `821ed39`.
4. Motivation: `[inferred]` Use attention over graph neighbors to learn relation-aware conditioning.
5. Hypothesis: `[inferred]` GAT can weight relevant ally/enemy/entity nodes better than uniform GCN aggregation.
6. Model part changed: `[repo-confirmed]` Full-head condition encoder.
7. Why it may have failed or been abandoned: `[conversation-derived]` Earlier graph/GAT versions were computationally expensive and conceptually mismatched with local subgraph expectations. `[inferred]` Historical full-head GAT also remained tied to the complex legacy stack.
8. Lesson: `[inferred]` Attention is useful, but full graph attention may be too broad and expensive.
9. How it shaped the final model: `[repo-confirmed]` Current relation pattern uses cross-attention over ally/enemy tokens rather than a full GAT as the main mechanism.
10. Paper placement: `[inferred]` Graph ablation/appendix if retained; otherwise as abandoned direction.

### 3.20 Temporal GNN Variant

1. Variant or idea name: `temporal_gnn`.
2. Where it appeared: `[repo-confirmed]` `ec4207e Add full-head graph variants: edge, relation, hetero enemy`; historical `full_head_variant: temporal_gnn`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Add temporal relation modeling before head generation.
5. Hypothesis: `[inferred]` Relation structure changes over time, so temporal graph memory could improve dynamic head parameters.
6. Model part changed: `[repo-confirmed]` Graph condition encoder for full-head generation.
7. Why it may have failed or been abandoned: `[inferred]` Adds complexity on top of the recurrent agent hidden state, risking redundancy and instability.
8. Lesson: `[inferred]` Temporal modeling is needed, but a compact relation GRU is cleaner than a full temporal GNN.
9. How it shaped the final model: `[repo-confirmed]` Current RPG relation pattern path includes temporal relation hidden state through a relation GRU.
10. Paper placement: `[inferred]` Internal lineage only.

### 3.21 Edge / Relation / Heterogeneous Enemy GNN Variants

1. Variant or idea name: `edge_gnn`, `relation_gnn`, `hetero_enemy`.
2. Where it appeared: `[repo-confirmed]` `ec4207e Add full-head graph variants: edge, relation, hetero enemy`; historical `group.yaml` lists `full_head_relation_num` and `full_head_hetero_enemy_nodes`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Encode typed relations or heterogeneous enemy nodes, rather than a homogeneous agent graph.
5. Hypothesis: `[inferred]` Ally/enemy relation type matters for head generation and cannot be captured by a plain graph.
6. Model part changed: `[repo-confirmed]` Full-head graph condition encoder and enemy-node construction.
7. Why it may have failed or been abandoned: `[conversation-derived]` Heterogeneous/full graph ideas raised cost and fixed-slot concerns. `[inferred]` The old implementation also likely made train/test and CTDE assumptions harder to defend.
8. Lesson: `[inferred]` Typed relation information is important, but should be handled through environment-semantic decomposition rather than a large fixed heterogeneous graph.
9. How it shaped the final model: `[repo-confirmed]` Current models explicitly split self, ally, and other/enemy information.
10. Paper placement: `[inferred]` Mention as an abandoned graph path only if needed.

### 3.22 Subgraph Full-Head Variant

1. Variant or idea name: `graph_input_fusion_node_embed_subgraph_full_head`.
2. Where it appeared: `[repo-confirmed]` `47f528f Add subgraph-based no-group full-head mode`; historical `n_group_agent.py` had `_build_graph_input_fusion_no_group_subgraph`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Use local subgraph context for each agent's generated head.
5. Hypothesis: `[inferred]` Local relation neighborhoods are more relevant than global graphs.
6. Model part changed: `[repo-confirmed]` Full-head condition features.
7. Why it may have failed or been abandoned: `[conversation-derived]` The later local subgraph discussion found that some graph variants still used large fixed node sets with many uninformative slots. `[inferred]` The historical subgraph version may also have been expensive.
8. Lesson: `[inferred]` Local relation context is right, but fixed graph construction can be the wrong vehicle.
9. How it shaped the final model: `[inferred]` Supports compact self/ally/enemy relation pattern instead of explicit subgraph generation.
10. Paper placement: `[inferred]` Appendix/failure analysis only.

### 3.23 Episode-Once Dynamic Head

1. Variant or idea name: `episode_once_head`.
2. Where it appeared: `[repo-confirmed]` `037071d Add full-head episode-once and k-step update variants`; historical alias map includes `episode_once`, `episode_fixed`, `ep_once_head`, and `ep_fixed_head`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Reduce computation and stabilize dynamic parameters by generating a head once per episode.
5. Hypothesis: `[inferred]` Head parameters do not need to change every timestep to benefit from context conditioning.
6. Model part changed: `[repo-confirmed]` Cached dynamic full-head parameter generation.
7. Why it may have failed or been abandoned: `[inferred]` Episode-level heads may be too coarse for rapidly changing local enemy/ally relations in SMAC.
8. Lesson: `[inferred]` Dynamic adaptation needs a temporal granularity aligned with tactical changes.
9. How it shaped the final model: `[inferred]` Current models condition heads at the timestep/relation-pattern level rather than only episode level.
10. Paper placement: `[inferred]` Appendix if comparing adaptation frequency.

### 3.24 K-Step Dynamic Head

1. Variant or idea name: `kstep_head`.
2. Where it appeared: `[repo-confirmed]` `037071d Add full-head episode-once and k-step update variants`; historical config `full_head_update_interval: 5`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Trade off dynamic adaptation frequency and computational cost.
5. Hypothesis: `[inferred]` Recomputing parameters every k steps may preserve most gains while lowering overhead.
6. Model part changed: `[repo-confirmed]` Cached dynamic full-head generation interval.
7. Why it may have failed or been abandoned: `[inferred]` Adds another hyperparameter and may miss fast local interaction changes.
8. Lesson: `[inferred]` Cost-control matters, but simpler head architecture is cleaner than stale dynamic parameters.
9. How it shaped the final model: `[repo-confirmed]` Current `rpg_linear_interaction_hypercond` reduces generated interaction-head complexity instead of caching full heads.
10. Paper placement: `[inferred]` Omit unless cost-control history is discussed.

### 3.25 EMA Dynamic Head Variants

1. Variant or idea name: `ema_step`, `ema_ep_param_mean`, `ema_ep_struct_mean`.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml` lists EMA variants and `full_head_param_ema_beta`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Smooth dynamic parameter changes to reduce instability.
5. Hypothesis: `[inferred]` Dynamic heads are useful but need temporal smoothing to avoid noisy Q estimates.
6. Model part changed: `[repo-confirmed]` Cached/averaged generated full-head parameters or structure features.
7. Why it may have failed or been abandoned: `[inferred]` EMA can reduce adaptation responsiveness and complicates train/test behavior.
8. Lesson: `[inferred]` Smoothness should be framed in relation-to-head geometry, not only temporal parameter averaging.
9. How it shaped the final model: `[repo-confirmed]` Current `rpg_smooth_linear_interaction_hypercond` regularizes similar relation patterns to have similar generated heads.
10. Paper placement: `[inferred]` Internal predecessor to smoothness variant.

### 3.26 Gradient-Decoupled Full Head

1. Variant or idea name: `grad_decouple`.
2. Where it appeared: `[repo-confirmed]` Historical alias map includes `gradient_separate` and `grad_separate`; historical `full_head_variant: grad_decouple`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Reduce gradient interference between representation learning and hypernetwork/head generation.
5. Hypothesis: `[inferred]` Dynamic head training can destabilize shared features unless gradients are partially separated.
6. Model part changed: `[repo-confirmed]` Full-head generation gradient flow.
7. Why it may have failed or been abandoned: `[inferred]` Gradient tricks address symptoms but do not provide a strong model story by themselves.
8. Lesson: `[conversation-derived]` Later discussions explicitly asked whether hypernetworks can avoid gradient interference; the answer was cautious rather than absolute.
9. How it shaped the final model: `[inferred]` Supports careful claims: dynamic heads may reduce some conflict by conditioning decision rules, but do not automatically remove gradient interference.
10. Paper placement: `[inferred]` Discussion only.

### 3.27 RF / HyperMARL-Style Initialization

1. Variant or idea name: `rf` and `hypermarl_rf`.
2. Where it appeared: `[repo-confirmed]` `4273f48 Add HyperMARL-style full-head init variant`, `1060c19 Allow RF init for distillation variants`, and historical `full_head_rf_fan_mode`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[repo-confirmed]` Commit name explicitly references HyperMARL-style full-head initialization.
5. Hypothesis: `[inferred]` Hypernetwork-generated full heads may require careful initialization to preserve stable output scale.
6. Model part changed: `[repo-confirmed]` Dynamic full-head initialization.
7. Why it may have failed or been abandoned: `[inferred]` Initialization helps optimization but does not solve the core question of what condition should generate which decision parameters.
8. Lesson: `[inferred]` Hypernetwork engineering matters; claims should not ignore initialization and scale.
9. How it shaped the final model: `[inferred]` Current clean variants use more constrained generation to reduce initialization sensitivity.
10. Paper placement: `[inferred]` Related work/implementation note only if using HyperMARL as a technical ancestor.

### 3.28 ID-Conditioned Full Head

1. Variant or idea name: `id_cond`.
2. Where it appeared: `[repo-confirmed]` Historical `group.yaml`; alias map includes `with_id`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Test whether agent identity alone is enough to condition dynamic heads.
5. Hypothesis: `[inferred]` Agent-specific decision rules may help heterogeneous coordination.
6. Model part changed: `[repo-confirmed]` Full-head condition included agent ID.
7. Why it may have failed or been abandoned: `[inferred]` Agent ID is static and cannot capture changing local ally/enemy relations.
8. Lesson: `[inferred]` The condition should be relation/context-sensitive, not merely identity-sensitive.
9. How it shaped the final model: `[repo-confirmed]` Current relation-conditioned models use relation patterns as the dynamic condition.
10. Paper placement: `[inferred]` Useful baseline idea but not central unless implemented in current experiments.

### 3.29 Tri-Branch Full-Head Conditioning

1. Variant or idea name: `tri_branch`.
2. Where it appeared: `[repo-confirmed]` `857d951 Add tri-branch full-head conditioning variant`; alias map includes `three_branch`, `three_branch_mlp`, `triple_branch`, and `tri_mlp`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Split conditioning information into multiple branches before generating full-head parameters.
5. Hypothesis: `[inferred]` Different information sources should be encoded separately before conditioning the head.
6. Model part changed: `[repo-confirmed]` Full-head condition encoder.
7. Why it may have failed or been abandoned: `[inferred]` Branching by itself lacks a clear semantic/action decomposition.
8. Lesson: `[inferred]` Multi-branch structure is useful when branches have a defensible meaning.
9. How it shaped the final model: `[repo-confirmed]` Current method uses self/ally/enemy relation processing and ego/interaction action branches.
10. Paper placement: `[inferred]` Internal predecessor to structured maker.

### 3.30 No-Struct Full-Head Ablation

1. Variant or idea name: `no_struct`.
2. Where it appeared: `[repo-confirmed]` `bbc916e Add no-struct full-head ablation`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Test whether explicit structure features matter or whether a learned node embedding/head condition is sufficient.
5. Hypothesis: `[inferred]` If no-struct performs well, graph/structure decomposition may be unnecessary.
6. Model part changed: `[repo-confirmed]` Full-head condition input.
7. Why it may have failed or been abandoned: `[needs-human-confirmation]` Git confirms the ablation existed but not its result.
8. Lesson: `[inferred]` Necessary ablation logic: separate the value of dynamic generation from the value of structural relation condition.
9. How it shaped the final model: `[repo-confirmed]` Current experiments include fixed and dynamic structured controls.
10. Paper placement: `[inferred]` A similar ablation should appear if current experiments support it.

### 3.31 PTDE Strict Distillation

1. Variant or idea name: `ptde_strict`.
2. Where it appeared: `[repo-confirmed]` `1725e50 Add PTDE strict distill, belief-conditioned head, and PID dropout variants`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Handle train/test information mismatch or stabilize dynamic head learning with teacher-student training.
5. Hypothesis: `[inferred]` A privileged/training-time teacher can guide an execution-compatible student.
6. Model part changed: `[repo-confirmed]` Distillation pathway in `n_group_agent.py` and `group_learner.py`.
7. Why it may have failed or been abandoned: `[inferred]` PTDE-style distillation is a different paper story and could distract from relation-conditioned decision heads.
8. Lesson: `[inferred]` Avoid relying on complex teacher mechanisms unless the paper is about distillation.
9. How it shaped the final model: `[inferred]` Current clean CTDE setup keeps the learner conventional.
10. Paper placement: `[inferred]` Omit or future work.

### 3.32 Q-Value Teacher Distillation

1. Variant or idea name: `distill`, `distill_q_teacher_td`, `teacher_td_qdistill`.
2. Where it appeared: `[repo-confirmed]` `43815f3`, `cfdef06`, `bbbc62d`, `8f8ae08`, `b340b0e`, and historical `group_learner.py`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Train a smaller or execution-safe student head with supervision from a richer teacher head.
5. Hypothesis: `[inferred]` Distillation can transfer benefits of expressive dynamic heads while reducing deployment complexity or instability.
6. Model part changed: `[repo-confirmed]` Learner computed teacher and student Qs, teacher TD loss, and Q distillation loss.
7. Why it may have failed or been abandoned: `[inferred]` Adds training complexity and makes attribution of improvements difficult.
8. Lesson: `[inferred]` If dynamic heads are useful, the direct version should first be demonstrated without distillation.
9. How it shaped the final model: `[repo-confirmed]` Current `clean_learner.py` uses ordinary TD loss plus optional lightweight auxiliary losses, not teacher-student distillation.
10. Paper placement: `[inferred]` Appendix/future work only.

### 3.33 Feature and Multi-Distillation Variants

1. Variant or idea name: `teacher_td_featdistill`, `teacher_td_multidistill`.
2. Where it appeared: `[repo-confirmed]` `8f8ae08 Add mixed teacher-only distillation variant`; historical `group_learner.py` had feature distillation paths.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Distill not only Q outputs but also intermediate features.
5. Hypothesis: `[inferred]` Feature-level supervision may better transfer relation/head representation.
6. Model part changed: `[repo-confirmed]` Student and teacher feature encoders and distillation losses.
7. Why it may have failed or been abandoned: `[inferred]` Feature matching introduces extra design choices and may constrain the student too much.
8. Lesson: `[inferred]` Strong auxiliary training can obscure the core effect of dynamic head conditioning.
9. How it shaped the final model: `[inferred]` Current visualization/regularization focuses on relation-head geometry rather than teacher feature imitation.
10. Paper placement: `[inferred]` Omit unless human author has strong results.

### 3.34 Head-Parameter Distillation

1. Variant or idea name: `distill_head`, `distill_head_teacher_td`.
2. Where it appeared: `[repo-confirmed]` `43815f3 Add teacher-TD and head-distill full-head variants`; historical `group_learner.py` matched `teacher_head_params` and `student_head_params`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Directly distill generated parameter tensors, not only Q outputs.
5. Hypothesis: `[inferred]` The generated head parameters themselves contain transferable relation-conditioned decision logic.
6. Model part changed: `[repo-confirmed]` Head-parameter packing and distillation losses.
7. Why it may have failed or been abandoned: `[inferred]` Matching raw parameters can be ill-conditioned because equivalent functions may have different parameterizations.
8. Lesson: `[inferred]` Parameter geometry is interesting for analysis, but training by parameter matching may be brittle.
9. How it shaped the final model: `[conversation-derived]` Later visualization idea focuses on relation-pattern similarity versus generated-head similarity, not necessarily parameter-distillation training.
10. Paper placement: `[inferred]` Analysis lineage, not main method.

### 3.35 Automatic Two-Stage Distillation

1. Variant or idea name: two-stage teacher/student schedule.
2. Where it appeared: `[repo-confirmed]` `b340b0e Add two-stage full-head distillation flow`; `5523b2f Add automatic two-stage distillation schedule`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Train teacher first, then switch to student.
5. Hypothesis: `[inferred]` Separating teacher and student stages prevents unstable joint optimization.
6. Model part changed: `[repo-confirmed]` Historical config included `full_head_train_stage` and `full_head_two_stage_teacher_tmax`.
7. Why it may have failed or been abandoned: `[inferred]` Doubles experiment planning complexity and is expensive for SMAC.
8. Lesson: `[inferred]` Cost and interpretability matter; avoid methods that require fragile schedules unless absolutely necessary.
9. How it shaped the final model: `[inferred]` Current experiments favor direct comparisons under the same QMIX learner.
10. Paper placement: `[inferred]` Omit.

### 3.36 Belief-Conditioned Head

1. Variant or idea name: `belief_cond`.
2. Where it appeared: `[repo-confirmed]` `1725e50 Add PTDE strict distill, belief-conditioned head, and PID dropout variants`; historical `n_group_agent.py` had `belief_stat_encoder`, `belief_head_encoder`, and `belief_aux_loss`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Add latent belief/context variables to the dynamic head condition.
5. Hypothesis: `[inferred]` A latent belief can capture hidden or partially observable tactical state that raw relation features miss.
6. Model part changed: `[repo-confirmed]` Variational latent with KL regularization controlled by `full_head_belief_kl_alpha`.
7. Why it may have failed or been abandoned: `[inferred]` This shifts the story toward latent-state inference rather than relation-conditioned decision functions.
8. Lesson: `[inferred]` More expressive hidden context is tempting but needs strong justification.
9. How it shaped the final model: `[inferred]` Current relation pattern stays observable and interpretable.
10. Paper placement: `[inferred]` Future work, not main paper.

### 3.37 PID Dropout

1. Variant or idea name: `pid_dropout`.
2. Where it appeared: `[repo-confirmed]` `1725e50`; historical `group.yaml` included `full_head_pid_start`, `full_head_pid_end`, and `full_head_pid_anneal_steps`.
3. Status: `[repo-confirmed]` Deleted in `936cbf5`.
4. Motivation: `[inferred]` Regularize teacher/student or dynamic-head features through scheduled dropout.
5. Hypothesis: `[inferred]` Dropout forces robust student behavior or prevents over-reliance on privileged/dynamic features.
6. Model part changed: `[repo-confirmed]` Feature dropout in the full-head/distillation path.
7. Why it may have failed or been abandoned: `[inferred]` Regularization did not address the central novelty and added tuning burden.
8. Lesson: `[inferred]` Avoid regularization-heavy methods unless they clarify the main mechanism.
9. How it shaped the final model: `[inferred]` Current smoothness loss is more directly tied to the relation-to-head hypothesis.
10. Paper placement: `[inferred]` Omit.

### 3.38 Relation Mixer Gate

1. Variant or idea name: relation-conditioned mixer gate.
2. Where it appeared: `[repo-confirmed]` Current `clean_learner.py` includes `clean_relation_mixer_gate`; current `clean_hyper.yaml` includes `clean_relation_mixer_gate: False`.
3. Status: `[repo-confirmed]` Still present but disabled by default.
4. Motivation: `[conversation-derived]` The idea was to use relation patterns to adjust Q mixing weights or group influence.
5. Hypothesis: `[inferred]` Relation context can improve credit assignment or global value mixing, not only local action scoring.
6. Model part changed: `[repo-confirmed]` Learner-side post-agent Q modulation before mixing.
7. Why it may have failed or been abandoned: `[conversation-derived]` The user questioned the logic: why should relation condition alter the mixer rather than the local decision rule? `[inferred]` It is less conceptually central than adaptive local interaction heads.
8. Lesson: `[inferred]` Relation conditioning has a stronger story when tied to interaction-action scoring than to global value mixing.
9. How it shaped the final model: `[inferred]` It clarifies the paper's focus: local decision function adaptation, not arbitrary relation-conditioned training-side gating.
10. Paper placement: `[inferred]` Appendix or not at all unless experiments are strong.

### 3.39 Auxiliary Relation / Smoothness Loss

1. Variant or idea name: smoothness regularization / relation-to-head continuity.
2. Where it appeared: `[repo-confirmed]` Current `rpg_smooth_linear_interaction_hypercond` in `clean_hyper_agent.py`; current config options `clean_smooth_head_loss_coef`, `clean_smooth_head_knn`, and `clean_smooth_head_sample_size`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[conversation-derived]` The user proposed that agents with similar relation patterns should have similar MLP head parameters.
5. Hypothesis: `[repo-confirmed]` Current code implements a KNN smoothness regularizer for generated head parameters based on relation-condition proximity.
6. Model part changed: `[repo-confirmed]` Auxiliary loss added through `latest_aux_loss` and consumed by `clean_learner.py`.
7. Why it may have failed or been abandoned: `[needs-human-confirmation]` Git confirms implementation but not final experimental outcome.
8. Lesson: `[inferred]` Smoothness provides both a training regularizer and a visualization story: relation-pattern geometry should align with head-parameter geometry.
9. How it shaped the final model: `[inferred]` Strengthens the core claim that the hypernetwork learns a mapping from relation patterns to decision functions.
10. Paper placement: `[inferred]` Main ablation if results are good; otherwise visualization/analysis appendix.

### 3.40 RPG Relation Route Variant

1. Variant or idea name: `rpg_relation_route`.
2. Where it appeared: `[repo-confirmed]` `142f2ff Add RPG-inspired relation condition and route variants`; current `clean_hyper_agent.py` includes `rpg_relation_route`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[inferred]` Test routing as an alternative to continuous hypernetwork parameter generation.
5. Hypothesis: `[inferred]` Relation patterns may select or route among decision pathways instead of generating weights.
6. Model part changed: `[repo-confirmed]` Action head/condition path in `clean_hyper_agent.py`.
7. Why it may have failed or been abandoned: `[needs-human-confirmation]` Git does not show experimental outcome.
8. Lesson: `[inferred]` Routing is a natural control for dynamic heads: choose among fixed routes versus generate parameters.
9. How it shaped the final model: `[repo-confirmed]` Later `rpg_moe_interaction_head` is a cleaner expert/routing-style variant for interaction heads.
10. Paper placement: `[inferred]` Ablation or appendix.

### 3.41 Dynamic Route / Codebook Variant

1. Variant or idea name: `dynamic_route`.
2. Where it appeared: `[repo-confirmed]` Current `clean_hyper.yaml` comments describe `dynamic_route` as "local route/codebook selects the head condition"; current `clean_hyper_agent.py` has `route_logits_head`, `route_codebook`, `clean_route_num`, and `clean_route_temperature`.
3. Status: `[repo-confirmed]` Still present as a clean-hyper control, but not the current main model.
4. Motivation: `[inferred]` Replace continuous hypernetwork conditioning with discrete or semi-discrete route selection over learned condition embeddings.
5. Hypothesis: `[inferred]` A small route/codebook can represent recurring local decision regimes more stably than freely generated continuous head conditions.
6. Model part changed: `[repo-confirmed]` The condition used for head generation is selected through Gumbel-softmax route logits and a learned route codebook.
7. Why it may have failed or been abandoned: `[needs-human-confirmation]` Git confirms implementation but not outcome. `[inferred]` Route/codebook methods can be less expressive than continuous relation-conditioned generation and may introduce route-collapse or temperature sensitivity.
8. Lesson: `[inferred]` Routing is a useful control for the core claim: if route/codebook works, the benefit may come from selecting among relation regimes; if continuous heads work better, the benefit is finer-grained adaptation.
9. How it shaped the final model: `[inferred]` It provides a bridge between fixed heads, MoE heads, and fully generated heads.
10. Paper placement: `[inferred]` Appendix or ablation if results exist.

### 3.42 Graph Route / Codebook Variant

1. Variant or idea name: `graph_route`.
2. Where it appeared: `[repo-confirmed]` Current `clean_hyper.yaml` comments describe `graph_route` as "obs-only graph + standard GCN -> route/codebook"; current `clean_hyper_agent.py` includes `graph_route`.
3. Status: `[repo-confirmed]` Still present as a graph-route control, but not the current main line.
4. Motivation: `[inferred]` Combine graph/entity relation encoding with route/codebook selection.
5. Hypothesis: `[inferred]` A graph encoder can identify relation regimes, and a route codebook can map those regimes to stable head-generation conditions.
6. Model part changed: `[repo-confirmed]` Graph-derived condition feeds route logits, then selects route-codebook embeddings.
7. Why it may have failed or been abandoned: `[conversation-derived]` Graph routes inherit the cost and fixed-slot concerns of graph variants. `[inferred]` It also mixes two mechanisms, graph encoding and routing, making attribution harder.
8. Lesson: `[inferred]` Route/codebook ideas are cleaner when applied to relation patterns directly than when stacked on graph construction.
9. How it shaped the final model: `[inferred]` Helped separate three alternatives: graph-conditioned route, relation-conditioned route, and relation-conditioned generated interaction head.
10. Paper placement: `[inferred]` Appendix graph/routing ablation only.

### 3.43 RPG Structured Hypercondition Maker

1. Variant or idea name: `rpg_structured_hypercond`.
2. Where it appeared: `[repo-confirmed]` `7ea9fcc Add RPG-inspired structured hypercondition maker`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present but later variants refine it.
4. Motivation: `[conversation-derived]` Incorporate RPG-style relation pattern generation and structured decision-maker decomposition.
5. Hypothesis: `[inferred]` Relation pattern should condition the decision head, and Q-values should be decomposed into ego and interaction components.
6. Model part changed: `[repo-confirmed]` Relation capturer, relation hidden state, structured head generation.
7. Why it may have failed or been modified: `[conversation-derived]` Full structured/hyper versions were computationally expensive; some branches generated too many parameters.
8. Lesson: `[inferred]` Structured relation conditioning is promising, but the generated parameter scope must be controlled.
9. How it shaped the final model: `[repo-confirmed]` Forms the conceptual base for linear interaction, residual, FiLM, MoE, and smooth variants.
10. Paper placement: `[inferred]` Main ancestor/baseline.

### 3.44 Local Structured Maker Control

1. Variant or idea name: `local_structured_hypercond`.
2. Where it appeared: `[repo-confirmed]` Commit `93ea9cf Add local structured maker control`; earlier user error showed `Unknown clean_model_type=local_structured_hypercond`, meaning it was later renamed/removed from current valid model list.
3. Status: `[repo-confirmed]` Removed or renamed; not in current `MODEL_SPECS`.
4. Motivation: `[conversation-derived]` Test local subgraph/observation-based structured generation.
5. Hypothesis: `[inferred]` A local-only condition may avoid global graph overhead and fixed-slot issues.
6. Model part changed: `[repo-confirmed]` Historical clean-hyper model type around structured maker.
7. Why it may have failed or been abandoned: `[conversation-derived]` The implemented local/global graph semantics were confusing; large graph/fixed-slot concerns led away from this path.
8. Lesson: `[inferred]` Model naming and execution-scope clarity matter.
9. How it shaped the final model: `[inferred]` Later configs distinguish CTDE/CTCE/GAT controls and current relation-conditioned variants.
10. Paper placement: `[inferred]` Do not mention unless explaining abandoned graph/local variants.

### 3.45 Ego GAT Relation Controls

1. Variant or idea name: `ego_gat` / ego GAT relation controls.
2. Where it appeared: `[repo-confirmed]` `1d6db18 Add ego GAT relation controls`; current `clean_hyper_agent.py` includes GAT-related RPG relation options.
3. Status: `[repo-confirmed]` Present as graph control variants, not the main current recommendation.
4. Motivation: `[conversation-derived]` The user asked whether graph/GAT could replace attention-based relation pattern construction.
5. Hypothesis: `[inferred]` A GAT over local entities can build a stronger relation condition than cross-attention over split ally/enemy tokens.
6. Model part changed: `[repo-confirmed]` Relation condition builder.
7. Why it may have failed or been abandoned: `[conversation-derived]` Graph versions had high runtime cost and fixed-slot concerns.
8. Lesson: `[inferred]` GAT is a valid ablation but weak as the main contribution unless it clearly beats simpler relation attention.
9. How it shaped the final model: `[repo-confirmed]` Main line returned to compact relation pattern and interaction-head generation.
10. Paper placement: `[inferred]` Appendix graph-control ablation if results exist.

### 3.46 Global CTCE GAT Graph Controls

1. Variant or idea name: global CTCE GAT graph controls.
2. Where it appeared: `[repo-confirmed]` `821ed39 Add global CTCE GAT graph controls`.
3. Status: `[repo-confirmed]` Present as control variants, not current main line.
4. Motivation: `[conversation-derived]` The user proposed CTCE-style centralized graph computation to avoid per-agent graph recomputation.
5. Hypothesis: `[conversation-derived]` Compute one ally graph and one enemy/other graph, or one heterogeneous graph, centrally to reduce repeated local cost.
6. Model part changed: `[repo-confirmed]` Graph relation condition path in clean agent.
7. Why it may have failed or been abandoned: `[conversation-derived]` CTCE/global graph assumptions were not aligned with the main CTDE execution story, and graph cost/slot issues remained.
8. Lesson: `[inferred]` Efficiency can be improved by centralizing graph computation, but the paper story becomes about centralized graph reasoning rather than local adaptive heads.
9. How it shaped the final model: `[inferred]` Reinforced the need to keep execution assumptions clear.
10. Paper placement: `[inferred]` Appendix only.

### 3.47 Full RPG Structured Hyper Maker

1. Variant or idea name: `rpg_full_structured_hypercond`.
2. Where it appeared: `[repo-confirmed]` `da3d412 Add full RPG structured hyper maker`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present but later reduced by `1c341d0 Reduce full RPG interaction hyper maker cost` and followed by lighter variants.
4. Motivation: `[conversation-derived]` Correct the structured maker so both ego and interaction scoring were more fully generated/conditioned.
5. Hypothesis: `[inferred]` Fully dynamic structured heads should improve relation-sensitive decision making.
6. Model part changed: `[repo-confirmed]` Generated structured action head.
7. Why it may have failed or been modified: `[conversation-derived]` Training time reportedly became around two days versus about 16 hours for a previous version; interaction branch parameter generation was too expensive.
8. Lesson: `[inferred]` More generated parameters can hurt practicality without guaranteeing better performance.
9. How it shaped the final model: `[repo-confirmed]` Led directly to readout and linear interaction variants.
10. Paper placement: `[inferred]` Ablation showing why the final model is lightweight.

### 3.48 Readout Structured Hyper Maker

1. Variant or idea name: `rpg_readout_structured_hypercond`.
2. Where it appeared: `[repo-confirmed]` `c88cfe4 Add RPG readout structured hyper maker`; current config comments.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[conversation-derived]` Reduce interaction-branch cost by changing what the generated readout/head does.
5. Hypothesis: `[inferred]` A lighter readout-level generated component may preserve benefits while lowering runtime.
6. Model part changed: `[repo-confirmed]` Structured maker readout branch.
7. Why it may have failed or been modified: `[needs-human-confirmation]` Git confirms implementation, but final outcome needs run logs.
8. Lesson: `[inferred]` Cost-aware dynamic head design matters.
9. How it shaped the final model: `[repo-confirmed]` Precedes `rpg_linear_interaction_hypercond`.
10. Paper placement: `[inferred]` Ablation if results are informative.

### 3.49 Linear Interaction Hypercondition Variant

1. Variant or idea name: `rpg_linear_interaction_hypercond`.
2. Where it appeared: `[repo-confirmed]` `5a9df99 Add linear interaction RPG structured variant`; current `clean_hyper_agent.py` and config comments.
3. Status: `[repo-confirmed]` Still present and appears central in recent experiments.
4. Motivation: `[conversation-derived]` User requested "interaction branch only use one linear layer" to reduce cost and keep fixed/dynamic comparison fair.
5. Hypothesis: `[inferred]` The interaction-action branch is the right place for dynamic adaptation, but generating a single linear interaction scorer may be enough.
6. Model part changed: `[repo-confirmed]` Interaction branch head complexity.
7. Why it may have failed or been modified: `[needs-human-confirmation]` Current results vary by map; some plots showed strong performance on corridor/MMM2 and close performance on easier maps.
8. Lesson: `[inferred]` Lightweight dynamic heads are the most defensible tradeoff between performance, cost, and story.
9. How it shaped the final model: `[repo-confirmed]` It is the anchor for residual, FiLM, MoE, and smooth variants.
10. Paper placement: `[inferred]` Main candidate method if experiments support it.

### 3.50 Fixed Linear Structured Maker

1. Variant or idea name: `rpg_fixed_linear_structured_maker`.
2. Where it appeared: `[repo-confirmed]` `5da7429 Add fixed linear RPG structured control`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[conversation-derived]` Establish a fair fixed-parameter control where the interaction branch is also one linear layer.
5. Hypothesis: `[inferred]` If dynamic heads help beyond capacity/structure, they should beat this fixed structured baseline on relation-sensitive maps.
6. Model part changed: `[repo-confirmed]` Same structured decomposition but fixed interaction scorer.
7. Why it may have failed or complicated the story: `[conversation-derived]` User observed fixed versions can perform well, raising the question of whether hypernetworks add value.
8. Lesson: `[inferred]` The paper must argue dynamic adaptation under changing relation patterns, not just additional structure.
9. How it shaped the final model: `[repo-confirmed]` It is the critical ablation baseline.
10. Paper placement: `[inferred]` Main ablation, essential for fairness.

### 3.51 Residual Interaction Hypercondition Variant

1. Variant or idea name: `rpg_residual_interaction_hypercond`.
2. Where it appeared: `[repo-confirmed]` `42fc2fb Add relation-conditioned head variants`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[conversation-derived]` Explore residual dynamic correction rather than full dynamic replacement.
5. Hypothesis: `[inferred]` A fixed base interaction rule plus relation-conditioned residual is more stable and easier to train.
6. Model part changed: `[repo-confirmed]` Interaction action branch uses gated residual dynamic output; config has `clean_rpg_residual_gate_bias`.
7. Why it may fail: `[needs-human-confirmation]` Outcome not recoverable from Git.
8. Lesson: `[inferred]` Residualization directly answers the concern that dynamic heads may over-adapt or erase useful fixed rules.
9. How it shaped the final model: `[inferred]` Provides a theoretically cleaner dynamic-head variant.
10. Paper placement: `[inferred]` Main ablation or appendix depending on results.

### 3.52 FiLM Interaction Hypercondition Variant

1. Variant or idea name: `rpg_film_interaction_hypercond`.
2. Where it appeared: `[repo-confirmed]` `42fc2fb`; current `clean_hyper_agent.py` includes `rpg_interaction_film_gamma` and `rpg_interaction_film_beta`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[inferred]` Use feature-wise modulation as a safer alternative to generating full weights.
5. Hypothesis: `[inferred]` Relation pattern can modulate interaction features through scale/shift while preserving a fixed head.
6. Model part changed: `[repo-confirmed]` Interaction hidden features are modulated by relation-conditioned gamma/beta.
7. Why it may fail: `[needs-human-confirmation]` Outcome not recoverable from Git.
8. Lesson: `[inferred]` Separates "dynamic decision function" into modulation versus parameter generation.
9. How it shaped the final model: `[inferred]` Helps locate the source of improvement: weight generation or conditional feature modulation.
10. Paper placement: `[inferred]` Ablation.

### 3.53 MoE Interaction Head

1. Variant or idea name: `rpg_moe_interaction_head`.
2. Where it appeared: `[repo-confirmed]` `42fc2fb`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[inferred]` Use relation-conditioned expert selection/routing instead of free-form generated weights.
5. Hypothesis: `[inferred]` A small set of fixed interaction experts can represent different relation regimes more stably and interpretably.
6. Model part changed: `[repo-confirmed]` Interaction head becomes mixture/routing over expert heads.
7. Why it may fail: `[needs-human-confirmation]` Outcome not recoverable from Git.
8. Lesson: `[inferred]` MoE is a middle ground between fixed head and fully generated head.
9. How it shaped the final model: `[inferred]` Helps answer whether continuous dynamic generation is necessary.
10. Paper placement: `[inferred]` Ablation or appendix.

### 3.54 QMix Minimal Fixed MLP Baseline

1. Variant or idea name: `qmix_minimal`.
2. Where it appeared: `[repo-confirmed]` `10fee58 Make qmix minimal baseline use fixed MLP head`; current `clean_hyper_agent.py`.
3. Status: `[repo-confirmed]` Still present.
4. Motivation: `[conversation-derived]` User identified that the original baseline was unfair if it lacked an equivalent MLP layer.
5. Hypothesis: `[inferred]` A fair baseline should have a comparable fixed two-layer head before claiming dynamic generation helps.
6. Model part changed: `[repo-confirmed]` Baseline model head.
7. Why it may have failed or been updated: `[conversation-derived]` Earlier baseline under-capacity would make hypernetwork comparisons unfair.
8. Lesson: `[inferred]` Fair capacity controls are essential.
9. How it shaped the final model: `[repo-confirmed]` Current comparisons should use fixed-head controls.
10. Paper placement: `[inferred]` Main baseline.

## 4. Cross-Cutting Failure Themes From Git

### 4.1 Too Many Interacting Axes

- `[repo-confirmed]` The old `group.yaml` exposed dozens of switches for grouping, graph construction, dynamic head scope, distillation, regularization, and initialization.
- `[inferred]` This made experimental conclusions hard to defend because multiple mechanisms changed simultaneously.
- `[inferred]` The clean rewrite solved this by reducing the stack to QMIX/VDN-style learners plus explicit `clean_model_type` variants.

### 4.2 Graphs Were Appealing But Costly and Hard to Position

- `[repo-confirmed]` Git history contains multiple graph attempts: `graph_group`, `graph_pseudo_attn`, `graph_local_subgraph`, `graph_local_fusion`, `gcn`, `standard_gcn`, `gat`, `temporal_gnn`, `edge_gnn`, `relation_gnn`, and `hetero_enemy`.
- `[conversation-derived]` Prior discussion found graph/GAT variants could be too expensive and sometimes used fixed large node slots with sparse information.
- `[inferred]` Graphs helped clarify that relation structure matters, but they were not the cleanest way to support the final hypernetwork story.

### 4.3 Full Dynamic Generation Was Too Broad

- `[repo-confirmed]` Historical code generated full-head or even full-model parameters.
- `[conversation-derived]` Full structured interaction hyper maker had unacceptable runtime in some experiments.
- `[inferred]` The final method became narrower: relation-conditioned adaptation should focus on the interaction-action head.

### 4.4 Distillation and Auxiliary Training Were a Side Quest

- `[repo-confirmed]` Historical learner supported teacher TD, Q distillation, feature distillation, head-parameter distillation, belief KL, prototype losses, group balance/confidence/sparse losses, and threshold-similarity regularizers.
- `[inferred]` These were useful attempts to stabilize or interpret complex models, but they would shift the paper story away from dynamic relation-conditioned decision functions.

### 4.5 Fixed Baselines Became More Important Over Time

- `[conversation-derived]` The user repeatedly noted that fixed structured versions can perform well and that the comparison must be fair.
- `[repo-confirmed]` `10fee58` updated the QMIX minimal baseline to use a fixed MLP head, and `5da7429` added a fixed linear RPG structured control.
- `[inferred]` The paper must treat fixed structured models as strong baselines, not weak strawmen.

## 5. How These Failures Shaped the Current Research Line

- `[inferred]` Early group variants shifted the project from "which agents should be grouped?" to "how should local relation context change the decision rule?"
- `[inferred]` Graph variants showed that relation structure matters, but full graph processing can be expensive, hard to interpret, and not always aligned with CTDE execution assumptions.
- `[inferred]` Full-head/full-model hypernetworks showed that dynamic parameter generation is powerful but can be too broad, expensive, and unstable.
- `[inferred]` Distillation variants showed that stabilization is possible but can create a different research problem.
- `[inferred]` Residual, FiLM, MoE, and smoothness variants are the mature descendants of these failures: they ask more precise questions about how relation patterns should influence an interaction-action scorer.
- `[repo-confirmed]` The current codebase therefore centers on `clean_hyper_agent.py` and the clean model variants, with fixed baselines and relation-conditioned head variants under a common QMIX-style learner.

## 6. Recommended Paper Treatment

### Main Paper

- `[repo-confirmed]` `qmix_minimal` should be a baseline because it is the cleaned QMIX-style fixed MLP version.
- `[repo-confirmed]` `rpg_fixed_linear_structured_maker` should be a main structural baseline because it isolates structure from dynamic parameter generation.
- `[repo-confirmed]` `rpg_linear_interaction_hypercond` should be the main dynamic-head candidate if experiments support it.
- `[repo-confirmed]` `rpg_smooth_linear_interaction_hypercond` can be main or appendix depending on whether it improves performance or produces clearer relation-head alignment.
- `[repo-confirmed]` Relation-head visualization should be used to show whether similar relation patterns map to similar generated heads.

### Appendix

- `[repo-confirmed]` `rpg_residual_interaction_hypercond`, `rpg_film_interaction_hypercond`, and `rpg_moe_interaction_head` are natural appendix/main-ablation candidates.
- `[repo-confirmed]` Relation mixer gate should be appendix only unless it has strong results and a cleaner story.
- `[repo-confirmed]` Graph/GAT variants should be appendix or negative-evidence discussion if results exist.

### Discussion or Not at All

- `[repo-confirmed]` Legacy `group`, `graph_group`, old graph regrouping, distillation, belief, PID, and full-model generation should generally not be main contributions.
- `[inferred]` They can be summarized as internal design exploration if the paper has space for an ablation narrative, but naming every old variant in the paper would distract reviewers.

## 7. Missing Ideas Not Recoverable From Git

The following questions need human author input because Git history identifies code paths but not all motivations, papers, W&B outcomes, or advisor feedback.

- `[needs-human-confirmation]` Which graph-based MARL papers did we discuss when motivating `graph_group`, GCN/GAT, relation GNN, and heterogeneous enemy graph variants?
- `[needs-human-confirmation]` Which hypernetwork papers inspired early full-head and full-model dynamic generation beyond the code-level HyperMARL-style initialization reference?
- `[needs-human-confirmation]` Which variants were run but never committed?
- `[needs-human-confirmation]` Which W&B runs correspond to failed graph, distillation, full-head, residual, FiLM, MoE, and smoothness models?
- `[needs-human-confirmation]` Which models were deleted during refactoring because they failed experimentally, and which were deleted only to simplify the repo?
- `[needs-human-confirmation]` Which failures directly changed the direction of the project from grouping/graph reasoning to relation-conditioned interaction heads?
- `[needs-human-confirmation]` Which advisor comments shaped the current narrative around human-player-inspired reasoning, hard-map emphasis, dynamic MLP heads, or fair fixed baselines?
- `[needs-human-confirmation]` Did any graph/GAT model achieve strong performance before being removed, or were graph variants mainly abandoned due to cost and conceptual mismatch?
- `[needs-human-confirmation]` Did distillation variants fail because of performance, cost, instability, or because they made the research story too complicated?
- `[needs-human-confirmation]` Which exact papers motivated FiLM, MoE/routing, residual dynamic heads, and smoothness regularization?
- `[needs-human-confirmation]` Which map-specific observations first showed that easy maps hide the benefit of dynamic heads?
- `[needs-human-confirmation]` Which current variant is intended as the final main method: `rpg_linear_interaction_hypercond`, `rpg_smooth_linear_interaction_hypercond`, residual, FiLM, MoE, or another successor?

## 8. Compact Research-Lineage Summary

- `[repo-confirmed]` Git history shows a broad exploration before the current clean model: group-conditioned GoMARL, graph-based grouping, learned structural grouping, graph/node input fusion, full-head and full-model hypernetworks, graph-conditioned full heads, residual full heads, episode/k-step/EMA caching, teacher-student distillation, belief latents, and multiple regularizers.
- `[repo-confirmed]` Commit `936cbf5` deleted the legacy exploratory stack and replaced it with a clean QMIX/VDN-style setup centered on `clean_hyper_agent.py`.
- `[inferred]` The research question narrowed over time from "can learned groups/graphs improve MARL coordination?" to "can local relation patterns adapt the action-value decision function?"
- `[inferred]` The final model direction is strongest when framed as relation-conditioned interaction-action head adaptation, not as generic graph reasoning, generic grouping, or generic hypernetwork use.
- `[inferred]` Failed and abandoned variants are useful because they justify why the final design is lightweight, structured, and focused on interaction actions rather than dynamically generating every part of the agent.
