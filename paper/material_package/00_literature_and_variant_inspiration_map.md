# Literature and Variant Inspiration Map

This file supplements `00_full_research_lineage.md`. It focuses on the external intellectual sources and design motivations behind model variants, including failed and abandoned directions.

It does not rewrite the previous lineage file and is not a paper draft.

Evidence tags:

- `[repo-confirmed]`: supported by repository code, configs, README, comments, or existing material-package files.
- `[conversation-derived]`: reconstructed from visible prior Codex/user conversation or summarized conversation memory.
- `[inferred]`: logically inferred from code, model names, comments, experiment setup, or design changes.
- `[needs-human-confirmation]`: important but not directly verifiable from accessible local context.

Important citation boundary:

- Exact paper titles should be verified before manuscript use. [needs-human-confirmation]
- I can confirm GoMARL from `README.md`, QMIX/VDN as implemented learner/mixer names, RPG from repository comments and conversation, and CTEM from conversation. [repo-confirmed] [conversation-derived]
- For HyperNetworks, FiLM, GCN, GAT, MoE, residual learning, smoothness/metric-learning style regularization, and graph/entity MARL, I can reconstruct the idea line and likely citation categories, but the exact paper(s) that you discussed must be confirmed by the human author. [needs-human-confirmation]

## 1. External Literature Inspiration Map

### 1.1 Hypernetworks / Dynamic Parameter Generation

Core idea:

- A hypernetwork generates the parameters of another network from a condition vector. In this project, that target network is the local action-value head or interaction-action scorer. [repo-confirmed]
- The repository contains multiple generated-head variants: `baseline`, `hypermarl_id`, `hypermarl_fullnet`, `rpg_relation_hypercond`, `rpg_structured_hypercond`, `rpg_full_structured_hypercond`, `rpg_readout_structured_hypercond`, `rpg_linear_interaction_hypercond`, and later constrained variants. [repo-confirmed: `src/config/algs/clean_hyper.yaml`, `src/modules/agents/clean_hyper_agent.py`]

Known or likely inspiration sources:

- General HyperNetworks literature: a network produces weights for another network. This is a likely conceptual ancestor for all generated-head models. [inferred] [needs-human-confirmation for exact cited paper and whether it was discussed]
- HyperMARL-style agent-id hypernetwork: the config explicitly names `hypermarl_id` and `hypermarl_fullnet`. `hypermarl_fullnet` is described as a HyperMARL-style learned `e_i` table feeding an MLP hypernetwork that generates the full post-RNN two-layer action network. [repo-confirmed]
- QMIX itself uses hypernetworks inside the mixer to generate non-negative mixing weights from global state in standard implementations; our project discussed that this is also a kind of hypernetwork generation. [conversation-derived] [repo-confirmed for QMIX mixer use in learner, but exact mixer internals should be checked before citing]
- RPG-inspired relation pattern: instead of task embedding or agent identity, our final main direction uses local self-ally-enemy relation patterns as the condition for dynamic decision heads. [repo-confirmed]

Condition sources in related ideas:

- Agent identity: `hypermarl_id` and `hypermarl_fullnet` use learned agent embeddings. [repo-confirmed]
- Local observation/context: `baseline` uses local condition from observation, previous action, and recurrent hidden state. [repo-confirmed]
- Route/codebook: `dynamic_route` and `rpg_relation_route` convert condition into route logits and codebook selection. [repo-confirmed]
- Relation pattern: RPG-inspired variants use self/ally/enemy relation condition. [repo-confirmed]
- Task embedding: RPG paper may use task-specific embeddings or continual-learning components, but our implementation comments explicitly state we do not reproduce RPG task embeddings or continual-learning regularizers. [repo-confirmed] [needs-human-confirmation for exact RPG mechanism]

How our use differs:

- Our main use is not task-conditioned or identity-conditioned full policy generation. It is relation-conditioned interaction-action scoring. [repo-confirmed]
- The condition is an online local relation pattern extracted from each agent's current self/ally/enemy observation, not a static agent id or task id. [repo-confirmed]
- The target of generation was narrowed from a full two-layer action head to a lightweight one-layer interaction scorer over `[agent hidden, enemy token]`. [repo-confirmed]

Variants from this line:

- `baseline`: local observation/previous-action/hidden condition generates a unified head. [repo-confirmed]
- `hypermarl_id`: learned agent id embedding conditions generated head. [repo-confirmed]
- `hypermarl_fullnet`: HyperMARL-style learned `e_i` generates full post-RNN two-layer action network. [repo-confirmed]
- `rpg_relation_hypercond`: RPG relation condition feeds generated head. [repo-confirmed]
- `rpg_structured_hypercond`: relation-conditioned structured maker with generated ego branch and conditioned interaction scorer. [repo-confirmed]
- `rpg_full_structured_hypercond`: both ego and interaction branches generated by hypernetworks. [repo-confirmed]
- `rpg_readout_structured_hypercond`: relation condition generates only the final interaction readout. [repo-confirmed]
- `rpg_linear_interaction_hypercond`: relation condition generates one linear interaction scorer. [repo-confirmed]
- `rpg_residual_interaction_hypercond`: relation-generated residual scorer corrects fixed interaction scorer. [repo-confirmed]
- `rpg_smooth_linear_interaction_hypercond`: generated linear scorer plus smoothness regularizer. [repo-confirmed]

Research role:

- This literature line supports the central claim that dynamic parameterization can represent a family of local decision functions. [inferred]
- It does not by itself prove performance improvement; the fixed relation-conditioned control is required to isolate dynamic generation from extra relation features. [repo-confirmed]

Missing citation questions:

- Which specific HyperNetworks paper did you intend to cite for dynamic parameter generation? [needs-human-confirmation]
- Which HyperMARL paper/version motivated `hypermarl_id` and `hypermarl_fullnet`? [needs-human-confirmation]
- Did we discuss QMIX mixer hypernetworks as a rhetorical bridge, or only as background? [needs-human-confirmation]

### 1.2 Graph-based and Entity-relation MARL

Core idea:

- Cooperative MARL in entity-rich environments can model agents and entities as nodes/tokens and use graph neural networks or attention to reason over relations. [inferred]
- This project tried several graph alternatives to RPG-style self-centered cross-attention. [repo-confirmed]

Known implementation sources:

- `graph_hypercond`: observation-only graph plus standard GCN condition. [repo-confirmed]
- `graph_route`: observation-only graph plus standard GCN followed by route/codebook. [repo-confirmed]
- `two_graph_gat_hypercond`: CTDE ego-graph GAT with two local graphs, self+allies and self+enemies. [repo-confirmed]
- `hetero_gat_hypercond`: CTDE ego-heterogeneous GAT with typed self-loop, ally-to-self, and enemy-to-self messages. [repo-confirmed]
- `global_two_graph_gat_hypercond`: CTCE whole-graph upper-bound model with friendly graph, enemy graph, and cross-graph attention. [repo-confirmed]
- `global_hetero_gat_hypercond`: CTCE whole-graph heterogeneous graph with node-type and edge-type embeddings. [repo-confirmed]

Likely external idea families:

- GCN-style message passing over learned or constructed adjacency. [repo-confirmed as implementation family] [needs-human-confirmation for exact paper]
- GAT-style attention over graph neighbors. [repo-confirmed as implementation family] [needs-human-confirmation for exact paper]
- Entity-centric MARL attention, where agents attend over entities such as allies/enemies. [inferred] [needs-human-confirmation for exact papers]
- Coordination graphs or relational MARL, where pairwise/multi-agent relations guide coordination. [inferred] [needs-human-confirmation for exact papers]
- Heterogeneous graph neural networks, where node/edge types encode semantic relation types. [repo-confirmed as implementation family] [needs-human-confirmation for exact paper]

Why graph-based approaches were appealing:

- They seem naturally aligned with SMAC: allies, enemies, and self can be viewed as entities with relations. [inferred]
- They could avoid hand-designed fixed ordering by modeling relations as edges or attention neighborhoods. [conversation-derived]
- They offered a possible answer to the user's question: "If pattern can be built with attention, can graph also work?" [conversation-derived]

What was tried:

- Local ego graph variants: two-graph GAT and hetero GAT. [repo-confirmed]
- Global graph variants: global two-graph GAT and global hetero GAT, labelled CTCE validation/upper-bound modes. [repo-confirmed]
- Generic graph condition variants: GCN condition and graph route. [repo-confirmed]

Why graph-based approaches were not kept as main direction:

- Cost: the user observed that graph versions increased training time dramatically, reaching around a day or more compared with earlier versions. [conversation-derived]
- Repeated local graph computation: local ego graph construction per agent created overhead. [conversation-derived]
- CTDE mismatch: global graph variants reduce repeated computation but become CTCE validation because graph construction uses all agents' observations at execution. [repo-confirmed]
- Conceptual mismatch: graph variants still did not clearly solve the fixed-slot observation issue in the way the user initially expected; in one discussion, the graph was described as large with many empty or uninformative slots. [conversation-derived]
- Weak story relative to head adaptation: graph reasoning changes relation encoding, while the paper's sharper question became whether relation patterns should change decision functions. [inferred]

What was learned:

- Explicit graph reasoning is attractive but can be too expensive for the available SMAC training budget. [conversation-derived]
- Relation extraction and decision-function adaptation should be separated: graph encoders are one possible relation extractor, but not the core contribution unless results justify them. [inferred]
- The final paper should not overclaim graph novelty if graph variants are not used as main evidence. [inferred]

Missing citation questions:

- Which specific graph MARL papers did we discuss when motivating GAT/GNN/entity relation variants? [needs-human-confirmation]
- Did we discuss DGN, G2ANet, TarMAC, MAGIC, UPDeT, Entity Attention, or other entity-centric MARL papers? [needs-human-confirmation]
- Which graph paper, if any, motivated the two-graph versus heterogeneous-graph distinction? [needs-human-confirmation]

### 1.3 FiLM-style Conditioning

Core idea:

- FiLM-style conditioning uses a condition vector to generate feature-wise scale and bias, modulating intermediate features instead of generating the whole network weights. [repo-confirmed for implementation; exact FiLM paper needs confirmation]

Variant:

- `rpg_film_interaction_hypercond` keeps the generated ego branch, encodes `[agent hidden, enemy token]` with a fixed interaction encoder, and lets relation pattern generate FiLM gamma/beta modulation before the final attack score. [repo-confirmed]

Why FiLM seemed safer:

- Full generated heads can be expensive or unstable. [conversation-derived]
- A fixed encoder provides a stable interaction representation; FiLM allows the relation pattern to adapt this representation without replacing all weights. [inferred]
- This is a middle ground between fixed relation-conditioned scoring and fully generated scoring. [inferred]

Hypothesis tested:

- Relation-conditioned modulation may capture context-specific interaction scoring with less over-parameterization than generated weights. [inferred]
- If FiLM performs well, it supports the broader claim that relation patterns should adapt the decision function, even if direct weight generation is not the best mechanism. [inferred]

Outcome:

- Implemented as an active improvement variant. [repo-confirmed]
- Final experimental outcome is not recorded in accessible material. [needs-human-confirmation]

Paper positioning:

- Main text if it becomes one of the best-performing or most stable variants. [needs-human-confirmation]
- Otherwise, present as an ablation showing a constrained modulation alternative to direct hypernetwork generation. [inferred]
- Describe as "FiLM-style relation-conditioned modulation" unless exact FiLM citation is verified. [needs-human-confirmation]

Missing citation questions:

- Which FiLM paper or conditional-modulation paper did you want to cite? [needs-human-confirmation]
- Did the FiLM idea come from visual reasoning literature, conditional normalization, federated personalization, or another source? [needs-human-confirmation]

### 1.4 Mixture-of-Experts / Routing

Core idea:

- MoE/routing methods select or mix among multiple expert functions instead of using one shared function or generating arbitrary weights. [repo-confirmed for implemented MoE-style variant] [needs-human-confirmation for exact external papers]

Variants:

- `dynamic_route`: local condition selects a route/codebook embedding before head generation. [repo-confirmed]
- `rpg_relation_route`: relation pattern selects a learned route/codebook embedding. [repo-confirmed]
- `rpg_moe_interaction_head`: relation condition selects a soft mixture of fixed interaction experts. [repo-confirmed]

Hypothesis tested:

- Relation patterns may correspond to soft interaction regimes. [inferred]
- Instead of generating a new scorer for each relation pattern, the model can select among several fixed interaction experts. [repo-confirmed] [inferred]
- This may improve stability and interpretability because each expert can be seen as a reusable decision rule. [inferred]

Why it might be more stable:

- Fixed experts avoid generating arbitrary scorer weights. [inferred]
- Soft gating keeps the model differentiable and lets relation patterns interpolate between regimes. [repo-confirmed] [inferred]
- It may reduce parameter-space noise compared with full hypernetwork generation. [inferred]

Outcome:

- Implemented. [repo-confirmed]
- Final experimental outcome is not recorded in accessible material. [needs-human-confirmation]

Paper positioning:

- Likely ablation or appendix unless it becomes a main-performing variant. [inferred]
- Useful to show the design space of relation-conditioned decision adaptation: direct generation, residual correction, FiLM modulation, and expert selection. [inferred]

Missing citation questions:

- Which MoE/routing paper motivated this variant? [needs-human-confirmation]
- Was the expert-head idea inspired by routing networks, role-based MARL, option policies, or sparse MoE? [needs-human-confirmation]

### 1.5 Residual Dynamic Heads

Core idea:

- Residual dynamic heads keep a fixed base decision rule and add a relation-generated correction. [repo-confirmed]

Variant:

- `rpg_residual_interaction_hypercond`: fixed interaction scorer plus gated relation-generated linear residual scorer. [repo-confirmed]

Conceptual difference from fully generated heads:

- Fully generated heads replace the scorer parameters with condition-generated parameters. [repo-confirmed]
- Residual dynamic heads preserve a stable fixed scorer and only add a relation-dependent correction. [repo-confirmed]
- A learned gate decides how strongly the dynamic residual participates. [repo-confirmed]

Problem it was intended to solve:

- Fixed controls can be strong, so replacing the fixed scorer everywhere may be unnecessary. [conversation-derived]
- Full dynamic generation can be unstable or expensive. [conversation-derived]
- A residual branch can express "default rule plus relation-specific adjustment," which is easier to defend than "new rule for every relation pattern." [inferred]

What it taught us:

- Dynamic adaptation can be framed as correction rather than wholesale parameter generation. [inferred]
- This gives a cleaner story if direct hypernetwork generation has noisy or mixed results. [inferred]

Outcome:

- Implemented. [repo-confirmed]
- Final experiment status is not recorded in accessible material. [needs-human-confirmation]

Missing citation questions:

- Was this inspired by ResNet/residual learning generally, personalization residual adapters, or another paper? [needs-human-confirmation]
- Did any advisor or paper suggest retaining a base policy/rule and learning residual adaptation? [needs-human-confirmation]

### 1.6 Smoothness Regularization / Relation-to-head Continuity

Core idea:

- Similar relation patterns should produce similar decision-head parameters. [repo-confirmed] [conversation-derived]

Variant:

- `rpg_smooth_linear_interaction_hypercond`: same generated linear interaction scorer as `rpg_linear_interaction_hypercond`, plus KNN smoothness regularizer encouraging nearby relation patterns to generate nearby interaction heads. [repo-confirmed]

Hypothesis tested:

- The mapping from relation condition to generated head should be continuous rather than arbitrary. [repo-confirmed] [conversation-derived]
- If two agents face similar coordination relations, their generated interaction scorers should be similar. [conversation-derived]

Intended purpose:

- Performance: may improve generalization and reduce noisy generated weights. [inferred]
- Stability: may regularize the hypernetwork mapping. [inferred]
- Interpretability: directly supports relation-head alignment visualization. [repo-confirmed]
- Visualization: gives a stronger reason to inspect relation-space and head-parameter-space trajectories. [repo-confirmed]

Relation to analysis:

- The visualization package includes relation/head similarity heatmaps, distance-alignment scatter, and relation/head dynamics video. [repo-confirmed]
- A positive relation-distance to head-distance alignment supports, but does not prove, that the hypernetwork learned a meaningful relation-to-decision-function mapping. [repo-confirmed]

Outcome:

- Implemented and actively requested for `5m_vs_6m`/`5m6m` experiments. [repo-confirmed] [conversation-derived]
- Final performance outcome is not recorded here. [needs-human-confirmation]

Missing citation questions:

- Which metric-learning, smoothness regularization, manifold regularization, contrastive learning, or personalization paper inspired this? [needs-human-confirmation]
- Was this mainly your own hypothesis from visualization needs, or connected to a specific paper? [needs-human-confirmation]

### 1.7 Relation Mixer Gate / Training-side Relation Conditioning

Core idea:

- Relation condition may influence centralized credit assignment by reweighting per-agent Q-values before QMIX mixing. [repo-confirmed]

Variant:

- `clean_relation_mixer_gate`: optional training-only gate in `CleanLearner`; it computes positive softmax gates from relation conditions and multiplies each agent's selected Q before QMIX. [repo-confirmed]

Possible inspiration sources:

- Attention/gating over agents. [inferred]
- Value factorization and credit assignment. [inferred]
- Relation-conditioned importance weighting. [inferred]
- Exact paper source is not recorded. [needs-human-confirmation]

Why less central than local head adaptation:

- The head adaptation story directly connects relation pattern to action scoring. [inferred]
- The mixer gate only affects centralized training, not decentralized execution. [repo-confirmed]
- The user explicitly questioned the logic of relation-conditioned mixer: why should relation pattern change mixer weights? [conversation-derived]
- Without strong evidence, the mixer gate risks looking like an arbitrary module. [inferred]

Paper positioning:

- Appendix or future-work candidate unless experiments show clear gains and the conceptual argument is refined. [inferred]
- If included, describe as "training-side relation-conditioned credit gate," not as the main contribution. [repo-confirmed] [inferred]

Missing citation questions:

- Which attention-based credit assignment or value-factorization paper motivated relation mixer gating? [needs-human-confirmation]
- Did we run any successful relation mixer gate experiments? [needs-human-confirmation]

### 1.8 Human-player-inspired Reasoning

Core idea:

- Human players often reason separately about their own unit, allies, enemies, and target-specific interactions. [conversation-derived]
- This provides an intuitive narrative for self/ally/enemy observation decomposition and ego/interaction action decomposition. [conversation-derived]

Important boundary:

- The project does not use human demonstrations. [repo-confirmed by absence of demonstration pipeline] [inferred]
- The model should not be called human-player mimicry or imitation learning. [conversation-derived]
- The safer phrasing is "human-player inspired relation-aware reasoning" or "human-inspired structural intuition." [conversation-derived]

Connection to model:

- Self/ally/enemy encoders match the intuition of separately reading self status, ally support, and enemy threat/targets. [repo-confirmed] [conversation-derived]
- Interaction-action scoring matches target-specific decision making: each attack Q-value depends on the agent hidden state and one enemy token. [repo-confirmed]
- Relation pattern conditioning represents the idea that the local decision rule can change with battle context. [inferred]

Missing questions:

- Did an advisor explicitly encourage or discourage human-player-inspired language? [needs-human-confirmation]
- Which human-player/game-AI papers, if any, were discussed? [needs-human-confirmation]

### 1.9 Narrative-only Papers

Definition:

- Narrative-only papers are papers used mainly to learn writing, framing, taxonomy, or result interpretation, not direct technical ancestry. [inferred]

Known or suspected narrative functions:

- Taxonomy: organize related work into value factorization, relation/entity reasoning, dynamic parameterization, and structured decision heads. [inferred]
- Prior-information defense: explain that using environment-provided entity/action semantics is an inductive bias, not hidden expert knowledge. [inferred]
- Easy-vs-hard map narrative: explain why a method may be competitive rather than superior on easy maps but stronger on hard/relation-sensitive maps. [conversation-derived]
- Figure strategy: show method as a pipeline from observation split to relation pattern to generated/modulated head to QMIX. [repo-confirmed via visualization and material plans] [inferred]

Concrete papers:

- CTEM was discussed as a recent benchmark comparison and possibly narrative reference, but exact role needs confirmation. [conversation-derived] [needs-human-confirmation]
- Other narrative-only papers are not recorded in repository-accessible material. [needs-human-confirmation]

Paper positioning:

- Keep narrative-only papers out of closest-technical-ancestor paragraphs unless their methods are actually related. [inferred]
- Use them to guide writing style, taxonomy, or result interpretation only. [inferred]

## 2. Model Variant Inspiration Map

### `qmix_minimal`

1. Direct technical inspiration: QMIX-style recurrent local agent with fixed MLP Q head. [repo-confirmed]
2. Broader motivation: fair minimal baseline after concern that earlier baseline had too little capacity. [conversation-derived]
3. Hypothesis tested: relation-conditioned/dynamic variants should beat a fair fixed-head QMIX baseline, not a weakened baseline. [conversation-derived]
4. Changed part: fixed two-layer head after GRU, no hypernetwork. [repo-confirmed]
5. Outcome: earlier screenshots suggested minimal QMIX was much slower on `3s5z`-type tasks, but exact numbers need confirmation. [conversation-derived] [needs-human-confirmation]
6. Status: kept as baseline. [repo-confirmed]
7. Story contribution: establishes lower-bound baseline and fairness discipline. [inferred]

### `baseline`

1. Direct technical inspiration: generic local hypernetwork conditioning. [repo-confirmed]
2. Broader motivation: test whether local context-conditioned parameter generation helps before relation-specific design. [inferred]
3. Hypothesis tested: observation/previous-action/hidden context can generate a useful dynamic head. [inferred]
4. Changed part: local condition from `obs + prev_action + h` generates unified head. [repo-confirmed]
5. Outcome: not current main story. [inferred]
6. Status: background/early clean-hyper baseline. [repo-confirmed]
7. Story contribution: shows progression from generic dynamic heads to relation-conditioned heads. [inferred]

### `hypermarl_id`

1. Direct technical inspiration: HyperMARL-style agent identity conditioning. [repo-confirmed]
2. Broader motivation: test whether dynamic heads from learned agent embeddings specialize agents. [inferred]
3. Hypothesis tested: static agent identity can condition action-head parameters. [inferred]
4. Changed part: learned `e_i` embedding feeds condition encoder for generated head. [repo-confirmed]
5. Outcome: not recorded as final main result. [needs-human-confirmation]
6. Status: background baseline. [repo-confirmed]
7. Story contribution: contrasts static identity-conditioned generation with online relation-conditioned generation. [inferred]

### `hypermarl_fullnet`

1. Direct technical inspiration: HyperMARL-style learned `e_i` table generating full post-RNN action-network parameters. [repo-confirmed]
2. Broader motivation: full generated local head baseline. [repo-confirmed] [inferred]
3. Hypothesis tested: full action-network generation can specialize decision rules by agent identity. [inferred]
4. Changed part: `MLPHyperParameterGenerator` generates two-layer post-RNN action head. [repo-confirmed]
5. Outcome: not recorded as final main result. [needs-human-confirmation]
6. Status: background hypernetwork baseline. [repo-confirmed]
7. Story contribution: shows why our final condition is relation pattern, not static identity. [inferred]

### `dynamic_route`

1. Direct technical inspiration: routing/codebook/expert-selection family. [repo-confirmed] [needs-human-confirmation for exact paper]
2. Broader motivation: constrain dynamic head conditioning through route embeddings. [inferred]
3. Hypothesis tested: selecting among learned condition codes may stabilize generation. [inferred]
4. Changed part: local condition produces route logits and route codebook embedding. [repo-confirmed]
5. Outcome: not central; exact results unknown. [needs-human-confirmation]
6. Status: early/exploratory variant. [repo-confirmed]
7. Story contribution: precursor to relation-route and MoE-style thinking. [inferred]

### `local_structured_hypercond`

1. Direct technical inspiration: structured ego/interaction decision maker without RPG relation capturer. [repo-confirmed]
2. Broader motivation: isolate whether action-structure split helps independently of relation-pattern extraction. [repo-confirmed] [inferred]
3. Hypothesis tested: structured maker itself may improve action scoring. [inferred]
4. Changed part: local condition source plus ego-action/enemy-interaction maker split. [repo-confirmed]
5. Outcome: initially missing/unknown model type in an early run, later implemented. [conversation-derived] [repo-confirmed]
6. Status: control/exploratory. [repo-confirmed]
7. Story contribution: separates "condition source" from "decision-maker structure." [inferred]

### `rpg_relation_hypercond`

1. Direct technical inspiration: RPG relation pattern extraction. [repo-confirmed]
2. Broader motivation: relation-aware local context should be better than raw local context. [repo-confirmed] [inferred]
3. Hypothesis tested: self/ally/enemy relation pattern is a useful hypernetwork condition. [repo-confirmed]
4. Changed part: RPG-inspired relation capturer replaces generic local condition. [repo-confirmed]
5. Outcome: earlier results suggested faster convergence than minimal QMIX on `3s5z`-type settings. [conversation-derived] [needs-human-confirmation]
6. Status: ancestor to structured variants. [repo-confirmed]
7. Story contribution: relation pattern became the central condition signal. [repo-confirmed]

### `rpg_relation_route`

1. Direct technical inspiration: RPG relation pattern plus route/codebook selection. [repo-confirmed]
2. Broader motivation: relation patterns may select discrete coordination/decision regimes. [inferred]
3. Hypothesis tested: relation-conditioned routing may be more stable or interpretable than direct generation. [inferred]
4. Changed part: relation condition -> route logits -> codebook condition. [repo-confirmed]
5. Outcome: not recorded as central. [needs-human-confirmation]
6. Status: exploratory. [repo-confirmed]
7. Story contribution: connects relation patterns to regime selection, later echoed by MoE. [inferred]

### `rpg_structured_hypercond`

1. Direct technical inspiration: RPG-style structured maker split. [repo-confirmed]
2. Broader motivation: non-attack/self actions and attack/interaction actions have different semantics. [repo-confirmed] [inferred]
3. Hypothesis tested: relation-conditioned structured action decomposition improves Q estimation. [inferred]
4. Changed part: splits Q into ego-action and enemy-interaction branches; ego branch generated, interaction branch fixed/conditioned scorer. [repo-confirmed]
5. Outcome: became early strong version in `3s5z` experiments, but exact results need confirmation. [conversation-derived] [needs-human-confirmation]
6. Status: important ancestor, not final current main due to later fairness/cost refinements. [repo-confirmed] [conversation-derived]
7. Story contribution: introduced the decision-maker split that remains central. [repo-confirmed]

### `rpg_full_structured_hypercond`

1. Direct technical inspiration: stronger RPG-like hypernetwork maker where both ego and interaction branches are generated. [repo-confirmed]
2. Broader motivation: fix fairness concern that enemy-scoring branch was not generated in earlier structured version. [conversation-derived]
3. Hypothesis tested: fully generated structured maker should be closer to the intended dynamic-decision-function story. [repo-confirmed] [inferred]
4. Changed part: generates both interaction bottleneck and interaction output parameters. [repo-confirmed]
5. Outcome: user reported runtime around two days, much higher than earlier versions. [conversation-derived]
6. Status: abandoned/superseded by cheaper readout and linear variants. [conversation-derived] [repo-confirmed]
7. Story contribution: negative evidence that full dynamic generation is too expensive; motivates lightweight interaction-only generation. [inferred]

### `rpg_readout_structured_hypercond`

1. Direct technical inspiration: partial hypernetwork generation / generated readout. [repo-confirmed]
2. Broader motivation: reduce full structured maker cost. [repo-confirmed] [inferred]
3. Hypothesis tested: fixed interaction encoder plus generated final readout may preserve dynamic adaptation at lower cost. [repo-confirmed]
4. Changed part: fixed encoder over `[hidden, enemy token]`, relation-generated final readout. [repo-confirmed]
5. Outcome: user clarified the intended comparison was a one-layer interaction branch, not this exact readout structure. [conversation-derived]
6. Status: intermediate/possibly appendix. [inferred]
7. Story contribution: demonstrates iterative cost-control design. [inferred]

### `rpg_fixed_structured_maker`

1. Direct technical inspiration: fixed relation-conditioned control. [repo-confirmed]
2. Broader motivation: isolate whether relation pattern and manual structure explain gains without hypernetwork generation. [repo-confirmed] [conversation-derived]
3. Hypothesis tested: fixed relation-conditioned structured network may already be enough. [conversation-derived]
4. Changed part: fixed ego and interaction MLPs conditioned by concatenated relation pattern. [repo-confirmed]
5. Outcome: fixed versions sometimes performed strongly, weakening broad hypernetwork claims. [conversation-derived]
6. Status: important control, later refined to fixed-linear control. [repo-confirmed]
7. Story contribution: forces the paper to make a precise dynamic-generation claim. [inferred]

### `rpg_fixed_linear_structured_maker`

1. Direct technical inspiration: matched fixed control for one-layer dynamic interaction head. [repo-confirmed]
2. Broader motivation: fair comparison with `rpg_linear_interaction_hypercond`. [conversation-derived] [repo-confirmed]
3. Hypothesis tested: dynamic parameter generation helps beyond fixed relation-conditioned one-layer scoring. [repo-confirmed] [inferred]
4. Changed part: fixed one-layer interaction scorer over `[hidden, relation condition, enemy token]`. [repo-confirmed]
5. Outcome: on `5m6m`, fixed and dynamic were both high; on `corridor`, dynamic appeared clearly better in screenshots; on `MMM2`, dynamic appeared faster but both solved. [repo-confirmed via material package] [conversation-derived]
6. Status: central fixed control. [repo-confirmed]
7. Story contribution: most important ablation for defending dynamic head generation. [repo-confirmed]

### `rpg_linear_interaction_hypercond`

1. Direct technical inspiration: lightweight hypernetwork generation and structured interaction scoring. [repo-confirmed]
2. Broader motivation: focus dynamic generation where it is most semantically meaningful: target-specific interaction actions. [inferred]
3. Hypothesis tested: relation-generated one-layer interaction scorer is enough to capture adaptive target scoring with lower cost. [repo-confirmed] [conversation-derived]
4. Changed part: relation condition generates a single linear scorer applied to `[agent hidden, enemy token]`. [repo-confirmed]
5. Outcome: current main dynamic model; preliminary results favorable on `corridor` and fast on `MMM2`; exact final results need confirmation. [repo-confirmed] [conversation-derived]
6. Status: main candidate. [repo-confirmed]
7. Story contribution: final practical form of relation-conditioned dynamic interaction head. [repo-confirmed]

### `rpg_residual_interaction_hypercond`

1. Direct technical inspiration: residual correction over fixed base rule. [repo-confirmed] [needs-human-confirmation for external paper]
2. Broader motivation: retain strong fixed scorer while allowing relation-specific dynamic correction. [repo-confirmed] [inferred]
3. Hypothesis tested: dynamic adaptation is useful as a gated correction, not necessarily as full replacement. [inferred]
4. Changed part: fixed interaction scorer plus gated generated residual scorer. [repo-confirmed]
5. Outcome: implemented; final result unknown. [needs-human-confirmation]
6. Status: active improvement variant. [repo-confirmed]
7. Story contribution: helps defend against instability and fixed-control competitiveness. [inferred]

### `rpg_film_interaction_hypercond`

1. Direct technical inspiration: FiLM-style feature-wise modulation. [repo-confirmed] [needs-human-confirmation for exact paper]
2. Broader motivation: adapt interaction features without generating full weights. [repo-confirmed] [inferred]
3. Hypothesis tested: relation-conditioned modulation can be a stable middle ground. [inferred]
4. Changed part: fixed interaction encoder plus relation-generated gamma/beta before fixed scorer. [repo-confirmed]
5. Outcome: implemented; final result unknown. [needs-human-confirmation]
6. Status: active improvement/ablation. [repo-confirmed]
7. Story contribution: broadens design space beyond direct hypernetwork weight generation. [inferred]

### `rpg_moe_interaction_head`

1. Direct technical inspiration: mixture-of-experts / routing / expert selection. [repo-confirmed] [needs-human-confirmation]
2. Broader motivation: relation patterns may select among reusable interaction regimes. [inferred]
3. Hypothesis tested: expert mixing is more stable/interpretable than generating arbitrary weights. [inferred]
4. Changed part: relation condition gates multiple fixed interaction expert heads. [repo-confirmed]
5. Outcome: implemented; final result unknown. [needs-human-confirmation]
6. Status: active improvement/ablation. [repo-confirmed]
7. Story contribution: relation patterns as regime selectors. [inferred]

### `rpg_smooth_linear_interaction_hypercond`

1. Direct technical inspiration: relation-to-head continuity / smoothness regularization. [repo-confirmed] [conversation-derived]
2. Broader motivation: make generated heads interpretable and less arbitrary. [conversation-derived] [inferred]
3. Hypothesis tested: nearby relation patterns should generate nearby interaction-head parameters. [repo-confirmed]
4. Changed part: adds KNN smoothness auxiliary loss to generated linear interaction head. [repo-confirmed]
5. Outcome: implemented and requested for new experiments; final result unknown. [conversation-derived]
6. Status: active improvement variant. [repo-confirmed]
7. Story contribution: connects model design to visualization and mechanism evidence. [repo-confirmed]

### `graph_hypercond`

1. Direct technical inspiration: GCN-style graph encoding. [repo-confirmed]
2. Broader motivation: replace hand relation pattern with learned observation graph. [inferred]
3. Hypothesis tested: graph convolution over agent observations can provide better hypernetwork condition. [inferred]
4. Changed part: observation graph encoder feeds condition encoder. [repo-confirmed]
5. Outcome: CTCE validation mode, not main result. [repo-confirmed]
6. Status: exploratory/appendix if used. [inferred]
7. Story contribution: negative/alternative path toward relation extraction. [inferred]

### `graph_route`

1. Direct technical inspiration: graph encoding plus route/codebook selection. [repo-confirmed]
2. Broader motivation: combine graph relations with regime selection. [inferred]
3. Hypothesis tested: graph-derived condition can choose useful route embeddings. [inferred]
4. Changed part: graph condition -> route logits -> codebook. [repo-confirmed]
5. Outcome: CTCE validation mode, not main result. [repo-confirmed]
6. Status: exploratory. [repo-confirmed]
7. Story contribution: shows graph and routing ideas were explored but not central. [inferred]

### `two_graph_gat_hypercond`

1. Direct technical inspiration: GAT-style ego graph reasoning. [repo-confirmed]
2. Broader motivation: model self+ally and self+enemy relations as two separate local graphs. [repo-confirmed]
3. Hypothesis tested: graph attention may capture relation structure better than RPG cross-attention. [conversation-derived] [repo-confirmed]
4. Changed part: replaces RPG relation capturer with two local GAT graphs. [repo-confirmed]
5. Outcome: expensive in preliminary runtime discussions. [conversation-derived]
6. Status: implemented but not main. [repo-confirmed] [inferred]
7. Story contribution: failed/alternative relation encoder; reinforces efficiency of final attention-based capturer. [inferred]

### `hetero_gat_hypercond`

1. Direct technical inspiration: heterogeneous GAT / typed relations. [repo-confirmed]
2. Broader motivation: encode ally/enemy/self relation types explicitly. [repo-confirmed] [inferred]
3. Hypothesis tested: typed relation messages improve relation condition quality. [inferred]
4. Changed part: typed self-loop, ally-to-self, enemy-to-self messages with type-level attention. [repo-confirmed]
5. Outcome: expensive and not current main. [conversation-derived]
6. Status: implemented exploratory variant. [repo-confirmed]
7. Story contribution: tested richer relational inductive bias but cost/story were worse than final path. [inferred]

### `global_two_graph_gat_hypercond`

1. Direct technical inspiration: whole-graph GAT / CTCE upper-bound reasoning. [repo-confirmed]
2. Broader motivation: reduce repeated per-agent graph computation by computing whole graphs once. [conversation-derived]
3. Hypothesis tested: central whole-graph relation encoding may be stronger or cheaper in aggregate. [inferred]
4. Changed part: friendly graph plus enemy graph with cross-graph attention. [repo-confirmed]
5. Outcome: labelled CTCE validation mode because execution uses all agents' observations. [repo-confirmed]
6. Status: upper-bound/exploratory. [repo-confirmed]
7. Story contribution: warns that efficiency fixes can change the execution setting. [inferred]

### `global_hetero_gat_hypercond`

1. Direct technical inspiration: global heterogeneous graph reasoning. [repo-confirmed]
2. Broader motivation: typed whole-graph relation encoding. [repo-confirmed] [inferred]
3. Hypothesis tested: global typed graph can generate useful relation condition. [inferred]
4. Changed part: friendly and enemy nodes in one typed graph with node/edge type embeddings. [repo-confirmed]
5. Outcome: CTCE validation mode, not main CTDE result. [repo-confirmed]
6. Status: exploratory/upper-bound. [repo-confirmed]
7. Story contribution: emphasizes CTDE boundary. [repo-confirmed]

### `clean_relation_mixer_gate`

1. Direct technical inspiration: relation-conditioned gating / credit assignment. [repo-confirmed] [needs-human-confirmation for exact paper]
2. Broader motivation: relation patterns might influence agent contribution in centralized training. [inferred]
3. Hypothesis tested: relation-aware reweighting before QMIX improves training. [repo-confirmed] [inferred]
4. Changed part: learner multiplies selected agent Q-values by positive relation gates before mixer. [repo-confirmed]
5. Outcome: implemented but conceptually questioned; final result unknown. [conversation-derived] [needs-human-confirmation]
6. Status: likely appendix/future work. [inferred]
7. Story contribution: shows why final paper should focus on local head adaptation, where the logic is cleaner. [inferred]

### Auxiliary losses

1. Direct technical inspiration: smoothness/metric regularization over generated parameters. [repo-confirmed] [needs-human-confirmation for exact paper]
2. Broader motivation: impose structure on the relation-to-head map. [repo-confirmed]
3. Hypothesis tested: nearby relation conditions should generate nearby heads. [repo-confirmed]
4. Changed part: `latest_aux_loss` is collected by the learner and added to TD loss; smooth variant supplies the auxiliary loss. [repo-confirmed]
5. Outcome: implemented; final result unknown. [needs-human-confirmation]
6. Status: active mechanism/visualization support. [repo-confirmed]
7. Story contribution: turns visualization hypothesis into a trainable regularizer. [inferred]

## 3. Failed Ideas and Negative Evidence

### Full generated structured maker

- Inspiration: RPG structured maker plus full hypernetwork generation. [repo-confirmed]
- Why promising: closest to a strong "relation pattern generates decision maker" claim. [repo-confirmed] [inferred]
- What failed: training cost became too high; user reported around two days for `rpg_full_structured_hypercond`. [conversation-derived]
- Failure type: cost/over-parameterization. [conversation-derived] [inferred]
- Lesson: dynamic generation should be focused and lightweight. [inferred]
- Shaped final model: motivated readout and then linear interaction generation. [repo-confirmed]

### Readout-only interaction head

- Inspiration: partial generated readout after fixed interaction encoder. [repo-confirmed]
- Why promising: cheaper than full interaction hypernetwork. [repo-confirmed]
- What failed: it did not match the user's intended ablation of making the interaction branch only one linear layer. [conversation-derived]
- Failure type: weak ablation alignment. [conversation-derived]
- Lesson: ablations must match the claim and fixed-control architecture. [inferred]
- Shaped final model: led to `rpg_linear_interaction_hypercond` and `rpg_fixed_linear_structured_maker`. [repo-confirmed]

### Graph/GAT local variants

- Inspiration: graph/entity relation MARL, GAT/GNN reasoning. [repo-confirmed] [needs-human-confirmation for exact papers]
- Why promising: graph explicitly represents relations among self, allies, and enemies. [inferred]
- What failed: user observed runtime grew too much; graph was not as lightweight or clean as hoped. [conversation-derived]
- Failure type: cost and possible mismatch with fixed-slot local observations. [conversation-derived]
- Lesson: explicit graph relation extraction is not automatically better than semantically split attention. [inferred]
- Shaped final model: kept RPG-style self-centered attention as main relation extractor. [repo-confirmed] [inferred]

### Global graph variants

- Inspiration: compute whole graph once per timestep to avoid repeated local graph overhead. [conversation-derived]
- Why promising: could reduce computation and enable graph collaboration between ally/enemy graphs. [conversation-derived]
- What failed: changed execution setting to CTCE; code labels these variants as CTCE validation. [repo-confirmed]
- Failure type: mismatch with CTDE/decentralized execution assumption. [repo-confirmed]
- Lesson: efficiency improvements must preserve the experimental setting. [inferred]
- Shaped final model: global graph kept as upper-bound/exploratory, not central. [repo-confirmed]

### Fixed structured controls performing strongly

- Inspiration: fair ablation design. [repo-confirmed]
- Why promising: necessary to isolate dynamic generation. [repo-confirmed]
- What failed: not a model failure, but a narrative failure for the broad claim "hypernetwork improves performance." [conversation-derived]
- Failure type: overbroad hypothesis and easy-map saturation. [inferred]
- Lesson: the claim must focus on hard/relation-sensitive maps and sample efficiency. [repo-confirmed] [inferred]
- Shaped final model: led to sharper dynamic-vs-fixed comparisons on `corridor`, `MMM2`, and asymmetric maps. [repo-confirmed]

### Relation mixer gate

- Inspiration: relation-aware credit weighting or attention/gating. [inferred]
- Why promising: relation condition could influence centralized training credit assignment. [inferred]
- What failed: user found the logic less convincing than local head adaptation. [conversation-derived]
- Failure type: unclear story and possibly weak ablation. [conversation-derived] [needs-human-confirmation for results]
- Lesson: prioritize mechanisms with a direct causal path from relation pattern to action Q-values. [inferred]
- Shaped final model: local interaction head adaptation remains central. [inferred]

### Generic hypernetwork baselines

- Inspiration: HyperNetworks/HyperMARL-style generated heads. [repo-confirmed]
- Why promising: dynamic parameter generation may personalize or adapt policies. [inferred]
- What failed or became insufficient: generic local/identity conditioning does not explain relation-specific interaction decisions. [inferred]
- Failure type: story too broad. [inferred]
- Lesson: relation pattern should be the condition, not merely agent id or raw observation. [inferred]
- Shaped final model: moved toward RPG-inspired relation extraction and structured interaction scoring. [repo-confirmed]

### GPU/high-utilization training push

- Inspiration: practical need to use GPU servers after CPU servers became unavailable. [conversation-derived]
- Why promising: larger batch and AMP could speed training. [repo-confirmed] [conversation-derived]
- What failed: OOMs, AMP mask overflow, CPU memory bottlenecks, and SC2 sampling bottleneck. [repo-confirmed for AMP fix and configs] [conversation-derived]
- Failure type: infrastructure/resource mismatch. [conversation-derived]
- Lesson: SMAC sampling remains CPU-bound; GPU utilization is not the only bottleneck. [conversation-derived]
- Shaped final model: safer V100 configs, AMP mask fix, and low-memory command templates. [repo-confirmed]

### Visualization first attempt

- Inspiration: need to show relation-to-head mapping. [conversation-derived]
- Why promising: similarity heatmaps and alignment plots can show whether relation patterns map to generated heads. [repo-confirmed]
- What failed: user found the first battle trace and heatmap hard to understand and not timestep-oriented enough. [conversation-derived]
- Failure type: visualization interpretability. [conversation-derived]
- Lesson: visualization must be per-timestep, agent-specific, and explain axes/colors clearly. [conversation-derived]
- Shaped final model/material: added relation/head dynamics videos and clearer battle trace intent visualization. [repo-confirmed]

## 4. Missing Literature Questions for Human Author

- Which exact HyperNetworks paper should be cited for dynamic parameter generation? [needs-human-confirmation]
- Which exact HyperMARL paper/version inspired `hypermarl_id` and `hypermarl_fullnet`? [needs-human-confirmation]
- Did we discuss QMIX mixer hypernetworks as a motivation for generated local heads, or only as background? [needs-human-confirmation]
- Which graph MARL papers inspired GCN/GAT/entity-relation attempts? [needs-human-confirmation]
- Did we discuss DGN, G2ANet, TarMAC, MAGIC, UPDeT, Entity Attention, GraphMIX, or coordination-graph methods? [needs-human-confirmation]
- Which paper motivated the heterogeneous graph design with node/edge type embeddings? [needs-human-confirmation]
- Which FiLM or conditional modulation paper should be cited for `rpg_film_interaction_hypercond`? [needs-human-confirmation]
- Was FiLM inspired by visual reasoning, conditional normalization, adapters, personalization, or another domain? [needs-human-confirmation]
- Which MoE/routing/expert-selection paper motivated `rpg_moe_interaction_head`? [needs-human-confirmation]
- Was `rpg_moe_interaction_head` inspired by sparse MoE, routing networks, options, role policies, or codebook routing? [needs-human-confirmation]
- Which residual/adaptation paper motivated `rpg_residual_interaction_hypercond`, if any? [needs-human-confirmation]
- Which smoothness, metric-learning, manifold regularization, contrastive learning, or personalization paper motivated `rpg_smooth_linear_interaction_hypercond`? [needs-human-confirmation]
- Was smoothness primarily your own visualization-driven hypothesis or a literature-driven regularizer? [needs-human-confirmation]
- Which attention/gating or value-factorization paper motivated `clean_relation_mixer_gate`? [needs-human-confirmation]
- Which paper or advisor comment motivated the "competitive on easy maps, stronger on hard maps" narrative? [needs-human-confirmation]
- Did an advisor encourage or discourage human-player-inspired language? [needs-human-confirmation]
- Which papers were used only as writing/narrative examples rather than technical ancestors? [needs-human-confirmation]
- Which failed variants have W&B runs or logs that can support efficiency/stability claims? [needs-human-confirmation]

## 5. Revised Related Work Structure Based on Full Literature Lineage

### 5.1 CTDE value factorization as background

- Why it belongs: the method is built on a QMIX-style CTDE learner. [repo-confirmed]
- How it shaped our model: local per-agent Q-values are mixed centrally during training. [repo-confirmed]
- How our work differs: we do not primarily change the mixer; we change local relation-conditioned decision heads. [repo-confirmed]
- Placement: short background paragraph in main related work. [inferred]

### 5.2 Grouping, roles, and structured cooperation

- Why it belongs: repository lineage comes from GoMARL, and the project started from cooperative structure. [repo-confirmed]
- How it shaped our model: motivated thinking about structure, but final method moves from group assignment to relation-conditioned local decision adaptation. [inferred]
- How our work differs: no explicit learned group assignment is central to the clean-hyper models. [repo-confirmed]
- Placement: main text if contrasting with GoMARL; otherwise concise. [inferred]

### 5.3 Entity-centric and graph-based MARL

- Why it belongs: the project uses self/ally/enemy entity semantics and implemented graph/GAT variants. [repo-confirmed]
- How it shaped our model: graph/entity ideas motivated relation representation and local relational reasoning. [repo-confirmed] [inferred]
- How our work differs: final focus is not just relation encoding; relation patterns condition or generate decision heads. [repo-confirmed]
- Placement: main text because reviewers will expect comparison to entity/graph MARL; graph variant details can go appendix. [inferred]

### 5.4 Hypernetworks and dynamic parameter generation

- Why it belongs: dynamic head generation is central. [repo-confirmed]
- How it shaped our model: local heads and interaction scorers are generated from condition vectors. [repo-confirmed]
- How our work differs: condition is online local relation pattern, not only task embedding, global context, or agent identity. [repo-confirmed] [inferred]
- Placement: main text; closest to core contribution. [inferred]

### 5.5 RPG and relation-pattern-to-decision methods

- Why it belongs: RPG is the closest named conceptual ancestor for relation pattern and structured maker. [repo-confirmed] [conversation-derived]
- How it shaped our model: self/ally/enemy relation capture, temporal relation hidden, and ego/interaction split. [repo-confirmed]
- How our work differs: single-task QMIX/SMAC setting, no RPG continual-learning regularizers or task embeddings, and emphasis on lightweight interaction-head generation plus fixed controls. [repo-confirmed] [inferred]
- Placement: main text as closest technical ancestor, but avoid overemphasizing it to the exclusion of hypernetwork/graph/modulation inspirations. [inferred]

### 5.6 Conditional modulation, residual adapters, MoE, and smoothness

- Why it belongs: these ideas motivate the improvement variants. [repo-confirmed]
- How it shaped our model: FiLM modulation, residual correction, expert gating, and smoothness regularization are all ways to constrain relation-conditioned adaptation. [repo-confirmed]
- How our work differs: these mechanisms are applied specifically to relation-conditioned interaction-action scoring in CTDE MARL. [repo-confirmed] [inferred]
- Placement: main text if variants become important results; otherwise compact paragraph plus appendix details. [inferred]

### 5.7 Visualization and interpretability of dynamic heads

- Why it belongs: relation-head alignment is central to explaining the mechanism. [repo-confirmed]
- How it shaped our model: smoothness regularizer and relation/head dynamics visualization were designed around the hypothesis that relation-similar agents should get similar heads. [repo-confirmed] [conversation-derived]
- How our work differs: visualization is not generic attention visualization; it targets the mapping from relation condition to generated decision parameters. [repo-confirmed]
- Placement: method/analysis section rather than standard related work, unless citing interpretability or representation-alignment papers. [inferred] [needs-human-confirmation for citations]

### 5.8 Human-player-inspired structural reasoning

- Why it belongs: useful motivation for self/ally/enemy and target-specific decomposition. [conversation-derived]
- How it shaped our model: supports manually structured relation extraction and interaction-action scoring. [conversation-derived] [repo-confirmed]
- How our work differs: no imitation learning or demonstrations. [repo-confirmed] [inferred]
- Placement: motivation/introduction only, not a major related-work category unless backed by specific papers. [inferred]

