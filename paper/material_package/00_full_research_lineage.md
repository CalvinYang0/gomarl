# Full Research Lineage Reconstruction

This document is a research-memory reconstruction for ChatGPT and the human author. It is not a paper draft.

Evidence tags:

- `[repo-confirmed]`: supported by repository files, configs, comments, code, README, or existing `paper/material_package` files.
- `[conversation-derived]`: reconstructed from the visible Codex/user conversation history or summarized prior discussion.
- `[inferred]`: logically inferred from model names, code structure, experiment setup, comments, or design changes.
- `[needs-human-confirmation]`: important but not directly verifiable from accessible sources.

Access boundary:

- I can access the current repository and the visible conversation context in this thread, including the summarized prior Codex conversation. [repo-confirmed] [conversation-derived]
- I cannot access private advisor meetings, external ChatGPT chats, W&B raw tables, or prior accounts unless their contents were pasted into this conversation or recorded in the repository. [needs-human-confirmation]
- Therefore, exact paper lists, advisor feedback, and some failed-result interpretations should be confirmed by the human author before being used in a manuscript. [needs-human-confirmation]

Primary local evidence anchors:

- `README.md`: original GoMARL repository framing and example SMAC maps. [repo-confirmed]
- `src/config/algs/clean_hyper.yaml`: model variant comments and clean-hyper configuration. [repo-confirmed]
- `src/modules/agents/clean_hyper_agent.py`: implementation of relation capturer, dynamic/fixed heads, graph variants, and diagnostics. [repo-confirmed]
- `src/learners/clean_learner.py`: QMIX/VDN learner, optional relation mixer gate, AMP support, and auxiliary losses. [repo-confirmed]
- `src/utils/battle_trace.py`: battle trace, relation/head similarity, alignment, and dynamics visualization. [repo-confirmed]
- `paper/material_package/*.md`: current paper context, method inventory, experiment plan, visualization plan, prompts, run commands, and AI writing workflow. [repo-confirmed]

## 1. The Broad Research Question Before the Model Existed

The broad question was not simply how to improve the GoMARL codebase. The deeper question was how cooperative MARL agents should reason about local ally-enemy relations when selecting actions under CTDE. [conversation-derived]

The working problem became: should local relation patterns merely be encoded as input features, or should they change the local action-value decision function itself? [conversation-derived] [repo-confirmed: `paper/material_package/01_chatgpt_context.md`]

The project gradually focused on whether a local Q-head can be observation-adaptive rather than a single fixed function shared across all relational regimes. [repo-confirmed: `paper/material_package/01_chatgpt_context.md`]

This is nontrivial because a standard recurrent agent already receives observations and can, in principle, output different Q-values for different states. The challenge is to justify why changing the decision function through relation-conditioned parameters is different from simply adding more features. [conversation-derived]

The current answer is: a fixed head can represent different outputs, but it must encode all local decision rules inside one shared mapping. A relation-conditioned head instead represents a family of local decision functions indexed by the current relation pattern. [repo-confirmed: `paper/material_package/01_chatgpt_context.md`]

The hard part is proving that this family-of-decision-functions view adds value beyond relation-aware feature encoding. That is why the fixed relation-conditioned structured maker became a central control. [repo-confirmed: `paper/material_package/03_experiment_plan.md`]

The original GoMARL angle contributed a group-level coordination starting point: GoMARL learns automatic grouping for cooperative MARL, and the repository README frames it as promoting intra- and inter-group coordination. [repo-confirmed: `README.md`]

The new research line moved from group-level coordination to relation-level decision adaptation: rather than asking which agents belong in a group, the model asks what local relation pattern an agent is facing and how that pattern should shape its action-value computation. [inferred]

## 2. Literature and Idea Sources We Discussed

### GoMARL: Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning

- Paper/method name: GoMARL, "Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning", NeurIPS 2023. [repo-confirmed: `README.md`]
- Idea contributed: automatic grouping as a structural approach to efficient cooperation. [repo-confirmed: `README.md`]
- Influence on our thinking: the project began inside the GoMARL codebase, and the earlier framing was about group-level structure. [repo-confirmed] [inferred]
- Final status: not the final technical contribution, but it provides repository lineage and a contrast between group-level coordination and relation-conditioned local decision adaptation. [inferred]
- Paper distinction: do not claim the current model is GoMARL unless the group algorithm is actually used. The current clean-hyper experiments use the clean hypernetwork family and QMIX-style learner, not the original group learner. [repo-confirmed: `src/config/algs/clean_hyper.yaml`, `src/learners/clean_learner.py`]

### QMIX, VDN, and CTDE value factorization

- Paper/method name: QMIX/VDN-style CTDE value factorization. [repo-confirmed: `src/learners/clean_learner.py`]
- Idea contributed: decentralized agent Q-values are trained with centralized value mixing. [repo-confirmed: `src/learners/clean_learner.py`]
- Influence on our thinking: the model keeps a standard recurrent per-agent policy and modifies the local Q-head, while QMIX remains the centralized training mixer. [repo-confirmed: `paper/material_package/01_chatgpt_context.md`]
- Final status: central experimental backbone. [repo-confirmed]
- Paper distinction: the contribution is not a new mixer by default. The main contribution is relation-conditioned local decision heads under a QMIX-style CTDE learner. [repo-confirmed]

### RPG: From General Relation Patterns to Task-Specific Decision-Making in Continual Multi-Agent Coordination

- Paper/method name: RPG, named in conversation as "From General Relation Patterns to Task-Specific Decision-Making in Continual Multi-Agent Coordination." [conversation-derived]
- Idea contributed: self/ally/other relation capture, relation pattern, temporal relation hidden state, and structured decision-maker split. [repo-confirmed: comments in `src/modules/agents/clean_hyper_agent.py` and `src/config/algs/clean_hyper.yaml`] [conversation-derived]
- Influence on our thinking: RPG motivated splitting observation semantics and separating ego-action values from interaction-action values. [repo-confirmed: `src/config/algs/clean_hyper.yaml`]
- Final status: strong inspiration, not a reproduction. The code comments explicitly say the relation capturer is an RPG-inspired single-task adaptation and does not reproduce continual-learning regularizers, task embedding, or some original RPG mechanisms. [repo-confirmed: `src/modules/agents/clean_hyper_agent.py`]
- Paper distinction: avoid implying exact RPG reproduction. Say "RPG-inspired relation pattern and structured decision maker" unless exact paper details are verified. [repo-confirmed] [needs-human-confirmation for exact RPG citation]

### Hypernetworks and HyperMARL-style parameter generation

- Paper/method name: hypernetworks; HyperMARL-style learned agent embedding to generate full post-RNN action network. [repo-confirmed: `src/config/algs/clean_hyper.yaml`]
- Idea contributed: generate action-head parameters from a condition vector rather than using one fixed head. [repo-confirmed]
- Influence on our thinking: the early clean-hyper family explored id-conditioned and local-observation-conditioned generated heads. [repo-confirmed]
- Final status: dynamic parameter generation remains central, but the best paper story should focus on relation-conditioned decision functions rather than "hypernetwork is always better." [repo-confirmed: `paper/material_package/README.md`]
- Paper distinction: do not claim general hypernetwork superiority; compare against fixed relation-conditioned heads. [repo-confirmed]

### Role-based or grouping-based MARL

- Paper/method name: role/grouping-based MARL, including GoMARL-style automatic grouping. [repo-confirmed for GoMARL; broader role-based literature needs confirmation]
- Idea contributed: structural organization can reduce coordination complexity. [inferred]
- Influence on our thinking: relation pattern adaptation can be framed as a finer-grained alternative to static or dynamic grouping. [inferred]
- Final status: useful related-work category, but not the final mechanism. [inferred]
- Paper distinction: our method does not learn explicit groups as the main output; it conditions decision functions from local relation patterns. [repo-confirmed]

### Entity-centric and graph/attention-based MARL

- Paper/method name: entity-centric attention, GCN/GAT, heterogeneous graph reasoning. [repo-confirmed through implemented variants; exact literature list needs confirmation]
- Idea contributed: agents and entities can be represented as nodes/tokens, with attention or graph message passing capturing local structure. [repo-confirmed: graph/GAT variants in `src/modules/agents/clean_hyper_agent.py`]
- Influence on our thinking: the project tested graph alternatives, including two-graph GAT, heterogeneous GAT, and global CTCE graph variants. [repo-confirmed]
- Final status: implemented as controls/alternatives but not currently the main story due to overhead and conceptual mismatch with the desired local relation adaptation. [repo-confirmed] [conversation-derived]
- Paper distinction: graph variants should be described as explored alternatives unless they become main experiments. Do not present them as final contribution if they are not used in final results. [inferred]

### FiLM

- Paper/method name: FiLM-style feature-wise linear modulation. [repo-confirmed: `rpg_film_interaction_hypercond` implementation]
- Idea contributed: relation condition modulates interaction features through scale and bias instead of directly generating a full scorer. [repo-confirmed]
- Influence on our thinking: dynamic adaptation might be more stable when constrained to modulation around a fixed encoder. [repo-confirmed: `paper/material_package/03_experiment_plan.md`]
- Final status: implemented improvement variant. [repo-confirmed]
- Paper distinction: unless FiLM-specific literature is cited and verified, describe as "FiLM-style modulation" rather than claiming novelty in FiLM itself. [needs-human-confirmation]

### Mixture-of-experts

- Paper/method name: mixture-of-experts style soft expert selection. [repo-confirmed: `rpg_moe_interaction_head`]
- Idea contributed: relation pattern may choose among several fixed interaction regimes rather than generating arbitrary parameters. [repo-confirmed] [inferred]
- Influence on our thinking: relation patterns might define soft decision regimes. [repo-confirmed: `paper/material_package/02_method_and_code_inventory.md`]
- Final status: implemented improvement variant. [repo-confirmed]
- Paper distinction: do not claim a new MoE method; present it as an ablation/variant for constrained relation-conditioned adaptation. [inferred]

### Residual dynamic heads

- Paper/method name: residual correction principle, not necessarily tied to one paper in the accessible record. [repo-confirmed: `rpg_residual_interaction_hypercond`]
- Idea contributed: keep a robust fixed scorer and let the generated branch apply relation-specific corrections. [repo-confirmed]
- Influence on our thinking: this addresses the concern that full dynamic generation may be unstable or unnecessary. [repo-confirmed: `paper/material_package/03_experiment_plan.md`]
- Final status: implemented improvement variant. [repo-confirmed]
- Paper distinction: present as a design variant motivated by stability and fixed-control competitiveness, not as a proven final best model unless experiments support it. [inferred]

### Smoothness regularization

- Paper/method name: relation-head smoothness regularization; exact external inspiration is not recorded. [repo-confirmed]
- Idea contributed: similar relation patterns should generate similar MLP head parameters. [repo-confirmed: `rpg_smooth_linear_interaction_hypercond`]
- Influence on our thinking: directly supports the visualization thesis that the model learns a meaningful mapping from relation space to decision-function space. [repo-confirmed: `paper/material_package/04_visualization_plan.md`]
- Final status: implemented improvement variant and current requested experiment target. [repo-confirmed] [conversation-derived]
- Paper distinction: this is a hypothesis and regularizer; performance benefit requires results. [repo-confirmed]

### Relation-conditioned mixer gates

- Paper/method name: relation-conditioned mixer gate. [repo-confirmed: `clean_relation_mixer_gate`]
- Idea contributed: use relation pattern to reweight agent contributions before QMIX mixing during training. [repo-confirmed]
- Influence on our thinking: explored whether relation patterns should affect credit assignment, not only local heads. [conversation-derived] [repo-confirmed]
- Final status: implemented but conceptually weaker than head adaptation; the user explicitly questioned why relation-conditioned mixer is logically necessary. [conversation-derived]
- Paper distinction: likely keep as exploratory or future work unless strong experiments and a clean argument emerge. [inferred]

### Auxiliary relation losses

- Paper/method name: current implemented auxiliary loss is the smooth head loss. [repo-confirmed: `src/learners/clean_learner.py` and `rpg_smooth_linear_interaction_hypercond`]
- Idea contributed: impose structure on relation-to-parameter mapping. [repo-confirmed]
- Influence on our thinking: connects training to visualization and interpretability. [repo-confirmed]
- Final status: implemented for smooth variant. [repo-confirmed]
- Paper distinction: do not describe unrelated auxiliary relation losses unless implemented or tested. [needs-human-confirmation]

### Human-player-inspired reasoning

- Paper/method name: human-player-inspired reasoning, not imitation learning. [conversation-derived]
- Idea contributed: human players often separately reason about self status, ally support, enemy threats, and target-specific actions. [conversation-derived]
- Influence on our thinking: helped motivate manual semantic decomposition and interaction-action scoring. [conversation-derived]
- Final status: narrative support only, not a technical source of supervision. [repo-confirmed by absence of human demo loading] [inferred]
- Paper distinction: do not claim human mimicry or use of demonstrations. Say "inspired by human-like relation-aware reasoning" only if worded carefully. [conversation-derived]

### CTEM: Self-Supervised Multi-Agent Diversity with Nonparametric Entropy Maximization

- Paper/method name: CTEM, discussed by the user when comparing `3s5z_vs_3s6z` and `5m6m` performance. [conversation-derived]
- Idea contributed: recent benchmark comparison and a possible reference point for SOTA-style numbers. [conversation-derived]
- Influence on our thinking: highlighted that our method may perform well on `3s5z_vs_3s6z` but lag on `5m6m`, motivating map-specific interpretation. [conversation-derived]
- Final status: related empirical benchmark, not a direct technical ancestor based on accessible evidence. [conversation-derived]
- Paper distinction: verify exact reported numbers and whether CTEM uses extra priors before citing. [needs-human-confirmation]

## 3. Model Genealogy: Every Important Variant and Its Story

### `qmix_minimal`

- Motivation: create a clean minimal QMIX-style baseline with shared recurrent trunk and fixed MLP head. [repo-confirmed]
- Hypothesis tested: relation-conditioned/hypernetwork variants should be compared against a fair fixed-head baseline. [conversation-derived] [repo-confirmed]
- Inspiration: QMIX-style recurrent agent plus fixed two-layer head. [repo-confirmed]
- Changed part: no hypernetwork; fixed two-layer MLP after GRU maps `h_t` to Q-values. [repo-confirmed: `src/modules/agents/clean_hyper_agent.py`]
- Expected to solve: provide a fair baseline after the user noticed earlier baseline comparison was unfair because a fixed baseline had fewer layers. [conversation-derived]
- Experimental outcome: earlier screenshots suggested minimal QMIX was much slower on `3s5z`-type tasks than relation-pattern variants. [conversation-derived]
- Status: kept as baseline. [repo-confirmed]
- Lesson: baseline fairness matters; if the baseline is underpowered, any dynamic-head improvement is not credible. [conversation-derived]
- Paper description: "minimal QMIX-style recurrent agent with fixed two-layer Q head." [repo-confirmed]

### `baseline`

- Motivation: initial clean hypernetwork baseline using local condition from observation, previous action, and hidden state. [repo-confirmed: `src/config/algs/clean_hyper.yaml`]
- Hypothesis tested: a dynamic head conditioned on local information may help even without relation-specific structure. [inferred]
- Inspiration: generic hypernetwork-conditioned Q head. [repo-confirmed]
- Changed part: local condition encoder produces condition for generated unified head. [repo-confirmed]
- Outcome: not the current main paper focus. [inferred]
- Lesson: dynamic generation alone is too broad; the paper needs relation-specific structure and strong fixed controls. [inferred]
- Paper description: optional baseline if included; avoid confusing it with `qmix_minimal`. [repo-confirmed]

### `hypermarl_id`

- Motivation: test agent-id-conditioned dynamic head. [repo-confirmed]
- Hypothesis: learned agent identity embeddings could specialize action heads. [repo-confirmed] [inferred]
- Inspiration: HyperMARL-style id conditioning. [repo-confirmed]
- Changed part: learned embedding `e_i` provides condition vector for generated head. [repo-confirmed]
- Outcome: not central in current plan. [inferred]
- Lesson: agent identity conditioning is less aligned with the paper's relation-adaptive hypothesis than observation relation conditioning. [inferred]
- Paper description: if mentioned, use as a background hypernetwork baseline, not central contribution. [inferred]

### `hypermarl_fullnet`

- Motivation: closer HyperMARL-style full post-RNN action network parameter generation from learned agent embeddings. [repo-confirmed]
- Hypothesis: full dynamic head generation can specialize action networks by agent identity. [repo-confirmed] [inferred]
- Changed part: `MLPHyperParameterGenerator` generates two-layer post-RNN action network parameters. [repo-confirmed]
- Outcome: not current main line. [inferred]
- Lesson: full hypernetwork generation exists as a reference point, but relation-conditioned generation is the more relevant direction. [inferred]
- Paper description: use only if comparing to id-conditioned hypernetwork baselines. [needs-human-confirmation]

### `dynamic_route`

- Motivation: test whether local conditions should select a discrete/soft route or codebook rather than directly generate parameters. [repo-confirmed]
- Hypothesis: route/codebook selection may stabilize dynamic heads by limiting condition space. [inferred]
- Changed part: local route logits select route codebook embedding before head generation. [repo-confirmed]
- Outcome: not current main line. [inferred]
- Lesson: routing is conceptually related to MoE, but the final story moved to relation-conditioned interaction heads. [inferred]
- Paper description: likely omit unless route experiments are used. [needs-human-confirmation]

### `local_structured_hypercond`

- Motivation: isolate the structured maker from the RPG relation capturer. [repo-confirmed]
- Hypothesis: self/interaction action decomposition may help even when condition source is local obs/prev_action/hidden rather than RPG relation pattern. [repo-confirmed] [inferred]
- Changed part: keeps local baseline condition source but replaces unified generated head with ego-action/enemy-interaction split. [repo-confirmed]
- Outcome: an early run failed before this model type existed or was registered, producing `Unknown clean_model_type=local_structured_hypercond`; later code includes it. [conversation-derived] [repo-confirmed current support]
- Lesson: separating condition source from decision-maker structure is necessary for ablations. [inferred]
- Paper description: a structure-only control if results exist. [needs-human-confirmation]

### `rpg_relation_hypercond`

- Motivation: use RPG-inspired relation pattern as condition for generated head. [repo-confirmed]
- Hypothesis: self/ally/enemy relation patterns provide a better condition than raw local features. [repo-confirmed] [conversation-derived]
- Inspiration: RPG relation capturer. [repo-confirmed]
- Changed part: adds observation split, self-centered cross-attention to allies/enemies, relation GRU, and output encoder. [repo-confirmed]
- Outcome: earlier results suggested relation-pattern versions improved convergence over minimal QMIX on `3s5z`-type tasks. [conversation-derived]
- Lesson: relation pattern extraction is valuable but does not by itself isolate the value of structured action decomposition. [inferred]
- Paper description: relation-condition baseline before structured maker. [inferred]

### `rpg_relation_route`

- Motivation: combine RPG relation pattern with route/codebook selection. [repo-confirmed]
- Hypothesis: relation pattern may select a discrete coordination regime. [repo-confirmed] [inferred]
- Changed part: relation condition is converted to route logits and route codebook condition. [repo-confirmed]
- Outcome: not current main line. [inferred]
- Lesson: useful conceptual bridge to regime-based adaptation, but less directly tied to generated interaction-head story. [inferred]
- Paper description: omit or discuss as exploratory unless experiments matter. [needs-human-confirmation]

### `rpg_structured_hypercond`

- Motivation: add RPG-inspired structured decision maker on top of relation pattern. [repo-confirmed]
- Hypothesis: self-actions and enemy-interaction actions should be scored differently. [repo-confirmed] [conversation-derived]
- Inspiration: RPG maker split. [repo-confirmed]
- Changed part: ego branch is condition-generated; interaction branch is shared scorer conditioned on relation pattern and enemy tokens. [repo-confirmed]
- Outcome: became an important early strong version, including names like `3s5z_rpg_structured_hypercond_ctde_s1`. [conversation-derived]
- Lesson: structured maker improved interpretability and possibly performance, but the initial interaction branch did not fully generate parameters, leading to fairness and RPG-comparison concerns. [conversation-derived] [repo-confirmed]
- Paper description: early structured version; distinguish it from full hypernetwork maker. [repo-confirmed]

### `rpg_full_structured_hypercond`

- Motivation: correct the structured maker so both ego and enemy-interaction branches are generated from relation pattern. [repo-confirmed]
- Hypothesis: a closer RPG-like hypernetwork decision-maker split may be stronger. [repo-confirmed] [conversation-derived]
- Changed part: generates both interaction bottleneck and output parameters for enemy interaction scoring. [repo-confirmed]
- Expected to solve: fairness issue that enemy scoring in earlier structured version was not generated by hypernetwork. [conversation-derived]
- Outcome: training became extremely expensive, with user reporting roughly 2 days compared with around 16 hours for previous versions. [conversation-derived]
- Status: modified/superseded because cost was too high. [conversation-derived] [repo-confirmed by later lighter variants]
- Lesson: full parameter generation for interaction actions is over-expensive and may be unnecessary. Focus dynamic adaptation on the most useful low-cost part. [conversation-derived] [inferred]
- Paper description: use as motivation for lightweight interaction-head generation only if failed-cost evidence is documented. [needs-human-confirmation]

### `rpg_readout_structured_hypercond`

- Motivation: reduce cost of the full structured maker by using a fixed interaction encoder and generating only the final readout. [repo-confirmed]
- Hypothesis: generating only the final layer may retain adaptivity while reducing compute. [repo-confirmed] [inferred]
- Changed part: fixed encoder over `[hidden, enemy token]`; relation condition generates final readout layer. [repo-confirmed]
- Outcome: user later clarified that the intended comparison was interaction branch with only one linear layer, not necessarily the readout version. [conversation-derived]
- Status: intermediate variant, not current main. [repo-confirmed] [conversation-derived]
- Lesson: cost reduction must match the intended ablation and fixed control. [conversation-derived]
- Paper description: likely omit or place in appendix if not central. [needs-human-confirmation]

### `rpg_linear_interaction_hypercond`

- Motivation: keep dynamic relation-conditioned interaction scoring but remove the generated bottleneck layer. [repo-confirmed]
- Hypothesis: a one-layer relation-generated interaction scorer may be enough and much cheaper. [repo-confirmed] [conversation-derived]
- Changed part: relation condition generates one linear scorer over `[agent hidden, enemy token]`; ego branch remains generated. [repo-confirmed]
- Expected to solve: reduce the high cost of full interaction hypernetwork while preserving dynamic interaction adaptation. [repo-confirmed] [conversation-derived]
- Outcome: preliminary results show strong performance on `corridor`, fast convergence on `MMM2`, and competitive/high performance on `5m6m`, but evidence is mostly single-seed screenshots. [repo-confirmed: material package] [conversation-derived]
- Status: current main dynamic-head model. [repo-confirmed]
- Lesson: dynamic adaptation should focus on interaction actions where local relation changes matter most. [inferred]
- Paper description: main proposed lightweight dynamic interaction head. [repo-confirmed]

### `rpg_fixed_structured_maker`

- Motivation: control for relation pattern and structured maker without generated decision parameters. [repo-confirmed]
- Hypothesis: if fixed structured maker performs well, then relation decomposition rather than hypernetwork generation may be the key. [conversation-derived] [repo-confirmed]
- Changed part: fixed ego and interaction MLPs conditioned by concatenating relation pattern. [repo-confirmed]
- Outcome: fixed variants sometimes performed surprisingly well, motivating a sharper fixed-vs-dynamic comparison. [conversation-derived]
- Status: important control, later refined to linear interaction version for fairness. [repo-confirmed] [conversation-derived]
- Lesson: the paper cannot simply compare dynamic to weak QMIX; it must prove dynamic generation adds beyond fixed relation conditioning. [conversation-derived]
- Paper description: fixed structured control. [repo-confirmed]

### `rpg_fixed_linear_structured_maker`

- Motivation: create a matched fixed control for `rpg_linear_interaction_hypercond`. [repo-confirmed]
- Hypothesis: if dynamic one-layer interaction head is useful, it should beat a fixed one-layer relation-conditioned interaction scorer. [repo-confirmed] [conversation-derived]
- Changed part: fixed two-layer ego branch and fixed one-layer interaction scorer using `[hidden, relation condition, enemy token]`. [repo-confirmed]
- Expected to solve: fairness concern that fixed and dynamic interaction branches had different depths. [conversation-derived]
- Outcome: on `5m6m`, fixed and dynamic both reached high win rates; on `corridor`, dynamic appeared much stronger in preliminary results; on `MMM2`, dynamic appeared faster but both solved. [repo-confirmed: material package] [conversation-derived]
- Status: central fixed control. [repo-confirmed]
- Lesson: easy maps may not show the value of dynamic heads; hard/relation-sensitive maps matter more. [repo-confirmed]
- Paper description: matched fixed relation-conditioned control. [repo-confirmed]

### `rpg_residual_interaction_hypercond`

- Motivation: dynamic generation may be useful as correction rather than replacement. [repo-confirmed] [conversation-derived]
- Hypothesis: a fixed interaction scorer can learn the default rule, while a gated generated residual handles relation-specific deviations. [repo-confirmed]
- Inspiration: residual learning/stability principle. [inferred]
- Changed part: computes fixed score plus sigmoid gate times dynamic generated score. [repo-confirmed]
- Expected to solve: instability/over-flexibility of generated heads and strong fixed baseline competitiveness. [repo-confirmed] [conversation-derived]
- Outcome: implemented and planned for corridor screening; final results not yet recorded in material package. [repo-confirmed] [needs-human-confirmation]
- Status: active improvement variant. [repo-confirmed]
- Paper description: constrained dynamic adaptation variant. [repo-confirmed]

### `rpg_film_interaction_hypercond`

- Motivation: use relation pattern to modulate features rather than generate scorer weights directly. [repo-confirmed]
- Hypothesis: FiLM-style modulation may stabilize adaptation and reduce parameter-generation burden. [repo-confirmed] [inferred]
- Changed part: fixed interaction encoder plus relation-generated gamma/beta and fixed scorer. [repo-confirmed]
- Outcome: implemented and planned for screening; final results not recorded here. [repo-confirmed] [needs-human-confirmation]
- Status: active improvement variant. [repo-confirmed]
- Paper description: modulation-based relation-conditioned interaction head. [repo-confirmed]

### `rpg_moe_interaction_head`

- Motivation: relation patterns may correspond to soft regimes rather than arbitrary continuous heads. [repo-confirmed] [inferred]
- Hypothesis: a soft mixture of fixed experts can capture different interaction modes with less instability. [repo-confirmed] [inferred]
- Changed part: relation condition gates several fixed interaction expert heads. [repo-confirmed]
- Outcome: implemented and planned for screening; final results not recorded here. [repo-confirmed] [needs-human-confirmation]
- Status: active improvement variant. [repo-confirmed]
- Paper description: regime-selection ablation for relation-conditioned interaction scoring. [inferred]

### `rpg_smooth_linear_interaction_hypercond`

- Motivation: the user proposed that relation-pattern-similar agents should generate similar MLP head parameters. [conversation-derived]
- Hypothesis: a smooth relation-to-head mapping improves interpretability and may improve generalization/stability. [repo-confirmed]
- Changed part: same generated one-layer interaction head as `rpg_linear_interaction_hypercond`, plus KNN smoothness regularizer over relation condition and generated head parameters. [repo-confirmed]
- Expected to solve: make the mapping `relation condition -> generated head` less arbitrary and more visualizable. [repo-confirmed] [conversation-derived]
- Outcome: current requested experiment includes running this model on `5m_vs_6m`; final result not yet available. [conversation-derived]
- Status: active improvement variant. [repo-confirmed]
- Paper description: smoothness-regularized dynamic head; useful for both mechanism and visualization. [repo-confirmed]

### `clean_relation_mixer_gate`

- Motivation: explore whether relation pattern should affect centralized credit assignment in QMIX. [repo-confirmed] [conversation-derived]
- Hypothesis: agents with different relation states may deserve different mixing weights. [inferred]
- Changed part: learner multiplies selected per-agent Q-values by a positive relation-pattern gate before QMIX mixing; target side mirrors the gate. [repo-confirmed]
- Outcome: user questioned the logic of relation-conditioned mixer and asked why it would make sense. [conversation-derived]
- Status: implemented but not a core final claim. [repo-confirmed] [inferred]
- Lesson: not every relation-conditioned module has a strong conceptual story. The head adaptation story is cleaner because relation patterns directly affect local action scoring. [conversation-derived] [inferred]
- Paper description: exploratory extension or appendix only unless strong evidence emerges. [inferred]

### `two_graph_gat_hypercond`

- Motivation: test whether explicit graph attention over self+allies and self+enemies can replace RPG cross-attention. [repo-confirmed] [conversation-derived]
- Hypothesis: GAT may use graph information more explicitly than token cross-attention. [repo-confirmed] [conversation-derived]
- Changed part: two local ego graphs, one self+allies and one self+enemies, generate relation condition. [repo-confirmed]
- Outcome: user observed graph versions were much more expensive, with training time inflating toward a day or more. [conversation-derived]
- Status: implemented but not current main. [repo-confirmed]
- Lesson: graph structure can add overhead without clearly solving the fixed-slot observation issue. [conversation-derived]
- Paper description: graph-control alternative, not final main model unless results justify. [inferred]

### `hetero_gat_hypercond`

- Motivation: test a typed heterogeneous graph where node/edge types encode ally/enemy/self relations. [repo-confirmed] [conversation-derived]
- Hypothesis: typed graph attention may better capture heterogeneous relation semantics. [repo-confirmed] [conversation-derived]
- Changed part: typed self-loop, ally-to-self, enemy-to-self messages with type-level attention. [repo-confirmed]
- Outcome: similar overhead concerns as two-graph GAT; not current main. [conversation-derived]
- Status: implemented exploratory variant. [repo-confirmed]
- Lesson: richer graph semantics can be conceptually appealing but may not be computationally acceptable. [conversation-derived]
- Paper description: optional related ablation if included. [needs-human-confirmation]

### `global_two_graph_gat_hypercond`

- Motivation: reduce repeated per-agent local graph computation by computing whole friendly/enemy graphs once per timestep. [conversation-derived] [repo-confirmed]
- Hypothesis: CTCE whole-graph mode may be a computational upper-bound or validation mode. [repo-confirmed]
- Changed part: computes friendly graph and enemy graph with cross-graph attention, then generates condition. [repo-confirmed]
- Outcome: categorized as CTCE validation mode, not decentralized execution. [repo-confirmed]
- Status: not a fair CTDE final model. [repo-confirmed]
- Lesson: global graph can reduce repeated computation but changes the execution assumption. [conversation-derived] [repo-confirmed]
- Paper description: do not compare as a main CTDE method unless clearly labelled CTCE upper bound. [repo-confirmed]

### `global_hetero_gat_hypercond`

- Motivation: whole-graph heterogeneous version of typed graph reasoning. [repo-confirmed]
- Hypothesis: one typed graph over friendly/enemy nodes may capture global relation structure. [repo-confirmed] [inferred]
- Changed part: global typed graph with node-type and edge-type embeddings. [repo-confirmed]
- Outcome: CTCE validation mode, not decentralized execution. [repo-confirmed]
- Status: exploratory/upper-bound variant. [repo-confirmed]
- Paper description: not a main CTDE contribution. [repo-confirmed]

### `graph_hypercond` and `graph_route`

- Motivation: generic obs-only graph + GCN condition or route condition. [repo-confirmed]
- Hypothesis: graph convolution over observations might provide a useful condition for hypernetwork or route selection. [repo-confirmed] [inferred]
- Changed part: builds graph from observation tokens and applies standard GCN. [repo-confirmed]
- Outcome: currently marked CTCE validation. [repo-confirmed]
- Status: exploratory. [repo-confirmed]
- Lesson: generic graph construction is less aligned with the self/ally/enemy relation narrative than RPG-inspired decomposition. [inferred]
- Paper description: omit unless used as negative/appendix result. [needs-human-confirmation]

## 4. The Evolution of the Main Hypothesis

Stage 1: group-level coordination may help. GoMARL repository origin framed cooperation through automatic grouping. [repo-confirmed]

Limitation: group-level coordination does not directly explain how a single agent should change attack-target or movement scoring under changing local ally/enemy relations. [inferred]

Stage 2: local observation structure matters. The project began focusing on how SMAC observations contain self, ally, and enemy semantics. [repo-confirmed] [conversation-derived]

Limitation: feeding all observation features into a generic head does not explicitly separate relation extraction from action scoring. [inferred]

Stage 3: self/ally/enemy decomposition may help. RPG-inspired relation capturer was adopted to encode self, ally, and enemy tokens separately. [repo-confirmed]

Limitation: relation representation alone could still be just another feature; it did not yet explain decision-function adaptation. [conversation-derived]

Stage 4: ego-action and interaction-action values should be separated. Structured maker variants split self-action Q-values from enemy-interaction Q-values. [repo-confirmed]

Limitation: early structured variants did not generate all relevant maker parameters, causing fairness and RPG-comparison concerns. [conversation-derived]

Stage 5: relation representation alone may be insufficient. The user repeatedly questioned whether a fixed network can already learn different rules from different observations. [conversation-derived]

Limitation: the paper needed a sharper claim than "more features help." [conversation-derived]

Stage 6: relation patterns should condition the decision function. The main hypothesis became that relation patterns should generate or modulate the local Q-head, especially for interaction actions. [repo-confirmed] [conversation-derived]

Limitation: full dynamic interaction generation was too expensive and possibly unnecessary. [conversation-derived]

Stage 7: full dynamic heads may be unstable or unnecessary. `rpg_full_structured_hypercond` reportedly took around two days, far more than lighter versions. [conversation-derived]

Limitation: compute cost made full maker generation impractical as the main method. [conversation-derived]

Stage 8: dynamic adaptation should focus on interaction actions. `rpg_linear_interaction_hypercond` generates a one-layer interaction scorer and became the main lightweight version. [repo-confirmed]

Limitation: fixed linear structured maker can also perform well, especially on easy maps, so stronger maps and ablations are needed. [repo-confirmed] [conversation-derived]

Stage 9: hard maps reveal the benefit more clearly than easy maps. Preliminary observations suggested `corridor` showed a clearer dynamic-vs-fixed gap than `5m6m`. [repo-confirmed: material package] [conversation-derived]

Limitation: single-seed results and screenshots are not enough for final paper claims. [repo-confirmed]

Stage 10: relation-to-head mapping should be visualized. The user proposed that relation-pattern-similar agents should receive similar generated MLP heads, leading to smoothness and visualization work. [conversation-derived] [repo-confirmed]

## 5. Why We Used Manual Decomposition and How to Defend It

The model manually uses SMAC observation semantics to split local observations into movement/self, ally, enemy, and own features. [repo-confirmed: `_build_rpg_obs_layout` and `_split_rpg_obs` in `src/modules/agents/clean_hyper_agent.py`]

The model manually decomposes actions into ego/self actions and interaction/attack actions by using the number of enemies to split the action dimension. [repo-confirmed: `rpg_n_ego_actions = n_actions - n_enemies`]

This decomposition is not learned automatically. [repo-confirmed]

This decomposition should not be described as "no prior knowledge." It is a structural inductive bias based on public environment semantics. [inferred]

It is also not the same as privileged expert knowledge if the information is already part of the public observation/action specification available to all methods. [inferred]

A defensible phrase is: "environment-semantics-guided structural inductive bias." [inferred]

The decomposition differs from automatically discovered structure because the model does not infer which observation slots correspond to allies or enemies from scratch. It uses known SMAC layout. [repo-confirmed] [inferred]

Reviewer criticism to expect:

- The method may be map/domain-specific because it relies on SMAC observation/action layout. [inferred]
- The method may gain from handcrafted structure rather than dynamic parameter generation. [inferred]
- The method may not transfer to environments without entity/action semantics. [inferred]

Defenses:

- Public environment semantics are routinely used in entity-centric MARL architectures; they are not hidden labels or expert demonstrations. [inferred] [needs-human-confirmation for citations]
- Fixed structured controls test whether decomposition alone explains the result. [repo-confirmed]
- `rpg_fixed_linear_structured_maker` is the key defense because it uses relation pattern and matched action decomposition but does not generate interaction-head parameters. [repo-confirmed]
- If dynamic variants beat fixed controls on relation-sensitive maps, the evidence supports dynamic decision-function adaptation beyond manual decomposition. [inferred]

## 6. Why the Final Model Looks the Way It Does

Self/ally/enemy encoders exist because raw SMAC observation contains semantically different entity groups, and treating them as homogeneous features hides the relation structure. [repo-confirmed] [inferred]

Self-centered attention to allies and enemies exists because each agent needs a first-person relation pattern: which allies and enemies matter from this agent's viewpoint. [repo-confirmed]

The instant relation pattern exists to combine self token, ally context, and enemy context into a compact current-timestep relation representation. [repo-confirmed]

The temporal relation GRU exists because relation context changes over time and the agent's recurrent policy state alone may not explicitly preserve relation-pattern dynamics. [repo-confirmed] [inferred]

The relation condition encoder exists to map the relation hidden state into the condition dimension used by hypernetworks or modulators. [repo-confirmed]

The structured ego-action branch exists because movement/no-op/stop style actions have different semantics from target-specific attacks. [repo-confirmed] [inferred]

The structured interaction-action branch exists because each attack action corresponds to a particular enemy slot, so scoring should depend on both agent hidden state and enemy token. [repo-confirmed]

The relation-conditioned generated/modulated interaction head exists because the main hypothesis is that local interaction scoring rules should change with relation pattern. [repo-confirmed] [conversation-derived]

The CTDE/QMIX learner remains because the research aims to modify the local decision head while keeping a standard cooperative value-factorization training backbone. [repo-confirmed]

The final model is not arbitrary. It is the result of moving from broad grouping, to relation extraction, to structured action decomposition, to lightweight relation-conditioned interaction scoring after full dynamic maker cost became unacceptable. [conversation-derived] [repo-confirmed]

## 7. Experimental Storyline and Map Difficulty

`3s5z` and `3s5z_vs_3s6z` were important early tests for relation-pattern and structured-maker variants. [conversation-derived]

Observed result: earlier screenshots/discussions suggested relation-pattern structured variants converged much faster than minimal QMIX on `3s5z`-type tasks, and the user reported being much higher than CTEM on `3s5z_vs_3s6z` while worse on `5m6m`. [conversation-derived]

Interpretation: asymmetric or heterogeneous combat may expose relation-adaptive benefits more clearly than easy symmetric maps. [inferred]

`5m6m` or `5m_vs_6m` is useful as a sanity/easy or moderately asymmetric map, but preliminary results showed fixed and dynamic variants both reach high win rate quickly. [repo-confirmed: material package] [conversation-derived]

Interpretation: if both models solve the map, final win rate is not a strong discriminator. Sample efficiency, stability, and learning curve AUC matter more. [repo-confirmed]

`corridor` became the strongest screening map because preliminary results showed a much clearer gap between dynamic linear interaction hypercondition and fixed linear structured maker. [repo-confirmed: material package]

Interpretation: `corridor` is likely more relation-sensitive or interaction-heavy, making it a better map for testing dynamic interaction heads. [inferred]

`MMM2` is hard enough to test convergence speed, but preliminary screenshots showed both fixed and dynamic versions eventually reached near-perfect performance. [repo-confirmed: material package]

Interpretation: convergence speed may matter more than final win rate on this map. [repo-confirmed]

`5z_vs_1ul` was discussed as a possible task in the available SMAC map range, but its support and relevance were not fully documented in the material package. [conversation-derived] [needs-human-confirmation]

Important metrics beyond final win rate:

- Steps to 50%, 80%, and 90% win rate. [repo-confirmed: material package]
- Area under the learning curve. [repo-confirmed]
- Wall-clock time and environment steps per second. [repo-confirmed]
- Test episode length. [repo-confirmed]
- Stability/drop frequency after convergence. [repo-confirmed]
- Relation-head alignment metrics. [repo-confirmed]

Easy maps may not show a clear advantage because fixed relation-conditioned heads may already be expressive enough. [repo-confirmed]

Being competitive on easy maps is still acceptable if the method has clear gains on hard/relation-sensitive maps and does not collapse elsewhere. [inferred]

Hard maps are more important for the core claim because the method is designed for changing local interaction regimes. [repo-confirmed] [inferred]

## 8. What the Failed Attempts Taught Us

### Full RPG structured hypercondition was too expensive

- Idea: generate both ego and interaction branch parameters from relation pattern. [repo-confirmed]
- Why reasonable: closer to the RPG-style hypernetwork maker split and addresses fairness concern. [repo-confirmed] [conversation-derived]
- What went wrong: user reported training cost around two days, much slower than previous versions. [conversation-derived]
- Failure type: over-parameterization and compute/memory burden. [conversation-derived] [inferred]
- Lesson: the final method should generate only the most necessary part, especially interaction scoring. [inferred]
- Current influence: motivated readout and linear interaction variants. [repo-confirmed]

### Readout-only variant did not match the intended ablation

- Idea: fixed interaction encoder plus generated final readout. [repo-confirmed]
- Why reasonable: cheaper than full generated interaction branch. [repo-confirmed]
- What went wrong: user clarified the intended variant was the interaction branch using only one linear layer, not just a generated readout after a fixed encoder. [conversation-derived]
- Failure type: ablation mismatch. [conversation-derived]
- Lesson: architecture simplification must preserve the comparison question. [inferred]
- Current influence: led to `rpg_linear_interaction_hypercond` and fixed linear control. [repo-confirmed]

### Graph/GAT variants were conceptually appealing but costly

- Idea: use two graphs or heterogeneous graph attention for relation extraction. [repo-confirmed] [conversation-derived]
- Why reasonable: graphs explicitly model relations among self, allies, and enemies. [inferred]
- What went wrong: user observed the graph versions increased runtime dramatically, and there were concerns that fixed slots remained or graph construction did not truly solve the local subgraph issue. [conversation-derived]
- Failure type: computational overhead and conceptual mismatch. [conversation-derived] [inferred]
- Lesson: attention over semantically split entities is a cheaper and cleaner relation extractor for the current paper. [inferred]
- Current influence: graph variants remain as exploratory controls, not final main method. [repo-confirmed] [inferred]

### Global graph CTCE variants changed the execution assumption

- Idea: compute whole friendly/enemy or heterogeneous graph once per timestep to reduce repeated local computation. [repo-confirmed] [conversation-derived]
- Why reasonable: central computation can avoid per-agent repeated graph overhead. [conversation-derived]
- What went wrong: it becomes CTCE validation, not a decentralized CTDE execution model. [repo-confirmed]
- Failure type: changed problem setting. [repo-confirmed] [inferred]
- Lesson: computational efficiency fixes must not silently change execution assumptions. [inferred]
- Current influence: global graph variants are labelled CTCE upper-bound/validation. [repo-confirmed]

### Fixed structured controls performed too well on easy maps

- Idea: use fixed relation-conditioned structured maker as control. [repo-confirmed]
- Why reasonable: isolate dynamic generation from relation-feature conditioning. [repo-confirmed]
- What went wrong for the original narrative: if fixed performs similarly or better, then "hypernetwork is useful" is not established. [conversation-derived]
- Failure type: hypothesis too broad, map too easy, or fixed control already sufficient. [inferred]
- Lesson: paper claim must be conditional: dynamic heads help where relation-dependent decision rules change enough to matter. [repo-confirmed] [inferred]
- Current influence: experiment plan prioritizes `corridor` and hard maps. [repo-confirmed]

### Relation-conditioned mixer gate had a weaker story

- Idea: relation pattern gates agent Q contributions before QMIX mixing. [repo-confirmed]
- Why reasonable: relation context might influence credit assignment. [inferred]
- What went wrong: user questioned the logic of why a relation-conditioned mixer should be necessary. [conversation-derived]
- Failure type: conceptual justification gap. [conversation-derived]
- Lesson: local head adaptation has a clearer causal path: relation pattern affects local action scoring. [inferred]
- Current influence: mixer gate remains optional, not central. [repo-confirmed] [inferred]

### GPU acceleration and visualization introduced resource failures

- Idea: use GPU configs, AMP, and battle-trace visualization. [repo-confirmed]
- Why reasonable: CPU servers became unavailable, and visualization was needed for paper story. [conversation-derived]
- What went wrong: OOMs occurred; V100/driver issues and CPU memory bottlenecks appeared; SC2 sampling remained CPU-bound. [conversation-derived]
- Failure type: infrastructure bottleneck, not model concept. [conversation-derived]
- Lesson: report wall-clock carefully and separate sampling bottlenecks from model complexity. [inferred]
- Current influence: V100 AMP config, mask-value fix, and low-memory run templates. [repo-confirmed]

## 9. Relationship to Human-player-inspired Reasoning

The project can use a human-player-inspired narrative only carefully. [conversation-derived]

The model does not use human demonstrations. [repo-confirmed by absence of demo-loading pipeline] [inferred]

The model should not claim to mimic human players. [conversation-derived]

The safer framing is: the architecture is inspired by the way a player may separately reason about self status, ally support, enemy state, and target-specific interactions. [conversation-derived]

This framing supports self/ally/enemy decomposition and interaction-action scoring, but the technical claim should remain relation-conditioned observation-adaptive value estimation. [conversation-derived] [inferred]

If used in a paper, put the human-player framing in motivation or intuition, not as a method claim. [inferred]

## 10. Relationship to Narrative-only Papers or Papers Used as Writing Examples

Some papers were useful for writing strategy rather than technical method. The accessible record does not contain a complete list. [needs-human-confirmation]

Possible narrative lessons discussed:

- How to organize related work as a taxonomy of structure, relation reasoning, and dynamic parameterization. [conversation-derived]
- How to defend publicly available structural information as inductive bias rather than privileged expert knowledge. [conversation-derived]
- How to interpret results where a method is not best on easy tasks but is stronger on hard tasks. [conversation-derived]
- How to use figures/tables to position a method between fixed policies, relation encoders, and generated decision functions. [inferred]

CTEM may have been used both as a benchmark comparison and as a narrative example for recent MARL result positioning, but this needs confirmation. [conversation-derived] [needs-human-confirmation]

Do not confuse narrative inspiration with technical related work. For example, a paper that motivates hard-vs-easy benchmark interpretation is not necessarily a technical ancestor of relation-conditioned dynamic heads. [inferred]

## 11. Related Work Should Follow the Research Lineage

### Value factorization and CTDE

- Why discuss: the method is built on a QMIX-style learner. [repo-confirmed]
- Related part: recurrent per-agent Q network and centralized mixer. [repo-confirmed]
- Gap: standard value factorization does not specify how local decision functions should adapt to relation patterns. [inferred]
- Positioning: our method modifies local action-value heads while keeping CTDE value-factorized training. [repo-confirmed]

### Grouping, roles, and structured cooperation

- Why discuss: repository origin is GoMARL and the broader problem is efficient coordination structure. [repo-confirmed]
- Related part: group-level coordination versus relation-level adaptation. [inferred]
- Gap: grouping methods organize agents, but may not directly adapt target-specific local decision rules. [inferred]
- Positioning: relation-conditioned heads are a finer-grained mechanism than group assignment. [inferred]

### Entity-centric attention and graph MARL

- Why discuss: our relation capturer uses self-centered attention over ally/enemy entities; graph variants were implemented. [repo-confirmed]
- Related part: self/ally/enemy token encoders, cross-attention, GAT/graph alternatives. [repo-confirmed]
- Gap: relation encoders often produce better features, but may still feed a fixed decision head. [inferred]
- Positioning: our key step is not just relation encoding but using relation patterns to generate/modulate decision heads. [repo-confirmed]

### Hypernetworks and dynamic parameter generation

- Why discuss: generated heads are central. [repo-confirmed]
- Related part: condition-to-parameter mapping for Q heads. [repo-confirmed]
- Gap: generic hypernetworks may condition on task or agent id; our condition is an online local relation pattern. [repo-confirmed] [inferred]
- Positioning: relation-conditioned dynamic decision heads for interaction actions. [repo-confirmed]

### RPG and structured relation-to-decision methods

- Why discuss: closest conceptual ancestor for relation pattern and structured maker. [repo-confirmed] [conversation-derived]
- Related part: relation pattern, self/ally/enemy split, temporal relation hidden, ego/interaction maker split. [repo-confirmed]
- Gap: current project is single-task SMAC/QMIX, not RPG continual learning, and focuses on lightweight relation-conditioned interaction heads plus fixed controls. [repo-confirmed] [inferred]
- Positioning: RPG-inspired but not a reproduction. [repo-confirmed]

### Modulation, residual, and expert-based adaptation

- Why discuss: improvement variants constrain dynamic adaptation. [repo-confirmed]
- Related part: FiLM, residual, MoE, smoothness variants. [repo-confirmed]
- Gap: full generated heads can be expensive or unstable; constrained dynamic adaptation may be more practical. [conversation-derived] [inferred]
- Positioning: these are ablations/variants showing how relation patterns can affect interaction scoring. [repo-confirmed]

## 12. Safe Claims, Unsafe Claims, and Borderline Claims

### Safe claims

- The repository implements a clean hypernetwork family with multiple relation-conditioned variants. [repo-confirmed]
- The main models use a QMIX-style CTDE learner by default. [repo-confirmed]
- RPG-inspired variants split SMAC observations into self/ally/enemy components and use self-centered cross-attention plus a temporal relation GRU. [repo-confirmed]
- Structured maker variants decompose Q-values into ego-action and interaction-action branches. [repo-confirmed]
- `rpg_linear_interaction_hypercond` generates a one-layer interaction scorer from the relation condition. [repo-confirmed]
- `rpg_fixed_linear_structured_maker` is a matched fixed relation-conditioned control for the linear interaction version. [repo-confirmed]
- `rpg_smooth_linear_interaction_hypercond` adds a smoothness auxiliary loss encouraging nearby relation conditions to generate nearby interaction heads. [repo-confirmed]
- Preliminary results suggest map-dependent behavior: easy maps may not separate fixed and dynamic variants, while `corridor` appears more discriminative. [repo-confirmed via material package] [conversation-derived]
- Visualization tools exist to inspect battle traces, relation/head similarity, and relation/head dynamics. [repo-confirmed]

### Unsafe claims

- "We mimic human players." Unsafe because there is no human demonstration or imitation-learning pipeline. [repo-confirmed] [inferred]
- "We use no prior knowledge." Unsafe because the model uses SMAC observation/action semantics. [repo-confirmed]
- "The decomposition is automatically discovered." Unsafe because self/ally/enemy and ego/attack decomposition are manually specified. [repo-confirmed]
- "Hypernetworks universally improve MARL." Unsafe because fixed controls sometimes perform similarly or strongly. [repo-confirmed] [conversation-derived]
- "The method is SOTA on all maps." Unsafe because current evidence is preliminary and map-dependent. [repo-confirmed] [conversation-derived]
- "The method avoids gradient interference." Unsafe unless explicitly measured; the current code does not prove this. [conversation-derived] [inferred]
- "Relation-conditioned mixer is necessary." Unsafe without stronger conceptual and experimental support. [conversation-derived]

### Borderline claims

- "Dynamic interaction heads improve convergence on hard relation-sensitive maps." Borderline. Needs multiple seeds on `corridor`, `MMM2`, `3s5z_vs_3s6z`, and possibly `5m_vs_6m`, with steps-to-threshold and AUC. [repo-confirmed] [needs-human-confirmation]
- "Similar relation patterns generate similar MLP heads." Borderline. Needs relation-head alignment metrics across trained models and episodes, not just one trace. [repo-confirmed] [needs-human-confirmation]
- "Residual/FiLM/MoE constraints improve stability over direct generation." Borderline. Needs comparative learning curves and stability metrics. [repo-confirmed] [needs-human-confirmation]
- "Manual decomposition is fair." Borderline in reviewer perception. Needs clear framing as public environment semantics plus matched fixed controls. [inferred] [needs-human-confirmation for citations]
- "The method generalizes beyond SMAC." Borderline. Needs another environment or careful limitation statement. [needs-human-confirmation]

## 13. Final Research Lineage Summary

The project started from a broad question about cooperative MARL structure, originally in a GoMARL repository focused on automatic grouping. [repo-confirmed]

The research direction shifted from group-level coordination to local relation-level decision adaptation. [conversation-derived] [inferred]

RPG-inspired relation pattern modeling introduced the idea of splitting observations into self, ally, and enemy information, using self-centered attention, and maintaining temporal relation hidden states. [repo-confirmed]

Structured maker variants then separated ego-action values from enemy-interaction values, making the action head better aligned with SMAC action semantics. [repo-confirmed]

The key conceptual tension became whether relation patterns should simply be encoded as fixed-head inputs or should dynamically change the decision function. [conversation-derived]

Full dynamic structured makers were closer to RPG-style generation but became too expensive, motivating lighter variants that focus dynamic generation on interaction-action scoring. [repo-confirmed] [conversation-derived]

Fixed linear structured controls became essential because they test whether relation features and manual decomposition alone explain the gains. [repo-confirmed]

The current paper should position the contribution as relation-conditioned dynamic interaction-action heads under CTDE, not as generic hypernetwork superiority. [repo-confirmed] [inferred]

The most defensible story is conditional and precise: relation-conditioned dynamic heads may improve sample efficiency or performance when local interaction rules vary strongly across battle phases, while fixed relation-conditioned heads may already be sufficient on easier maps. [repo-confirmed] [inferred]

Failures were not side noise; they shaped the final design. Full generation taught us to avoid over-parameterized interaction hypernetworks. Graph variants taught us that explicit graph reasoning can be costly and may change execution assumptions. Strong fixed controls taught us to narrow the claim. Visualization and smoothness emerged because the paper needs to show not just performance, but a meaningful relation-to-decision-function mapping. [conversation-derived] [repo-confirmed]

## Missing Human Knowledge Needed

Please confirm the exact full title, venue, and key mechanism of the RPG paper. The repository only confirms RPG-inspired implementation comments; the exact bibliographic detail comes from conversation. [needs-human-confirmation]

Please list all papers discussed but not recorded here, especially top-conference MARL papers from AAMAS, IJCAI, NeurIPS, ICLR, and ICML that influenced framing or baselines. [needs-human-confirmation]

Please confirm whether CTEM is only an empirical benchmark comparison or also a narrative/technical inspiration. [needs-human-confirmation]

Please provide raw W&B run links or exported CSVs for `3s5z`, `3s5z_vs_3s6z`, `5m6m`, `5m_vs_6m`, `corridor`, and `MMM2` so claims can move from conversation-derived screenshots to documented evidence. [needs-human-confirmation]

Please confirm which variants actually finished training and which were stopped due to cost: full structured, readout, linear, fixed linear, residual, FiLM, MoE, smooth, graph, hetero graph, global graph, relation mixer gate. [needs-human-confirmation]

Please confirm whether any advisor suggested emphasizing or avoiding human-player-inspired language. [needs-human-confirmation]

Please confirm whether the final paper should target "dynamic head generation", "relation-conditioned interaction-action scoring", "smooth relation-to-head mapping", or "efficient structured relation-aware QMIX" as the main title-level contribution. [needs-human-confirmation]

Please confirm which maps will be the final main evidence and which will be sanity/appendix maps. [needs-human-confirmation]

Please confirm whether multiple seeds are affordable for the final selected map/model pairs. [needs-human-confirmation]

Please provide any failed-run logs that show runtime/OOM/instability for graph and full hypernetwork variants if the paper will discuss efficiency motivation. [needs-human-confirmation]

Please confirm whether `5m_vs_6m` is the correct map name in the server SMAC installation, because conversation used both `5m6m` and `5m_vs_6m`. [needs-human-confirmation]

