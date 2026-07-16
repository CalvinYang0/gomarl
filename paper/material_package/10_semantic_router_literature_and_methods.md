# Adaptive Semantic Routing: Literature, Implementation, and Ablation Notes

## 1. Precise positioning

The six routing variants are not direct reproductions of six published algorithms. Their scoring
principles have clear precedents, but the following combination is specific to this project:

1. Treat each raw SMAC observation scalar as a routable semantic slot.
2. Estimate a slot score during MARL training.
3. Route high- or low-scoring slots into one of two information-processing branches:
   - **TOKEN**: the slot contributes to the corresponding entity token and enters the public
     Transformer as content/value information.
   - **BIAS**: the slot is removed from the entity-token content and its residual embedding is
     converted into a lightweight attention bias.
4. Use the resulting relation representation to condition the hypernetwork-generated decision
   head.

Therefore, the defensible claim is:

> We adapt several established notions of information consistency, temporal slowness,
> gradient saliency, gradient sign purity, Jacobian sensitivity, and differentiable
> counterfactual selection into a unified semantic routing mechanism for hypernetwork-based MARL.

It would be inaccurate to say that MACKRL, SNIP, GradDrop, or COMA already proposed the current
TOKEN/BIAS observation routing algorithm.

## 2. Common routing pipeline

Let `x_j` be raw observation slot `j`. A slot can be, for example,
`enemy_3_health`, `ally_2_relative_x`, or `self_unit_type_0`. The current implementation gives
each raw slot an independent route.

### 2.1 Score smoothing

Each criterion produces a score `s_j`. The router keeps an exponential moving average:

```text
s_bar_j <- beta * s_bar_j + (1 - beta) * s_j
beta = 0.99
```

### 2.2 Score-to-probability mapping

For observer consistency, temporal stability, and gradient consistency, scores are already in
`[0, 1]`:

```text
p_j = clip(s_bar_j, 0, 1)
```

For gradient importance and parameter sensitivity, only relative scale is meaningful:

```text
p_j = sigmoid(((s_bar_j / mean(s_bar)) - 1) / temperature)
temperature = 0.1
```

Thus `p_j > 0.5` is equivalent to `s_bar_j > mean(s_bar)`.

For counterfactual routing, the signed score is normalized without removing its sign:

```text
p_j = sigmoid(s_bar_j / (mean(abs(s_bar)) * temperature))
```

### 2.3 Route decision

The normal direction is:

```text
p_j > 0.5  -> TOKEN
p_j <= 0.5 -> BIAS
```

The direction-ablation variants first compute the normal TOKEN budget `K`, then send the
**lowest-scoring K slots** to TOKEN. This preserves the TOKEN count while reversing the ranking.

### 2.4 Schedule

- Before `250k` environment steps: use the same manual initialization for every variant.
- From `250k` to `5M`: update the route from the selected score.
- At `5M`: freeze the route.

The manual initialization puts health, shield, and unit-type slots in TOKEN, while geometry,
visibility/attack availability, and movement availability start in BIAS. This is only a shared
warmup initialization; it is not the final adaptive split.

## 3. Variant 1: Observer consistency

### Literature origin

- **Multi-Agent Common Knowledge Reinforcement Learning**, NeurIPS 2019. MACKRL conditions
  hierarchical group policies on information that all relevant agents can reconstruct, allowing
  decentralized agents to coordinate on common knowledge.
- **Common Information based Approximate State Representations in Multi-Agent Reinforcement
  Learning**, AISTATS 2022. It formalizes approximate common and private state representations
  for Dec-POMDPs and derives performance-loss bounds for compressed representations.

These papers motivate separating information according to how consistently it is shared or
reconstructable across agents. They do **not** classify raw observation dimensions using variance.

### Current implementation

For slot `j`, executing agents are treated as repeated observers. Within each batch item:

```text
mu_j         = mean_agent(x_agent,j)
dispersion_j = mean_batch,agent((x_agent,j - mu_j)^2)
scale_j      = mean_batch,agent(x_agent,j^2)
score_j      = clip(1 - dispersion_j / max(scale_j, eps), 0, 1)
```

High cross-agent agreement gives a high score and therefore favors TOKEN.

### Intended interpretation

Slots that look similar from multiple agents may describe relatively shared scene information,
so they are allowed to participate in the richer joint Transformer representation.

### Important limitation

This score is only a proxy for common information. Equal numerical values do not establish
epistemic common knowledge. Zero padding is especially problematic: an always-zero invisible slot
has zero dispersion and can receive an artificially high consistency score. This is a plausible
reason for the observed poor performance.

## 4. Variant 2: Temporal stability

### Literature origin

- **Slow Feature Analysis: Unsupervised Learning of Invariances**, Neural Computation 2002.
  Slow Feature Analysis learns features whose outputs vary slowly over time, based on the idea
  that slowly varying latent factors often represent stable properties of a changing scene.

The current router borrows the temporal-slowness prior. It does not implement the full SFA
optimization with variance and decorrelation constraints.

### Current implementation

For adjacent observations:

```text
change_j = mean_batch,agent(abs(x_t,j - x_t-1,j))
scale_j  = mean_batch,agent(abs(x_t,j) + abs(x_t-1,j))
score_j  = clip(1 - change_j / max(scale_j, eps), 0, 1)
```

High temporal stability favors TOKEN.

### Intended interpretation

Stable attributes may provide a reliable scene context for the Transformer, while rapidly changing
attributes can act as lightweight decision modulation through BIAS.

### Important limitation and observed collapse

Zero padding, unit-type bits, and many inactive availability slots are naturally constant. They
therefore receive scores close to one. The router can collapse toward all-TOKEN even when those
slots are not useful. The resulting model is still reasonably capable because all-TOKEN degenerates
toward a full Transformer; this explains why route collapse did not necessarily produce the worst
return.

## 5. Variant 3: Gradient importance

### Literature origin

- **SNIP: Single-shot Network Pruning based on Connection Sensitivity**, ICLR 2019. SNIP adds
  multiplicative connection indicators and uses the magnitude of the loss derivative with respect
  to each indicator as first-order connection saliency.
- **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency
  Maps**, ICLR Workshop 2014. It uses output gradients with respect to inputs as local feature
  saliency.

SNIP is the closer mechanical precedent because our method also differentiates through a
multiplicative probe.

### Current implementation

Each raw slot receives a probe scale `alpha_j`, initialized to one and excluded from normal optimizer
updates:

```text
x'_j = alpha_j * x_j
g_j  = d(L_TD) / d(alpha_j)
score_j = abs(g_j)
```

The score is EMA-smoothed and compared with the current mean score. In the normal variant,
above-mean slots go to TOKEN. In the inverse variant, the same TOKEN budget is assigned to the
lowest-scoring slots.

### Intended interpretation

`abs(dL/dalpha_j)` is the first-order change in TD loss under a small multiplicative perturbation of
slot `j`. A large value means the current policy loss is locally sensitive to that slot.

### Important limitation

Gradient magnitude is local, noisy, and scale-dependent. It indicates sensitivity, not causality.
It can also prioritize unstable features. This is precisely why the normal-versus-inverse direction
ablation is required.

## 6. Variant 4: Gradient consistency

### Literature origin

- **Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout**, NeurIPS
  2020. GradDrop measures gradient-sign purity across multiple losses and probabilistically removes
  gradients with a conflicting sign.

GradDrop computes consistency across simultaneous task losses. Our adaptation computes sign
consistency across training time for each semantic slot.

### Current implementation

For each slot gradient `g_j`:

```text
m_j = EMA(g_j)
a_j = EMA(abs(g_j))
score_j = abs(m_j) / max(a_j, eps)
```

- `score_j` near one: gradients repeatedly point in the same direction.
- `score_j` near zero: positive and negative gradients cancel over time.
- `score_j > 0.5`: favors TOKEN.

### Intended interpretation

Semantics with a persistent optimization direction were treated as reliable content, while
sign-conflicting semantics were sent to the lightweight BIAS path.

### Important limitation and observed collapse

MARL is highly nonstationary: exploration, moving TD targets, and changing teammates naturally
cause gradient signs to change. Gradient inconsistency therefore does not mean semantic
unimportance. This router collapsed toward BIAS and produced the worst performance because BIAS
cannot replace the content/value capacity of entity tokens. The experiment argues against using
gradient consistency as a direct branch-selection criterion.

## 7. Variant 5: Hypernetwork parameter sensitivity

### Literature origin

- **HyperNetworks**, ICLR 2017. It establishes the use of one network to generate the weights of
  another network and trains both jointly end to end.
- **A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines**,
  Communications in Statistics 1989. Hutchinson introduces random-vector trace estimation,
  which underlies efficient stochastic estimation of large Jacobian norms.
- Jacobian-norm work such as **Improving DNN Robustness to Adversarial Attacks using Jacobian
  Regularization**, ECCV 2018, treats input-output Jacobian magnitude as local model sensitivity.

No cited paper uses this exact estimator to route MARL observation semantics into TOKEN/BIAS.

### Current implementation

Let `theta_jgen` denote the generated interaction-head parameter tensor and `alpha` the semantic
probe scales. Form a random Rademacher projection at selected timesteps:

```text
v_k in {-1, +1}
z = mean_k(<theta_gen,k, v_k> / sqrt(number_of_parameters))
g_j = d(z) / d(alpha_j)
score_j = abs(g_j)
```

The implementation samples four sequence timesteps for this probe. It uses vector-Jacobian
products instead of explicitly constructing the full Jacobian `d(theta_gen)/d(alpha)`.

The normal variant routes above-mean sensitivity to TOKEN. The inverse variant keeps the same
TOKEN budget but selects the least-sensitive slots.

### Intended interpretation

This criterion is hypernetwork-specific: it asks which observation slots most strongly change the
generated decision-head parameters, rather than which slots merely affect the current TD loss.

### Important limitation

A small number of random projections is a noisy estimate. Sensitivity also does not imply that the
change is beneficial. Normal-versus-inverse routing is therefore a necessary direction test.

## 8. Variant 6: First-order counterfactual routing

### Literature origin

- **Counterfactual Multi-Agent Policy Gradients**, AAAI 2018. COMA evaluates an agent's action
  against a counterfactual baseline that changes that agent's action while holding the other agents'
  actions fixed, addressing multi-agent credit assignment.
- **DARTS: Differentiable Architecture Search**, ICLR 2019. DARTS continuously relaxes discrete
  architecture choices so that operation selection can be optimized by gradient descent.

COMA supplies the counterfactual-credit principle; DARTS is closer to the differentiable branch
selection mechanism used here.

### Current implementation

For each slot, a zero-valued route probe `rho_j` is inserted with a straight-through identity:

```text
r'_j = r_j + rho_j - stop_gradient(rho_j)
```

The forward pass still uses the current hard route `r_j`, while backpropagation provides:

```text
g_j = d(L_TD) / d(rho_j)
score_j = -g_j
```

A positive score means that locally increasing the TOKEN assignment is expected to decrease TD
loss, so the slot is routed toward TOKEN.

### Important limitation

This is not an exact discrete leave-one-out experiment. The code does not run a complete TOKEN and
BIAS rollout for every slot; it uses a first-order local approximation. It should be described as
**first-order differentiable counterfactual routing**, not exact counterfactual evaluation.

## 9. Literature-to-method relationship table

| Variant | Main references | What is borrowed | What is new in this project |
|---|---|---|---|
| Observer consistency | MACKRL, NeurIPS 2019; Common Information ASR, AISTATS 2022 | Shared/reconstructable information is useful for decentralized coordination | Cross-agent slot consistency as a TOKEN/BIAS score |
| Temporal stability | Slow Feature Analysis, Neural Computation 2002 | Slowly varying information can represent stable scene factors | Normalized adjacent-observation difference used for routing |
| Gradient importance | SNIP, ICLR 2019; gradient saliency, ICLR Workshop 2014 | Multiplicative-gate gradient magnitude as first-order saliency | TD-loss sensitivity of semantic slots selects the processing branch |
| Gradient consistency | GradDrop, NeurIPS 2020 | Gradient-sign purity indicates agreement/conflict | EMA sign purity across MARL training time controls semantic routing |
| Parameter sensitivity | HyperNetworks, ICLR 2017; Hutchinson estimator, 1989; Jacobian regularization, ECCV 2018 | Generated weights and efficient Jacobian sensitivity estimation | Route semantics by their influence on generated decision-head parameters |
| Counterfactual | COMA, AAAI 2018; DARTS, ICLR 2019 | Counterfactual credit and differentiable discrete-choice relaxation | Straight-through first-order TOKEN-vs-BIAS route credit |

## 10. Current experimental interpretation

### Variants removed from the next stage

1. **Observer consistency**: poor performance; numerical agreement is an inadequate proxy for
   common information, and zero padding biases the score.
2. **Temporal stability**: route collapsed toward TOKEN. It retained some performance because
   all-TOKEN is a relatively safe full-Transformer fallback, but the criterion did not produce a
   meaningful split.
3. **Gradient consistency**: route collapsed toward BIAS and performed worst. In nonstationary
   MARL, sign changes are not evidence of irrelevance, and BIAS is too weak to carry most semantic
   content.

### Variants retained

1. **Gradient importance**: directly measures TD-loss sensitivity.
2. **Parameter sensitivity**: directly targets the hypernetwork-generated decision parameters.
3. **First-order counterfactual routing**: estimates which branch locally reduces TD loss.

### Current direction ablation

On `5m_vs_6m`, run:

1. Gradient importance, high score -> TOKEN.
2. Gradient importance, lowest scores receive the same TOKEN budget.
3. Parameter sensitivity, high score -> TOKEN.
4. Parameter sensitivity, lowest scores receive the same TOKEN budget.
5. Counterfactual, positive TOKEN benefit -> TOKEN.

The direction ablations are essential because the router does not learn the mapping from score to
branch. It learns the policy and therefore changes the measured scores, but the rule "high score
goes to TOKEN" is an explicit design choice.

## 11. Recommended presentation wording

Use the following three-level story:

1. **Problem**: Raw multi-agent observations mix semantic factors with different coordination,
   temporal, optimization, and hypernetwork effects. Sending every factor through the same path can
   create interference and unnecessary dynamic parameter variation.
2. **Framework**: Introduce a unified two-path semantic router. TOKEN provides rich relational
   content; BIAS provides lightweight attention modulation. Different literature-grounded scores
   instantiate the router without changing the downstream policy.
3. **Finding**: Not every plausible score is suitable. Observer/temporal/gradient consistency
   expose predictable failure modes, while TD-gradient importance, generated-parameter sensitivity,
   and first-order branch credit are better aligned with the functions of TOKEN and BIAS.

Do not present all six as successful contributions. Present the first six-way experiment as a
criterion-screening study, then present the retained three and direction controls as the validated
method-development path.

## 12. Primary references

1. Schroeder de Witt et al. [Multi-Agent Common Knowledge Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2019/hash/f968fdc88852a4a3a27a81fe3f57bfc5-Abstract.html), NeurIPS 2019.
2. Kao and Subramanian. [Common Information based Approximate State Representations in Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v151/kao22a.html), AISTATS 2022.
3. Wiskott and Sejnowski. [Slow Feature Analysis: Unsupervised Learning of Invariances](https://doi.org/10.1162/089976602317318938), Neural Computation 2002.
4. Lee, Ajanthan, and Torr. [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://www.robots.ox.ac.uk/~tvg/publications/2019/SNIP-ICLR-camera-ready.pdf), ICLR 2019.
5. Simonyan, Vedaldi, and Zisserman. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://ora.ox.ac.uk/objects/uuid:c46c7936-e692-4f58-b809-1e9e471f220e), ICLR Workshop 2014.
6. Chen et al. [Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout](https://proceedings.neurips.cc/paper/2020/file/16002f7a455a94aa4e91cc34ebdb9f2d-Paper.pdf), NeurIPS 2020.
7. Ha, Dai, and Le. [HyperNetworks](https://openreview.net/forum?id=rkpACe1lx), ICLR 2017.
8. Hutchinson. [A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines](https://doi.org/10.1080/03610918908812806), Communications in Statistics 1989.
9. Jakubovitz and Giryes. [Improving DNN Robustness to Adversarial Attacks using Jacobian Regularization](https://openaccess.thecvf.com/content_ECCV_2018/html/Daniel_Jakubovitz_Improving_DNN_Robustness_ECCV_2018_paper.html), ECCV 2018.
10. Foerster et al. [Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/view/11794), AAAI 2018.
11. Liu, Simonyan, and Yang. [DARTS: Differentiable Architecture Search](https://openreview.net/forum?id=S1eYHoC5FX), ICLR 2019.
