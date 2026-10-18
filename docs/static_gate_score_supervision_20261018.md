# Static gate: separate main TD and gate score supervision

Two models on 5m_vs_6m and 8m_vs_9m, seeds 1/2/3, 10M target
(`t_max=10050000`), exactly 12 jobs:

- `qmix_state_static10_aux_qchange`
- `qmix_state_static10_aux_calibration`

Exact names: `smac_{5m6m,8m9m}_{label}_10m_s{1,2,3}_ablate20261018`.
Reference settings: old 5m6m static10 and fresh 8m9m static10. This is a new
gate-training mechanism, not another grouped/dual-TD model.

## Two-way gradient isolation

Main TD still sees `state * sampled_gate`, but the gate tensor is detached.
Its optimizer contains only the ordinary agent and one QMIX mixer. Gate logits
are excluded from its parameters, optimizer state and gradient clipping.

After a main update, every 100 main updates (starting at update 0), reconstruct
executed-action utilities and compute fresh teacher targets under no_grad.
Then optimize only gate logits with `BCEWithLogits(logits, detached_target)`.
Gate uses a separate optimizer of the same Adam/RMSProp type, the main learning
rate by default, coefficient 1.0, and separate clipping. No main/target model
parameters are updated by the auxiliary step. Subsequent gate changes do affect
future main inputs; gradient isolation is not statistical independence.

Legacy static gate, L1, variance regularization and all grouped learners retain
their original behavior with `mixer_gate_aux_score=none`. Isolated supervision
cannot silently combine with L1 or variance penalties. There is no new mixer,
mask feature concatenation, dual TD objective, or actor input change.

## Matched attribution probes and target direction

Both versions use deterministic **soft** keep probabilities `p.detach()` for
scoring, not stochastic training samples or the hard evaluation mask. Hold
agent Q, mixer, p and valid samples fixed across all slots. Replace one raw
field with its mean over all valid TD transitions, holding p fixed. Every field
can be probed even if its hard evaluation mask is closed. Out-of-padding and
post-terminal transitions never enter the reference/probe pool.

Q-change: `c_j = mean(abs(Q - Q_replaced_j))`, target `c_j / max_k(c_k)`.
Higher sensitivity implies a larger keep target. If the maximum is <=1e-8,
skip the gate update rather than interpreting absent evidence as all-close.

MC quality: restrict probes to complete observed episodes and compute actual
discounted behavior returns G. Use
`d_j = (MSE(Q,G) - MSE(Q_replaced_j,G)) / max(mean(G^2),1e-6)`;
target `sigmoid(-clamp(d_j,-20,20))`. Positive replacement benefit means lower
quality and a lower keep target. The field reference mean still uses all valid
TD transitions. Missing complete episodes skip the gate optimizer step entirely,
including momentum. G is not known optimal-Q ground truth. The models differ
in score/target mapping and MC's complete-return sample eligibility.

No EMA, stale target reuse, direct minimization of Q-change, direct MC training
of the mixer, or extra sparsity/budget constraint is introduced. Probability
targets are supervision signals, not proven causal information quality.

## Preserve static gate forward settings and report limitations

Initial p=0.1, temperature=0.5, stochastic continuous training masks, soft target
masks and hard evaluation threshold p>0.1 stay as before. Only main TD's gradient
to gate is removed. Neutral MC quality maps to 0.5; with the unchanged 0.1 test
threshold, many/all fields may remain open. This is intentional: the experiment
tests supervision/routing, not guaranteed sparsity. Do not change the threshold
or add L1 mid-run to make a heatmap look sparse. Scoring at current p also means
the teacher depends on the current gated mixer and may have feedback bias.

The default gate schedule is slower than main TD (one gate step per 100 main
steps); probability learning may therefore be gradual. These settings are
recorded and checkpoint-validated, not silently tuned after seeing results.

## Diagnostics, checkpoints and launch

Native `mixer_input/gate_aux_heatmap` and the existing partition heatmap show
current p/hard mask, raw Q-change or signed MC benefit, detached target and
current probability. JSON includes criterion, targets, p at scoring, probe count,
probe update and auxiliary update count. Scores/targets are pre-gate-update;
the top current probability is post-update. Missing supervision is labelled,
not displayed as fresh zero/stale importance. Scalar logs include auxiliary
loss/gradient norm/availability, mean target, mean p and hard keep count.

Save/restore `opt_gate.th` and supervision buffers/counters along with the
original model. Reject criterion, interval, coefficient or learning-rate changes
on resume. Old unsupervised checkpoints retain the legacy default mode.

```bash
cd /home/kyang/code/gomarl-dual-branch &&
git pull --ff-only origin codex/linear-id-baselines &&
SUBMIT=YES REPO_DIR="$PWD" /home/kyang/.conda/envs/marl_cpu/bin/python -u scripts/ozstar_submit_static_gate_score_supervision_12jobs.py
```

Without SUBMIT=YES, plan-only. Submission validates both maps, unchanged 2 GiB
startup headroom (not a batch capacity guarantee), simulator-free gradient/media
tests, and all missing Slurm jobs before any submission. Exact active/completed
names are retained on repeat. No old jobs are restarted/cancelled and no data is
deleted. Running-only sync scope is unchanged. Ended raw backfill, preservation,
local-only figures and mean3 inventories include the new cohort. This command
does not itself launch plotting/cloud analysis uploads or a new tmux loop.

QA checks exact formulas, signed direction, fixed p/Q/probes, padding and MC
eligibility, all-zero/missing-data skip including optimizer momentum, Adam and
RMSProp two-way isolation, equal main updates under a 1e6 auxiliary multiplier,
both maps' real dimensions, native SDK media, checkpoint round trips, old mode
compatibility, and mock scheduler idempotence. No SC2 game or real cloud upload
is used in the preflight.
