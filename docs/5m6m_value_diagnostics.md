# 5m6m value-retention diagnostics

Six fresh runs: seeds 1, 2, 3 for `linear_baseline` (Linear observation-based
single generated head) and `hyper_hypermarl_id` (historical attention encoder,
ID-conditioned generated head). This is not a matched Linear-obs/Linear-ID
condition-source ablation. Both use main TD only, no gate/KL/QME.

Submission: `scripts/ozstar_submit_5m6m_value_diagnostics_10m_3seeds.py`.
Default invocation prints plans. `SUBMIT=YES` validates declared config keys,
home quota, smoke tests and all missing Slurm requests before submitting.
No cancellations; historical runs are not reused. Budget is 10,050,000 steps,
two days, 24G (Linear) / 32G (ID), eight rollout workers, batch 128, buffer 5000.
Checkpoint, test trajectory, battle trace and PCA output remain disabled.

Tests still run every 10K steps (32 episodes). Every 100K steps scalar
diagnostics are calculated over all those test episodes, with recurrent
histories replayed from zero hidden states. Padding, post-terminal steps and
dead-agent utilities are masked. Parameter gradients are not accumulated;
controller hidden state, module modes and RNG streams are restored.

W&B minimal logging explicitly allows `test_value/`. Each diagnostic logs
mean, population standard deviation, max absolute value and sample count:

- `q_tot`: executed joint-action value.
- `agent_q`, `agent_{i}_q`: executed-action agent utilities (not calibrated
  per-agent returns; QMIX utilities need not have a unique absolute scale).
- `mixer_w1`, `mixer_w_final`: absolute, state-generated effective mixer
  weights, not merely the hypernetwork's trainable parameter tensors.
- `mixer_dqtot_dqi`, `agent_{i}_dqtot_dqi`: derivative of joint value with
  respect to agent utility; this is not optimizer gradient norm.
- `episode_discounted_return`: reward-to-go until the end of the episode.
- `q_minus_finite_horizon_return`: comparison on all evaluated transitions,
  including time-limit truncations; not an unbiased value-error estimate.
- `naturally_terminal_episode`: 1 for naturally terminated episodes, 0 for
  time-limit truncations; mean is the fraction of naturally completed tests.
- `q_mc_bias`, `q_mc_abs_error`, `q_mc_squared_error`: joint Q minus sampled
  reward-to-go, excluding time-limit-truncated episodes.
- `initial_q_mc_bias`: same comparison at the first step of complete episodes.

Monte Carlo samples estimate, rather than reveal, expected value. Terminal-only
statistics have selection bias when the time-limit fraction is substantial;
always inspect their counts and termination fraction. A changing diagnostic
is a correlation, not proof of the cause of policy degradation.

Local validation: scalar diagnostic unit smoke tests and six-job config plans.
SMAC model smoke tests require the cluster's SMAC/PySC2 environment and are
mandatory at submission. Existing jobs are neither stopped nor modified.
