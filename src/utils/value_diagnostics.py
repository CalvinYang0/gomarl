"""Scalar-only value diagnostics on greedy evaluation episodes.

No optimizer step, persistent model copy, images, or trajectory files. Monte
Carlo returns are finite-episode samples, not exact expected values. Calibration
metrics exclude time-limit truncations; finite-horizon comparisons are separate.
"""
import random

import numpy as np
import torch as th

CORE_VALUE_METRICS = frozenset("test_value/" + key for key in (
    "q_tot_mean", "q_tot_std", "agent_q_mean", "q_mc_bias_mean",
    "q_mc_abs_error_mean", "mixer_dqtot_dqi_mean", "mixer_w1_mean",
    "mixer_w_final_mean", "naturally_terminal_episode_mean", "q_mc_bias_count",
))


class ValueDiagnosticSummary:
    def __init__(self):
        self.moments = {}

    def add(self, name, values):
        values = values.detach().double().reshape(-1)
        if not values.numel():
            return
        if not th.isfinite(values).all():
            raise RuntimeError("Non-finite value diagnostic: " + name)
        count, total, squared, max_abs = self.moments.get(name, (0, 0., 0., 0.))
        self.moments[name] = (
            count + values.numel(), total + values.sum().item(),
            squared + values.square().sum().item(),
            max(max_abs, values.abs().max().item()),
        )

    def log(self, logger, t_env):
        for name, (count, total, squared, max_abs) in self.moments.items():
            mean = total / count
            prefix = "test_value/" + name
            values = {"_mean": mean, "_std": max(0., squared / count - mean ** 2) ** .5,
                      "_count": count}
            for suffix, value in values.items():
                if prefix + suffix in CORE_VALUE_METRICS:
                    logger.log_stat(prefix + suffix, value, t_env)


def episode_value_samples(agent_q, mixer, batch, gamma, summary):
    """Aggregate executed-action Q values; padding/dead agents are excluded."""
    rewards = batch["reward"][:, :-1].float()
    terminated = batch["terminated"][:, :-1].float()
    filled = batch["filled"].float()
    valid = filled[:, :-1] * filled[:, 1:]
    if valid.shape[1] > 1:
        valid[:, 1:] *= (1. - terminated[:, :-1]).cumprod(dim=1)
    mask = valid.squeeze(-1).bool()
    actions = batch["actions"][:, :-1].long()
    utilities = agent_q.gather(-1, actions).squeeze(-1)
    states = batch["state"][:, :-1]
    # In SMAC, a dead agent has only the no-op action available.
    alive = batch["avail_actions"][:, :-1].sum(-1) > 1
    summary.add("agent_q", utilities[mask.unsqueeze(-1) & alive])
    for agent in range(utilities.shape[-1]):
        summary.add("agent_{}_q".format(agent), utilities[:, :, agent][mask & alive[:, :, agent]])

    with th.enable_grad():
        inputs = utilities.detach().requires_grad_(True)
        joint = mixer(inputs, states)
        sensitivity = th.autograd.grad(joint.sum(), inputs)[0]
    joint = joint.detach()
    summary.add("q_tot", joint.squeeze(-1)[mask])
    summary.add("mixer_dqtot_dqi", sensitivity[mask.unsqueeze(-1).expand_as(sensitivity)])
    for agent in range(utilities.shape[-1]):
        summary.add("agent_{}_dqtot_dqi".format(agent), sensitivity[:, :, agent][mask])
    if hasattr(mixer, "hyper_w1") and hasattr(mixer, "hyper_w_final"):
        with th.no_grad():
            flat_states = states.reshape(-1, mixer.state_dim)
            w1 = mixer.hyper_w1(flat_states).abs().reshape(*mask.shape, -1)
            w2 = mixer.hyper_w_final(flat_states).abs().reshape(*mask.shape, -1)
        summary.add("mixer_w1", w1[mask])
        summary.add("mixer_w_final", w2[mask])

    returns = th.zeros_like(rewards)
    future = th.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        future = (rewards[:, t] + gamma * (1. - terminated[:, t]) * future) * valid[:, t]
        returns[:, t] = future
    bias = (joint - returns).squeeze(-1)
    summary.add("episode_discounted_return", returns.squeeze(-1)[mask])
    summary.add("q_minus_finite_horizon_return", bias[mask])
    complete = ((terminated * valid).sum(dim=1).squeeze(-1) > 0)
    summary.add("naturally_terminal_episode", complete.float())
    calibration_mask = mask & complete.unsqueeze(-1)
    summary.add("q_mc_bias", bias[calibration_mask])
    summary.add("q_mc_abs_error", bias[calibration_mask].abs())
    summary.add("q_mc_squared_error", bias[calibration_mask].square())
    summary.add("initial_q_mc_bias", bias[:, 0][complete & mask[:, 0]])


def collect_value_diagnostics(mac, mixer, batch, gamma, summary):
    """Replay recurrent histories, preserving RNG and controller hidden state."""
    if mixer is None:
        raise ValueError("Value diagnostics require a joint value mixer")
    saved_hidden = mac.hidden_states
    saved_modes = [(module, module.training) for module in mac.agent.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_devices = [batch["obs"].device.index] if batch["obs"].is_cuda else []
    try:
        with th.random.fork_rng(devices=cuda_devices), th.no_grad():
            mac.init_hidden(batch.batch_size)
            q_values = th.stack([
                mac.forward(batch, t=t, test_mode=True)
                for t in range(batch.max_seq_length - 1)
            ], dim=1)
            episode_value_samples(q_values, mixer, batch, gamma, summary)
    finally:
        mac.hidden_states = saved_hidden
        for module, training in saved_modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
