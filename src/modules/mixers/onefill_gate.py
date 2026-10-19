"""Opt-in one-filled stochastic state gates; legacy mixer controls stay intact.

Only the global state supplied to QMIX is gated. Decentralized acting never
calls this module. There is no evaluation threshold or mask feature input.
"""
import copy
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .qmix import QMixer


class OneFillGateMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_dim = math.prod(args.state_shape) if isinstance(args.state_shape, (list, tuple)) else int(args.state_shape)
        self.n_agents = args.n_agents
        self.grouped = False
        self.temperature = float(args.mixer_input_gate_temperature)
        initial = float(args.mixer_input_initial_probability)
        if not 0 < initial < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid one-fill gate probability/temperature")
        self.gate_logits = nn.Parameter(th.full((self.state_dim,), math.log(initial / (1 - initial))))
        self.base = QMixer(args)
        self.residual = bool(args.mixer_onefill_residual)
        self.residual_scale = float(args.mixer_onefill_residual_scale)
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError("Residual scale must be positive and finite")
        if self.residual:
            # Only weights are corrected; biases and V remain all-ones based.
            # Do not allocate unused secondary biases/value networks.
            self.residual_w1 = copy.deepcopy(self.base.hyper_w1)
            self.residual_w2 = copy.deepcopy(self.base.hyper_w_final)
            for network in (self.residual_w1, self.residual_w2):
                head = network if isinstance(network, nn.Linear) else network[-1]
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def sample_mask(self, states, mode="train"):
        if mode not in {"train", "target", "eval"}:
            raise ValueError("Unknown one-fill forward mode")
        logits = self.gate_logits.expand_as(states)
        if mode != "train":
            # Retain the old soft-target convention, but no hard eval threshold.
            # f(Q, E[x]) is NOT E[f(Q,x)] for a nonlinear mixer.
            return logits.sigmoid()
        u = th.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        soft = th.sigmoid((logits + u.log() - th.log1p(-u)) / self.temperature)
        hard = (soft >= .5).to(soft)
        # Hard Bernoulli forward, biased Binary-Concrete straight-through grad.
        return hard + (soft - soft.detach())

    @staticmethod
    def controlled_input(states, mask):
        return mask * states + (1 - mask) * th.ones_like(states)

    def weight_parameters(self, inputs):
        flat = inputs.reshape(-1, self.state_dim)
        if not self.residual:
            return self.base.hyper_w1(flat).abs(), self.base.hyper_w_final(flat).abs()
        ones = th.ones_like(flat)
        # Center BEFORE abs, preserving monotonicity of the final combined W.
        # At inputs=ones this correction is exactly zero, including after train.
        first = self.base.hyper_w1(ones) + self.residual_scale * (
            self.residual_w1(flat) - self.residual_w1(ones))
        final = self.base.hyper_w_final(ones) + self.residual_scale * (
            self.residual_w2(flat) - self.residual_w2(ones))
        return first.abs(), final.abs()

    def conditioned_value(self, agent_qs, states, mask):
        inputs = self.controlled_input(states, mask)
        if not self.residual:
            return self.base(agent_qs, inputs)
        batch, time, _ = agent_qs.shape
        first, final = self.weight_parameters(inputs)
        ones = th.ones_like(inputs).reshape(-1, self.state_dim)
        bias = self.base.hyper_b1(ones).reshape(-1, 1, self.base.embed_dim)
        value = self.base.value(ones).reshape(-1, 1, 1)
        hidden = F.elu(th.bmm(agent_qs.reshape(-1, 1, self.n_agents),
                             first.reshape(-1, self.n_agents, self.base.embed_dim)) + bias)
        return (th.bmm(hidden, final.reshape(-1, self.base.embed_dim, 1)) + value).reshape(batch, time, 1)

    def forward(self, agent_qs, states, mode="eval", detach_gate=False):
        mask = self.sample_mask(states, mode)
        return self.conditioned_value(agent_qs, states, mask.detach() if detach_gate else mask)

    def diagnostic_parameters(self, states):
        inputs = self.controlled_input(states, self.sample_mask(states, "eval"))
        first, final = self.weight_parameters(inputs)
        return first.reshape(*states.shape[:-1], -1), final.reshape(*states.shape[:-1], -1)


def bernoulli_gate_kl(logits, prior):
    if not 0 < prior < 1:
        raise ValueError("KL prior is a KEEP probability in (0,1)")
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    reference = probability.new_tensor(prior)
    return (probability * (probability.log() - reference.log()) +
            (1 - probability) * (th.log1p(-probability) - th.log1p(-reference))).mean()
