"""Deterministic, per-state-coordinate sigmoid gates; no mask sampling."""
import math

import torch as th
import torch.nn as nn

from .qmix import QMixer


class SoftOneFillGateMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if (args.mixer_onefill_residual or
                getattr(args, "mixer_residual_combination", "centered") != "centered" or
                getattr(args, "mixer_residual_secondary_ones", False) or
                float(getattr(args, "mixer_residual_fill_value", 1)) != 1):
            raise ValueError("Soft one-fill is a single complete QMIX, with fill value 1")
        self.state_dim = math.prod(args.state_shape) if isinstance(args.state_shape, (list, tuple)) else int(args.state_shape)
        self.n_agents = args.n_agents
        initial = float(args.mixer_input_initial_probability)
        if not math.isfinite(initial) or not 0 < initial < 1:
            raise ValueError("Initial soft gate value must lie strictly between 0 and 1")
        self.gate_logits = nn.Parameter(th.full((self.state_dim,), math.log(initial / (1 - initial))))
        self.base = QMixer(args)
        # Compatibility with the common TD/checkpoint/diagnostic interface.
        self.grouped = self.residual = self.secondary_ones = False
        self.combination, self.fill_value = "centered", 1.
        self.alpha_logit = self.temperature = None
        self.residual_scale = 0.

    def gate_values(self, states, mode="train"):
        if mode not in {"train", "target", "eval"}:
            raise ValueError("Unknown soft one-fill forward mode")
        return self.gate_logits.sigmoid().expand_as(states)

    def controlled_input(self, states, values):
        return values * states + (1 - values)

    def conditioned_value(self, agent_qs, states, values):
        return self.base(agent_qs, self.controlled_input(states, values))

    def forward(self, agent_qs, states, mode="eval", detach_gate=False):
        values = self.gate_values(states, mode)
        return self.conditioned_value(agent_qs, states, values.detach() if detach_gate else values)

    def diagnostic_parameters(self, states):
        inputs = self.controlled_input(states, self.gate_values(states, "eval"))
        flat = inputs.reshape(-1, self.state_dim)
        return (self.base.hyper_w1(flat).abs().reshape(*states.shape[:-1], -1),
                self.base.hyper_w_final(flat).abs().reshape(*states.shape[:-1], -1))
