"""Opt-in state gates and centered/bounded/sum/alpha mixer weight composition.

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
        self.combination = str(getattr(args, "mixer_residual_combination", "centered"))
        self.secondary_ones = bool(getattr(args, "mixer_residual_secondary_ones", False))
        if self.combination not in {"centered", "bounded", "sum", "alpha_sum"}:
            raise ValueError("Unknown mixer weight combination")
        if not self.residual and (self.combination != "centered" or self.secondary_ones):
            raise ValueError("Weight composition requires two parameter branches")
        if self.secondary_ones and self.combination != "sum":
            raise ValueError("Both-ones control is defined only for direct sum")
        self.fill_value = float(getattr(args, "mixer_residual_fill_value", 1))
        if self.fill_value not in (0, 1) or (not self.residual and self.fill_value != 1):
            raise ValueError("Zero-fill override is only supported for the residual ablation")
        self.residual_scale = float(args.mixer_onefill_residual_scale)
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError("Residual scale must be positive and finite")
        if self.combination == "bounded" and self.residual_scale >= 1:
            raise ValueError("Bounded relative modulation requires 0 < epsilon < 1")
        self.alpha_logit = None
        if self.combination == "alpha_sum":
            initial_alpha = float(getattr(args, "mixer_residual_alpha_initial", .1))
            if not 0 < initial_alpha < 1:
                raise ValueError("Alpha initialization must be strictly between zero and one")
            self.alpha_logit = nn.Parameter(th.tensor(math.log(initial_alpha / (1 - initial_alpha))))
        if self.secondary_ones:
            # This control has no operative field gate. Do not update dead logits.
            self.gate_logits.requires_grad_(False)
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
        if self.secondary_ones:
            return th.ones_like(states)
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

    def controlled_input(self, states, mask):
        if self.secondary_ones:
            return th.ones_like(states)
        if self.fill_value == 0:
            return mask * states
        return mask * states + (1 - mask) * th.ones_like(states)

    def alpha(self):
        if self.alpha_logit is None:
            return None
        # Keep strict bounds even if finite-precision sigmoid saturates.
        eps = th.finfo(self.alpha_logit.dtype).eps
        return self.alpha_logit.sigmoid().clamp(eps, 1 - eps)

    def weight_components(self, inputs):
        """Signed raw heads/contributions and actual nonnegative effective W.

        Sum models take abs AFTER addition. These are parameter branches,
        never two separately evaluated Q_tot networks.
        """
        if not self.residual:
            return {}
        flat = inputs.reshape(-1, self.state_dim)
        ones = th.ones_like(flat)
        if self.secondary_ones:
            flat = ones
        closed = ones if self.fill_value == 1 else th.zeros_like(flat)
        result = {}
        for label, base_head, secondary_head in (
                ("w1", self.base.hyper_w1, self.residual_w1),
                ("w_final", self.base.hyper_w_final, self.residual_w2)):
            main, secondary = base_head(ones), secondary_head(flat)
            if self.combination in {"centered", "bounded"}:
                delta = secondary - secondary_head(closed)
                if self.combination == "bounded":
                    main_part = main.abs()
                    sub_part = main_part * (self.residual_scale * delta.tanh())
                else:
                    main_part, sub_part = main, self.residual_scale * delta
            elif self.combination == "sum":
                main_part, sub_part = main, secondary
            else:
                alpha = self.alpha()
                main_part, sub_part = alpha * main, (1 - alpha) * secondary
            combined = main_part + sub_part
            result[label] = dict(raw_main=main, raw_secondary=secondary,
                main_contribution=main_part, secondary_contribution=sub_part,
                combined=combined, effective=combined.abs())
        return result

    def weight_parameters(self, inputs):
        if not self.residual:
            flat = inputs.reshape(-1, self.state_dim)
            return self.base.hyper_w1(flat).abs(), self.base.hyper_w_final(flat).abs()
        parts = self.weight_components(inputs)
        return parts["w1"]["effective"], parts["w_final"]["effective"]

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

    def diagnostic_components(self, states):
        """Deterministic soft-p diagnostic; not a sampled training mask."""
        inputs = self.controlled_input(states, self.sample_mask(states, "eval"))
        return {layer: {key: value.reshape(*states.shape[:-1], -1)
                        for key, value in parts.items()}
                for layer, parts in self.weight_components(inputs).items()}


def bernoulli_gate_kl(logits, prior):
    if not 0 < prior < 1:
        raise ValueError("KL prior is a KEEP probability in (0,1)")
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    reference = probability.new_tensor(prior)
    return (probability * (probability.log() - reference.log()) +
            (1 - probability) * (th.log1p(-probability) - th.log1p(-reference))).mean()
