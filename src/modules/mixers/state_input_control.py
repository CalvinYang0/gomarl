"""QMIX state-only controls; never generate or gate agent action heads."""
from copy import copy
import math

import torch as th
import torch.nn as nn

from .qmix import QMixer


class StateInputControlMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variant = args.mixer_input_control
        if self.variant not in {"static_gate", "group_qchange", "group_tderror", "group_calibration"}:
            raise ValueError("Unknown mixer state control: " + self.variant)
        self.state_dim = math.prod(args.state_shape) if isinstance(args.state_shape, (tuple, list)) else int(args.state_shape)
        self.grouped = self.variant != "static_gate"
        self.threshold = float(getattr(args, "mixer_input_gate_threshold", .1))
        self.temperature = float(getattr(args, "mixer_input_gate_temperature", .5))
        p = float(getattr(args, "mixer_input_initial_probability", .1))
        if not 0 < p < 1 or not 0 <= self.threshold <= 1 or self.temperature <= 0:
            raise ValueError("Invalid mixer static gate settings")
        self.register_buffer("important_slots", th.zeros(self.state_dim, dtype=th.bool))
        self.register_buffer("score_ema", th.zeros(self.state_dim))
        self.register_buffer("score_initialized", th.zeros(self.state_dim, dtype=th.bool))
        self.register_buffer("qchange_scores", th.zeros(self.state_dim))
        self.register_buffer("tderror_scores", th.zeros(self.state_dim))
        if self.variant == "group_calibration":
            for name in ("removal_gain_scores", "original_mc_mse", "masked_mc_mse"):
                self.register_buffer(name, th.zeros(self.state_dim))
            self.register_buffer("calibration_probe_count", th.zeros((), dtype=th.long))
        self.register_buffer("grouping_updates", th.zeros((), dtype=th.long))
        self.ema = float(getattr(args, "slot_importance_ema", .8))
        self.demote_ratio = float(getattr(args, "slot_demote_ratio", .5))
        if not 0 <= self.ema < 1 or not 0 < self.demote_ratio < 1:
            raise ValueError("Invalid mixer grouping settings")
        self.gate_logits = nn.Parameter(th.full((self.state_dim,), math.log(p / (1 - p)))) if not self.grouped else None
        local = copy(args)
        # Partition bits distinguish excluded fields from observed zeros.
        local.state_shape = self.state_dim * (2 if self.grouped else 1)
        self.mixers = nn.ModuleList([QMixer(local) for _ in range(2 if self.grouped else 1)])

    def masks(self, states, mode="eval"):
        if mode not in {"train", "target", "eval"}:
            raise ValueError("Invalid mixer forward mode")
        if self.grouped:
            return th.stack((self.important_slots, ~self.important_slots)).to(states)
        probability = self.gate_logits.sigmoid()
        if mode == "train":
            logits = self.gate_logits.expand_as(states)
            u = th.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gate = th.sigmoid((logits + u.log() - th.log1p(-u)) / self.temperature)
        elif mode == "target":
            gate = probability.expand_as(states)
        else:
            gate = (probability > self.threshold).to(states).expand_as(states)
        return gate.unsqueeze(0)

    def controlled_input(self, states, mask):
        observed = states * mask
        if self.grouped:
            observed = th.cat((observed, mask.expand_as(states)), dim=-1)
        return observed

    def conditioned_value(self, agent_qs, states, group, mask):
        observed = self.controlled_input(states, mask)
        # W1, W2, b1 AND V all receive the controlled input, never raw state.
        return self.mixers[group](agent_qs, observed)

    def diagnostic_parameters(self, states, group=0):
        observed = self.controlled_input(states, self.masks(states, "eval")[group])
        core = self.mixers[group]
        flat = observed.reshape(-1, core.state_dim)
        shape = states.shape[:-1]
        return (core.hyper_w1(flat).abs().reshape(*shape, -1),
                core.hyper_w_final(flat).abs().reshape(*shape, -1))

    def values(self, agent_qs, states, mode="eval"):
        masks = self.masks(states, mode)
        return th.cat([self.conditioned_value(agent_qs, states, g, masks[g])
                       for g in range(len(self.mixers))], dim=-1)

    def forward(self, agent_qs, states):
        # Generic value diagnostics evaluate the important branch. Acting uses
        # only the single ordinary agent Q network and never calls this mixer.
        return self.values(agent_qs, states, mode="eval")[..., :1]

    @th.no_grad()
    def apply_scores(self, qchange, tderror, removal_gain=None):
        for scores in (qchange, tderror):
            if scores.shape != self.score_ema.shape or not th.isfinite(scores).all() or (scores < 0).any():
                raise ValueError("Invalid mixer state attribution")
        self.qchange_scores.copy_(qchange)
        self.tderror_scores.copy_(tderror)
        if not self.grouped:
            return None
        scores = qchange if self.variant == "group_qchange" else tderror
        if self.variant == "group_calibration":
            if (removal_gain is None or removal_gain.shape != self.score_ema.shape or
                    not th.isfinite(removal_gain).all()):
                raise ValueError("Invalid signed calibration improvement scores")
            self.removal_gain_scores.copy_(removal_gain)
            scores = th.sigmoid(-removal_gain.clamp(-20, 20))
        before = self.important_slots.clone()
        self.score_ema.copy_(th.where(self.score_initialized,
                                     self.ema * self.score_ema + (1 - self.ema) * scores, scores))
        self.score_initialized.fill_(True)
        important, other = before.nonzero().flatten(), (~before).nonzero().flatten()
        promoted = demoted = None
        if other.numel():
            top = other[self.score_ema[other].argmax()]
            if self.score_ema[top] > 1e-8:
                promoted = int(top)
        if important.numel() > 1:
            top = self.score_ema[important].max()
            last = important[self.score_ema[important].argmin()]
            if top > 1e-8 and self.score_ema[last] < self.demote_ratio * top:
                demoted = int(last)
        if promoted is not None:
            self.important_slots[promoted] = True
        if demoted is not None:
            self.important_slots[demoted] = False
        self.score_initialized[before != self.important_slots] = False
        self.grouping_updates.add_(1)
        return dict(promoted=promoted, demoted=demoted,
                    important_before=before.cpu().tolist(), important_after=self.important_slots.cpu().tolist())

    @th.no_grad()
    def sync_partition(self, source):
        for name in ("important_slots", "score_ema", "score_initialized", "qchange_scores",
                     "tderror_scores", "grouping_updates"):
            getattr(self, name).copy_(getattr(source, name))
        if self.variant == "group_calibration":
            for name in ("removal_gain_scores", "original_mc_mse", "masked_mc_mse", "calibration_probe_count"):
                getattr(self, name).copy_(getattr(source, name))
