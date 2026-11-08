"""Separate, opt-in gate parameterizations; historical sigmoid remains unchanged."""
import copy
import math

import torch as th
import torch.nn as nn

from .soft_onefill_gate import SoftOneFillGateMixer


class ProjectedOneFillGateMixer(SoftOneFillGateMixer):
    def __init__(self, args):
        initial = float(args.mixer_input_initial_probability)
        if not math.isfinite(initial) or not 0 < initial <= 1:
            raise ValueError('Projected gate initial value must be in (0,1]')
        initial_args = copy.copy(args)
        initial_args.mixer_input_initial_probability = 0.5
        super().__init__(initial_args)
        self.base.args = args
        del self.gate_logits
        self.gate_parameter = nn.Parameter(th.full((self.state_dim,), initial))

    def coefficients(self):
        # No sigmoid/clamp in the forward: boundary coefficients retain gradients.
        return self.gate_parameter

    @th.no_grad()
    def project(self):
        if not th.isfinite(self.gate_parameter).all():
            raise RuntimeError('Non-finite projected gate after optimizer update')
        self.gate_parameter.clamp_(0, 1)


class ProjectedFreezeOneFillGateMixer(ProjectedOneFillGateMixer):
    """Keep the direct coefficient as the fixed mask after the one-time freeze."""

    def __init__(self, args):
        super().__init__(args)
        self.register_buffer('freeze_t_env', th.tensor(-1, dtype=th.long))

    @th.no_grad()
    def freeze(self, mask, t_env):
        if self.freeze_t_env.item() >= 0:
            raise RuntimeError('Projected gate already frozen')
        if mask.shape != self.gate_parameter.shape or not th.all((mask == 0) | (mask == 1)):
            raise ValueError('Freeze mask must be a same-shape binary vector')
        self.gate_parameter.copy_(mask)
        self.freeze_t_env.fill_(int(t_env))
        self.restore_trainability()

    def restore_trainability(self):
        self.gate_parameter.requires_grad_(self.freeze_t_env.item() < 0)
        self.gate_parameter.grad = None

    def freeze_metadata(self):
        t_env = int(self.freeze_t_env.item())
        return dict(frozen=t_env >= 0, t_env=t_env,
            open_slots=int(self.gate_parameter.sum().item()) if t_env >= 0 else None)


class WarmupFreezeOneFillGateMixer(SoftOneFillGateMixer):
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer('gate_frozen', th.tensor(False))
        self.register_buffer('frozen_mask', th.zeros(self.state_dim))
        self.register_buffer('freeze_t_env', th.tensor(-1, dtype=th.long))

    def coefficients(self):
        return self.frozen_mask if self.gate_frozen.item() else super().coefficients()

    @th.no_grad()
    def freeze(self, mask, t_env):
        if mask.shape != self.frozen_mask.shape or not th.all((mask == 0) | (mask == 1)):
            raise ValueError('Freeze mask must be a same-shape binary vector')
        self.frozen_mask.copy_(mask)
        self.gate_frozen.fill_(True)
        self.freeze_t_env.fill_(int(t_env))
        self.restore_trainability()

    def restore_trainability(self):
        self.gate_logits.requires_grad_(not bool(self.gate_frozen.item()))
        self.gate_logits.grad = None

    def freeze_metadata(self):
        return dict(frozen=bool(self.gate_frozen.item()), t_env=int(self.freeze_t_env.item()),
            open_slots=int(self.frozen_mask.sum().item()) if self.gate_frozen.item() else None)
