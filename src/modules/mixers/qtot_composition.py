"""Two complete QMIX evaluations of the SAME agent Q vector, then Q_tot sum.

Unlike weight_composition this combines neither hypernetwork outputs nor raw
weights. Both branches have their own W1/W2/b1/b2. No abs is applied to Q_tot.
"""
import math

import torch as th
import torch.nn as nn

from .onefill_gate import OneFillGateMixer
from .qmix import QMixer


class QTotCompositionMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if (args.mixer_onefill_residual or getattr(args, "mixer_residual_combination", "centered") != "centered"
                or getattr(args, "mixer_residual_secondary_ones", False)
                or getattr(args, "mixer_residual_fill_value", 1) != 1):
            raise ValueError("Complete Q_tot composition must not silently enable weight composition")
        # Reuse EXACT historical gate and main-branch initialization. The new
        # complete secondary QMIX is independently default-initialized: zero
        # raw W heads would deadlock at abs(0) and are intentionally not used.
        template = OneFillGateMixer(args)
        for key in ("state_dim", "n_agents", "grouped", "temperature", "fill_value", "residual_scale"):
            setattr(self, key, getattr(template, key))
        self.gate_logits, self.base = template.gate_logits, template.base
        self.secondary = QMixer(args)
        self.residual = False  # It is NOT the historical weight-residual architecture.
        self.output_combination = str(args.mixer_qtot_combination)
        if self.output_combination not in {"sum", "alpha_sum"}:
            raise ValueError("Q_tot combination must be sum or alpha_sum")
        self.combination = "qtot_" + self.output_combination
        self.secondary_ones = bool(args.mixer_qtot_secondary_ones)
        if self.secondary_ones and self.output_combination != "sum":
            raise ValueError("Both-ones control is defined for direct Q_tot sum")
        if self.secondary_ones:
            self.gate_logits.requires_grad_(False)
        self.alpha_logit = None
        if self.output_combination == "alpha_sum":
            initial = float(args.mixer_qtot_alpha_initial)
            if not 0 < initial < 1:
                raise ValueError("Initial main alpha must be strictly between zero and one")
            self.alpha_logit = nn.Parameter(th.tensor(math.log(initial / (1 - initial))))
        self.latest_output_components = None

    def sample_mask(self, states, mode="train"):
        return OneFillGateMixer.sample_mask(self, states, mode)

    def controlled_input(self, states, mask):
        return OneFillGateMixer.controlled_input(self, states, mask)

    def alpha(self):
        return OneFillGateMixer.alpha(self)

    def output_components(self, agent_qs, states, mask):
        ones = th.ones_like(states)
        main = self.base(agent_qs, ones)
        secondary = self.secondary(agent_qs, self.controlled_input(states, mask))
        if self.output_combination == "sum":
            main_contribution, secondary_contribution = main, secondary
        else:
            alpha = self.alpha()
            main_contribution, secondary_contribution = alpha * main, (1 - alpha) * secondary
        return dict(main=main, secondary=secondary, main_contribution=main_contribution,
                    secondary_contribution=secondary_contribution,
                    combined=main_contribution + secondary_contribution)

    def conditioned_value(self, agent_qs, states, mask):
        parts = self.output_components(agent_qs, states, mask)
        self.latest_output_components = {key: value.detach() for key, value in parts.items()}
        return parts["combined"]

    def forward(self, agent_qs, states, mode="eval", detach_gate=False):
        mask = self.sample_mask(states, mode)
        return self.conditioned_value(agent_qs, states, mask.detach() if detach_gate else mask)

    def diagnostic_output_components(self, agent_qs, states):
        # Soft-probability replay diagnostic, NOT the mean of stochastic outputs.
        return self.output_components(agent_qs, states, self.sample_mask(states, "eval"))
