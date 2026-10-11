"""Matched Linear hyper-head controls; grouping is global, never Obs-routed.

Only the condition path is masked. The shared policy GRU sees the original
Obs, previous action and agent ID. No KL, distillation or sparsity loss.
"""
from copy import copy, deepcopy

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .clean_hyper_agent import CleanHyperAgent, ObservationConditionedBranchGate


VARIANTS = {
    "smac_slot_static10_hypercond": "static10",
    "smac_slot_forward_hierarchy_hypercond": "forward_hierarchy",
    "smac_slot_forward_dualtd_hypercond": "forward_dualtd",
    "smac_slot_reverse_hierarchy_hypercond": "reverse_hierarchy",
}
BLOCKS = ("hyper_bottleneck_w", "hyper_bottleneck_b", "hyper_out_w", "hyper_out_b")


class SlotGroupHyperAgent(CleanHyperAgent):
    def __init__(self, input_shape, args):
        requested = args.clean_model_type
        if requested not in VARIANTS or args.env != "sc2":
            raise ValueError("Unsupported SMAC slot-input control: " + requested)
        matched = copy(args)
        matched.clean_model_type = "smac_single_linear_suite_baseline_hypercond"
        super().__init__(input_shape, matched)
        self.args, self.model_type = args, requested
        self.slot_variant = VARIANTS[requested]
        self.is_grouped = self.slot_variant != "static10"
        self.hard_gate_threshold = .1 if not self.is_grouped else .5
        self.register_buffer("important_slots", th.full((self.obs_dim,),
                             self.slot_variant == "reverse_hierarchy", dtype=th.bool))
        self.register_buffer("slot_score_ema", th.zeros(self.obs_dim))
        self.register_buffer("slot_flip_rate", th.zeros(self.obs_dim))
        self.register_buffer("slot_score_initialized", th.zeros(self.obs_dim, dtype=th.bool))
        self.register_buffer("grouping_updates", th.zeros((), dtype=th.long))
        self.slot_ema = float(getattr(args, "slot_importance_ema", .8))
        self.slot_demote_ratio = float(getattr(args, "slot_demote_ratio", .5))
        if not 0 <= self.slot_ema < 1 or not 0 < self.slot_demote_ratio < 1:
            raise ValueError("Invalid grouping EMA or demotion ratio")
        self.target_gate_soft = False
        if self.is_grouped:
            # Separate parameters, same architecture/initialization family.
            self.other_encoder = deepcopy(self.rpg_relation_capturer.dual_linear_encoder)
            self.other_encoder.reset_parameters()
            self.other_generators = nn.ModuleDict({k: deepcopy(getattr(self, k)) for k in BLOCKS})
            for layer in self.other_generators.values():
                layer.reset_parameters()
            if self.apply_hypermarl_init:
                nn.init.orthogonal_(self.other_generators[BLOCKS[0]].weight, gain=2 ** .5)
                nn.init.orthogonal_(self.other_generators[BLOCKS[2]].weight, gain=1.)
                for k in BLOCKS:
                    nn.init.zeros_(self.other_generators[k].bias)
                nn.init.zeros_(self.other_generators[BLOCKS[1]].weight)
                nn.init.zeros_(self.other_generators[BLOCKS[3]].weight)
            self.rpg_relation_capturer.relation_encoder_style = "grouped_linear"
        else:
            self.rpg_relation_capturer.dynamic_branch_gate = ObservationConditionedBranchGate(
                self.obs_dim, 0, "binary_concrete", initial_keep_probability=.1,
                hard_threshold=.1, binary_concrete_temperature=.5, gate_scope="shared",
                observation_independent=True)
            self.rpg_relation_capturer.counter_transformer_profile = {
                "branch": "linear", "gate": True, "static_gate": True}

    def group_masks(self, reference):
        return th.stack((self.important_slots, ~self.important_slots)).to(reference)

    def head_q(self, observation, hidden, branch=0):
        """Pure deterministic head evaluation, with an already masked condition."""
        encoder = self.rpg_relation_capturer.dual_linear_encoder if branch == 0 else self.other_encoder
        generators = {k: getattr(self, k) for k in BLOCKS} if branch == 0 else self.other_generators
        condition = encoder(observation)
        shape = hidden.shape[:-1]
        c, h = condition.reshape(-1, self.cond_dim), hidden.reshape(-1, 1, self.hidden_dim)
        parameters = tuple(generators[k](c) for k in BLOCKS)
        w1, b1, w2, b2 = parameters
        q = th.bmm(F.elu(th.bmm(h, w1.reshape(-1, self.hidden_dim, self.hidden_dim))
                        + b1.unsqueeze(1)), w2.reshape(-1, self.hidden_dim, self.n_actions)) + b2.unsqueeze(1)
        return q.reshape(*shape, self.n_actions), condition, parameters

    def forward(self, inputs, hidden_state, context=None, test_mode=False):
        if context is None:
            raise ValueError("Slot controls require clean local Obs context")
        if hidden_state is None:
            hidden_state = self.init_hidden().expand(*inputs.shape[:-1], -1)
        override = context.get("policy_hidden_override")
        hidden = override if override is not None else self.rnn(
            F.relu(self.fc1(inputs.reshape(-1, inputs.shape[-1]))),
            hidden_state.reshape(-1, self.hidden_dim)).reshape(*inputs.shape[:-1], self.hidden_dim)
        obs = context["obs"]
        self.latest_aux_loss, self.latest_aux_stats = None, {}
        if self.is_grouped:
            masks = self.group_masks(obs)
            conditions = obs.unsqueeze(-3) * masks.view(2, 1, self.obs_dim)
            first, condition, parameters = self.head_q(conditions[..., 0, :, :], hidden, 0)
            second, _, other_parameters = self.head_q(conditions[..., 1, :, :], hidden, 1)
            self.latest_group_q = th.stack((first, second), dim=-3)
            self.latest_group_observations = conditions
            gates = masks[:, None, None, :].expand(2, *obs.shape)
            probabilities = gates
            parameters += other_parameters
        else:
            gate = self.rpg_relation_capturer.dynamic_branch_gate
            gates, probabilities = gate(obs, sample=not test_mode,
                                         deterministic_soft=self.target_gate_soft and not test_mode)
            first, condition, parameters = self.head_q(obs * gates[0], hidden)
            self.latest_group_q = self.latest_group_observations = None
        self.latest_condition = condition.detach()
        self.latest_condition_graph = condition
        self.latest_generated_parameter_graph = parameters
        self.latest_policy_hidden_graph = hidden
        self.latest_dynamic_branch_gates_graph = gates
        self.latest_dynamic_branch_probabilities_graph = probabilities
        self.latest_dynamic_branch_logits_graph = None
        return first, hidden

    @th.no_grad()
    def apply_slot_scores(self, scores, flips):
        """One atomic partition change. No parameter/optimizer reset or transfer."""
        if scores.shape != self.slot_score_ema.shape or not th.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Invalid slot importance scores")
        if flips.shape != scores.shape or not th.isfinite(flips).all():
            raise ValueError("Invalid slot flip rates")
        before = self.important_slots.clone()
        # Scores are measured by the owning head; restart a slot's EMA when
        # it changes owner rather than mixing two different head estimands.
        smoothed = self.slot_ema * self.slot_score_ema + (1 - self.slot_ema) * scores
        self.slot_score_ema.copy_(th.where(self.slot_score_initialized, smoothed, scores))
        self.slot_flip_rate.copy_(flips)
        self.slot_score_initialized.fill_(True)
        promoted = demoted = None
        important, other = before.nonzero().flatten(), (~before).nonzero().flatten()
        if self.slot_variant != "reverse_hierarchy" and other.numel():
            candidate = other[self.slot_score_ema[other].argmax()]
            if self.slot_score_ema[candidate] > 1e-8:
                promoted = int(candidate)
        if important.numel() > 1:
            top = self.slot_score_ema[important].max()
            last = important[self.slot_score_ema[important].argmin()]
            if top > 1e-8 and self.slot_score_ema[last] < self.slot_demote_ratio * top:
                demoted = int(last)
        if promoted is not None:
            self.important_slots[promoted] = True
        if demoted is not None:
            self.important_slots[demoted] = False
        self.slot_score_initialized[before != self.important_slots] = False
        self.grouping_updates.add_(1)
        return dict(promoted=promoted, demoted=demoted,
                    important_before=before.cpu().tolist(), important_after=self.important_slots.cpu().tolist())
