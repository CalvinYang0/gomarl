#!/usr/bin/env python3
"""Focused checks for the eight Counter gate ablations."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learners.clean_learner import CleanLearner  # noqa: E402
from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_DUAL_BRANCH_GATE_REGULARIZER_BY_MODEL,
    GRF_DUAL_BRANCH_GROUPED_PROPERTY_GATE_VARIANTS,
    GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL,
    GRF_DUAL_BRANCH_MASK_PARAMETER_RELATION_VARIANTS,
    GRF_DUAL_BRANCH_TRAIN_GATE_FREEZE_STEPS_BY_MODEL,
    GRF_DUAL_BRANCH_VARIANTS,
    GRFPublicPrivateBiasTransformerCapturer,
)


GROUPED = (
    "grf_abs_dual_branch_binary_concrete_"
    "grouped_property_param_stability_hypercond"
)
FREEZE = (
    "grf_abs_dual_branch_binary_concrete_"
    "temporal_param_stability_freeze2m_hypercond"
)
RELATION = (
    "grf_abs_dual_branch_binary_concrete_"
    "mask_parameter_relation_hypercond"
)
RELATION_TEMPORAL = (
    "grf_abs_dual_branch_binary_concrete_"
    "mask_parameter_relation_temporal_stability_hypercond"
)
RELATION_PERTURBED = (
    "grf_abs_dual_branch_binary_concrete_"
    "mask_parameter_relation_perturbed_head_hypercond"
)
KL90 = "grf_abs_dual_branch_binary_concrete_bayesg_kl90_hypercond"
KL80_RELATION_T70 = (
    "grf_abs_dual_branch_binary_concrete_"
    "bayesg_kl80_threshold70_relation_hypercond"
)
KL80_RELATION_KEEP = (
    "grf_abs_dual_branch_binary_concrete_"
    "bayesg_kl80_keep_relation_hypercond"
)


def build_grf(**kwargs):
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="dual",
        semantic_router_share_by_side=True,
        dynamic_branch_gate_mode="binary_concrete",
        dynamic_branch_gate_hidden_dim=16,
        dynamic_branch_gate_warmup_steps=250000,
        hard_gate_initial_keep_probability=0.95,
        **kwargs,
    )


def check_registration():
    variants = {
        GROUPED,
        FREEZE,
        RELATION,
        RELATION_TEMPORAL,
        RELATION_PERTURBED,
        KL90,
        KL80_RELATION_T70,
        KL80_RELATION_KEEP,
    }
    assert variants <= GRF_DUAL_BRANCH_VARIANTS
    assert GROUPED in GRF_DUAL_BRANCH_GROUPED_PROPERTY_GATE_VARIANTS
    assert GRF_DUAL_BRANCH_TRAIN_GATE_FREEZE_STEPS_BY_MODEL[FREEZE] == 2000000
    assert {
        RELATION,
        RELATION_TEMPORAL,
        RELATION_PERTURBED,
        KL80_RELATION_T70,
        KL80_RELATION_KEEP,
    } <= GRF_DUAL_BRANCH_MASK_PARAMETER_RELATION_VARIANTS
    assert GRF_DUAL_BRANCH_GATE_REGULARIZER_BY_MODEL[KL90] == (
        "bernoulli_kl",
        0.90,
    )
    for model in (KL80_RELATION_T70, KL80_RELATION_KEEP):
        assert GRF_DUAL_BRANCH_GATE_REGULARIZER_BY_MODEL[model] == (
            "bernoulli_kl",
            0.80,
        )
    assert GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL[KL80_RELATION_T70] == 0.70
    assert KL80_RELATION_KEEP not in GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL


def check_grouped_property_gate():
    model = build_grf(dynamic_branch_gate_group_properties=True)
    generator = model.dynamic_branch_gate
    assert generator.group_count < generator.obs_dim
    names = list(model.semantic_names)
    ids = generator.slot_group_ids.tolist()
    assert ids[names.index("opponent_0_direction_x")] == ids[
        names.index("opponent_1_direction_x")
    ]
    assert ids[names.index("ally_0_relative_y")] == ids[
        names.index("ally_2_relative_y")
    ]
    assert ids[names.index("opponent_0_direction_x")] != ids[
        names.index("opponent_0_direction_y")
    ]
    obs = th.randn(2, 4, 30)
    _, probabilities = generator(obs, sample=False)
    assert th.equal(
        probabilities[..., names.index("opponent_0_direction_x")],
        probabilities[..., names.index("opponent_1_direction_x")],
    )


def check_freeze_matches_evaluation():
    model = build_grf(dynamic_branch_gate_training_freeze_steps=2000000)
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    model.set_dynamic_branch_gate_t_env(2000000)
    train_condition, _ = model(obs, hidden)
    train_gates = model.latest_dynamic_branch_gates_graph.clone()
    assert th.all((train_gates == 0.0) | (train_gates == 1.0))
    assert model.latest_aux_stats["dynamic_gate_training_frozen"].item() == 1.0
    train_condition.sum().backward()
    gate_parameters = tuple(model.dynamic_branch_gate.parameters())
    assert all(parameter.grad is None for parameter in gate_parameters)

    model.zero_grad(set_to_none=True)
    model.set_semantic_test_mode(True)
    test_condition, _ = model(obs, hidden)
    test_gates = model.latest_dynamic_branch_gates_graph
    assert th.equal(train_gates, test_gates)
    assert th.allclose(train_condition, test_condition)


def check_relation_gradient_boundary():
    learner = CleanLearner.__new__(CleanLearner)
    learner.mask_parameter_relation_scale = 0.1
    learner.temporal_param_scale_eps = 1e-6
    batch_size, n_agents, slots = 2, 3, 5
    previous_parameter = th.randn(
        batch_size * n_agents, 4, requires_grad=True
    )
    current_parameter = th.randn(
        batch_size * n_agents, 4, requires_grad=True
    )
    previous_logits = th.randn(2, batch_size, n_agents, slots, requires_grad=True)
    current_logits = th.randn(2, batch_size, n_agents, slots, requires_grad=True)
    total, count, _, _ = learner._mask_parameter_relation_pair(
        (previous_parameter,),
        (current_parameter,),
        previous_logits.sigmoid(),
        current_logits.sigmoid(),
        th.tensor([True, False]),
        batch_size,
        n_agents,
    )
    (total / count.clamp(min=1.0)).backward()
    assert previous_logits.grad is not None
    assert current_logits.grad is not None
    assert previous_logits.grad.abs().sum().item() > 0.0
    assert current_logits.grad.abs().sum().item() > 0.0
    assert previous_parameter.grad is None
    assert current_parameter.grad is None


def main():
    th.manual_seed(23)
    check_registration()
    check_grouped_property_gate()
    check_freeze_matches_evaluation()
    check_relation_gradient_boundary()
    print("counter_mask_parameter_relation eight_variants=ok")


if __name__ == "__main__":
    main()
