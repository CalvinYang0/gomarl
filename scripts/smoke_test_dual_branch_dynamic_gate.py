#!/usr/bin/env python3
"""Tensor checks for corrected scalar hard gates and Transformer-only control."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    GRF_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS,
    GRF_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS,
    GRF_DUAL_BRANCH_SPLIT_HEAD_VARIANTS,
    GRF_DUAL_BRANCH_VARIANTS,
    GRFPublicPrivateBiasTransformerCapturer,
    RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    RPG_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS,
    RPG_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS,
    RPG_DUAL_BRANCH_SPLIT_HEAD_VARIANTS,
    RPG_DUAL_BRANCH_VARIANTS,
)
from learners.clean_learner import CleanLearner  # noqa: E402


def build_grf(gate_mode, gate_scope="both"):
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="dual",
        semantic_router_share_by_side=True,
        dynamic_branch_gate_mode=gate_mode,
        dynamic_branch_gate_hidden_dim=16,
        dynamic_branch_gate_warmup_steps=250000,
        dynamic_branch_gate_scope=gate_scope,
    )


def build_grf_transformer_only():
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="attention_only",
    )


def check_gate_module(model, obs):
    generator = model.dynamic_branch_gate
    assert generator is not None
    eval_gate_1, eval_prob_1 = generator(obs, sample=False)
    expected_shape = (2,) + tuple(obs.shape)
    assert eval_gate_1.shape == expected_shape
    assert eval_prob_1.shape == expected_shape
    assert th.all((eval_gate_1 == 0.0) | (eval_gate_1 == 1.0))
    if generator.mode == "binary_concrete":
        train_gate_1, train_prob_1 = generator(obs, sample=True)
        train_gate_2, train_prob_2 = generator(obs, sample=True)
        assert th.equal(eval_prob_1, train_prob_1)
        assert th.equal(train_prob_1, train_prob_2)
        assert not th.equal(train_gate_1, train_gate_2)
        assert th.any((train_gate_1 > 0.0) & (train_gate_1 < 1.0))
        with th.no_grad():
            rollout_gate_1, _ = generator(obs, sample=True)
            rollout_gate_2, _ = generator(obs, sample=True)
            target_gate, target_prob = generator(
                obs, sample=True, deterministic_soft=True
            )
        assert not th.equal(rollout_gate_1, rollout_gate_2)
        assert th.equal(target_gate, target_prob)
    else:
        eval_gate_2, eval_prob_2 = generator(obs, sample=True)
        assert th.equal(eval_gate_1, eval_gate_2)
        assert th.equal(eval_prob_1, eval_prob_2)
    final_layer = generator.gate_network[-1]
    expected_out_features = (
        generator.obs_dim
        if generator.gate_scope == "shared"
        else 2 * generator.obs_dim
    )
    assert final_layer.out_features == expected_out_features
    if generator.gate_scope == "shared":
        shared_gates, _ = generator(obs, sample=False)
        assert th.equal(shared_gates[0], shared_gates[1])
        return
    with th.no_grad():
        saved_bias = final_layer.bias.clone()
        final_layer.bias[0] = -1.0
        final_layer.bias[1] = 1.0
        independent_gates, _ = generator(obs, sample=False)
        final_layer.bias.copy_(saved_bias)
    assert th.all(independent_gates[0, ..., 0] == 0.0)
    assert th.all(independent_gates[0, ..., 1] == 1.0)


def check_grf(gate_mode):
    model = build_grf(gate_mode)
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    check_gate_module(model, obs)

    model.set_dynamic_branch_gate_t_env(250000)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None
    assert (
        model.dynamic_branch_gate.gate_network[-1].weight.grad.abs().sum().item()
        > 0.0
    )
    assert "dynamic_gate_attention_mean" in model.latest_aux_stats
    assert model.latest_aux_stats["dynamic_gate_warmup_active"].item() == 0.0

    model.zero_grad(set_to_none=True)
    model.set_dynamic_branch_gate_t_env(249999)
    model(obs, hidden)
    assert model.latest_aux_stats["dynamic_gate_warmup_active"].item() == 1.0
    assert model.latest_aux_stats["dynamic_gate_linear_mean"].item() == 1.0
    assert model.latest_aux_stats["dynamic_gate_attention_mean"].item() == 1.0


def check_grf_transformer_only():
    model = build_grf_transformer_only()
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    assert model.dynamic_branch_gate is None
    assert model.dual_linear_encoder is None
    assert model.dual_condition_fuser is None
    (condition.mean() + next_hidden.mean()).backward()
    assert model.transformer_layers[0].self_attn.qkv.weight.grad is not None


def check_shared_slot_gate():
    model = build_grf("binary_concrete", gate_scope="shared")
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    check_gate_module(model, obs)
    model.set_dynamic_branch_gate_t_env(250000)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None


def check_condition_gradient_consistency():
    model_name = "grf_abs_dual_branch_hard_gate_grad_consistency_hypercond"
    assert model_name in GRF_DUAL_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"

    learner = CleanLearner.__new__(CleanLearner)
    learner.condition_gradient_consistency_pairs = 2
    source = th.randn(3, 4, 2, 5, requires_grad=True)
    conditions = [source[:, time_index] for time_index in range(4)]
    predictions = th.stack(
        [
            (condition.pow(2).sum(dim=-1)).sum(dim=-1, keepdim=True)
            for condition in conditions
        ],
        dim=1,
    )
    td_error = predictions - 1.0
    td_mask = th.ones_like(td_error)
    loss, stats = learner._condition_gradient_consistency_loss(
        td_error, td_mask, conditions
    )
    assert th.isfinite(loss)
    assert stats["condition_grad_consistency_pairs"].item() == 2.0
    loss.backward()
    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0


def check_generated_parameter_stability():
    model_name = "grf_abs_dual_branch_hard_gate_param_stability_hypercond"
    assert model_name in GRF_DUAL_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"

    batch_size = 3
    n_agents = 2
    source = th.randn(batch_size * n_agents, 7, requires_grad=True)
    previous = (source[:, :4] * 0.5, source[:, 4:] * 0.25)
    current = (source[:, :4], source[:, 4:])
    pair_valid = th.tensor([True, False, True])
    total, count = CleanLearner._generated_parameter_stability_pair(
        previous, current, pair_valid, batch_size, n_agents
    )
    loss = total / count.clamp(min=1.0)
    assert th.isfinite(loss)
    assert count.item() == 4.0
    loss.backward()
    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0

    corridor_model_name = "rpg_dual_branch_hard_gate_param_stability_hypercond"
    assert corridor_model_name in RPG_DUAL_BRANCH_VARIANTS
    assert (
        RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[corridor_model_name]
        == "hard_st"
    )


def check_corridor_gradient_consistency_registration():
    model_name = "rpg_dual_branch_hard_gate_grad_consistency_hypercond"
    assert model_name in RPG_DUAL_BRANCH_VARIANTS
    assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"


def check_new_variant_registration():
    for prefix, variants, modes in (
        ("grf_abs", GRF_DUAL_BRANCH_VARIANTS, GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
        ("rpg", RPG_DUAL_BRANCH_VARIANTS, RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
    ):
        for auxiliary in ("param_stability", "grad_consistency"):
            concrete = f"{prefix}_dual_branch_binary_concrete_{auxiliary}_hypercond"
            adaptive = f"{prefix}_dual_branch_hard_gate_adaptive_{auxiliary}_hypercond"
            assert concrete in variants
            assert modes[concrete] == "binary_concrete"
            assert adaptive in variants
            assert modes[adaptive] == "hard_st"

    for auxiliary in ("param_stability", "grad_consistency"):
        attention_only = (
            f"rpg_dual_branch_attention_only_hard_gate_{auxiliary}_hypercond"
        )
        split_head = f"rpg_dual_branch_split_head_hard_gate_{auxiliary}_hypercond"
        assert attention_only in RPG_DUAL_BRANCH_VARIANTS
        assert attention_only in RPG_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS
        assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[attention_only] == "hard_st"
        assert split_head in RPG_DUAL_BRANCH_VARIANTS
        assert split_head in RPG_DUAL_BRANCH_SPLIT_HEAD_VARIANTS
        assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[split_head] == "hard_st"

    for prefix, variants, modes in (
        ("grf_abs", GRF_DUAL_BRANCH_VARIANTS, GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
        ("rpg", RPG_DUAL_BRANCH_VARIANTS, RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
    ):
        base = f"{prefix}_dual_branch_binary_concrete_adaptive"
        for suffix in ("", "_slot", "_attention_only", "_split_head"):
            model_name = f"{base}{suffix}_grad_consistency_hypercond"
            assert model_name in variants
            assert modes[model_name] == "binary_concrete"

    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS
    )
    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS
    )
    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_SPLIT_HEAD_VARIANTS
    )
    assert (
        "rpg_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond"
        in RPG_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS
    )


def check_adaptive_auxiliary_ratio_detached():
    learner = CleanLearner.__new__(CleanLearner)
    learner.adaptive_auxiliary_target_ratio = 0.1
    learner.adaptive_auxiliary_ema_decay = 0.9
    learner.adaptive_auxiliary_eps = 1e-8
    learner.adaptive_auxiliary_max_coef = 100.0
    learner.adaptive_auxiliary_ema_td = None
    learner.adaptive_auxiliary_ema_aux = None
    learner.latest_adaptive_auxiliary_stats = {}

    td_source = th.tensor(2.0, requires_grad=True)
    aux_source = th.tensor(0.5, requires_grad=True)
    td_loss = td_source.square()
    auxiliary_loss = aux_source.square()
    coefficient = learner._adaptive_auxiliary_coefficient(td_loss, auxiliary_loss)
    assert isinstance(coefficient, float)
    assert abs(coefficient * auxiliary_loss.item() / td_loss.item() - 0.1) < 1e-6
    total = td_loss + coefficient * auxiliary_loss
    total.backward()
    assert abs(td_source.grad.item() - 4.0) < 1e-6
    assert abs(aux_source.grad.item() - coefficient) < 1e-6


def main():
    th.manual_seed(17)
    check_grf("hard_st")
    print("dual_branch hard_st scalar_forward_backward=ok")
    check_grf("binary_concrete")
    print("dual_branch binary_concrete stochastic_recovery=ok")
    check_grf_transformer_only()
    print("single_transformer_branch forward_backward=ok")
    check_shared_slot_gate()
    print("shared_slot_binary_concrete forward_backward=ok")
    check_condition_gradient_consistency()
    print("condition_gradient_consistency second_order=ok")
    check_generated_parameter_stability()
    print("generated_parameter_stability exact_l1=ok")
    check_corridor_gradient_consistency_registration()
    print("corridor_stability_variants registration=ok")
    check_new_variant_registration()
    print("new_gate_and_branch_role_variants registration=ok")
    check_adaptive_auxiliary_ratio_detached()
    print("adaptive_auxiliary detached_ema_ratio=ok")


if __name__ == "__main__":
    main()
