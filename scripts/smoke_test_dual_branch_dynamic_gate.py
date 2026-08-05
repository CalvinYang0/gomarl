#!/usr/bin/env python3
"""Tensor checks for corrected scalar hard gates and Transformer-only control."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    GRF_DUAL_BRANCH_VARIANTS,
    GRFPublicPrivateBiasTransformerCapturer,
    RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    RPG_DUAL_BRANCH_VARIANTS,
)
from learners.clean_learner import CleanLearner  # noqa: E402


def build_grf(gate_mode):
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
    eval_gate_2, eval_prob_2 = generator(obs, sample=True)
    expected_shape = (2,) + tuple(obs.shape)
    assert eval_gate_1.shape == expected_shape
    assert eval_prob_1.shape == expected_shape
    assert th.equal(eval_gate_1, eval_gate_2)
    assert th.equal(eval_prob_1, eval_prob_2)
    assert th.all((eval_gate_1 == 0.0) | (eval_gate_1 == 1.0))
    final_layer = generator.gate_network[-1]
    assert final_layer.out_features == 2 * generator.obs_dim
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


def main():
    th.manual_seed(17)
    check_grf("hard_st")
    print("dual_branch hard_st scalar_forward_backward=ok")
    check_grf_transformer_only()
    print("single_transformer_branch forward_backward=ok")
    check_condition_gradient_consistency()
    print("condition_gradient_consistency second_order=ok")
    check_generated_parameter_stability()
    print("generated_parameter_stability exact_l1=ok")
    check_corridor_gradient_consistency_registration()
    print("corridor_stability_variants registration=ok")


if __name__ == "__main__":
    main()
