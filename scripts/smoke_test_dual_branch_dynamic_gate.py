#!/usr/bin/env python3
"""Tensor checks for observation-conditioned c-STG and BayesG dual gates."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRFPublicPrivateBiasTransformerCapturer,
    PublicTransformerRelationCapturer,
)


def build_smac(gate_mode):
    return PublicTransformerRelationCapturer(
        move_dim=4,
        own_dim=1,
        ally_feat_dim=5,
        enemy_feat_dim=5,
        relation_dim=16,
        output_dim=16,
        obs_all_health=True,
        obs_own_health=True,
        n_allies=4,
        n_enemies=6,
        mode="full_obs",
        num_heads=4,
        relation_encoder_style="dual",
        semantic_router_share_by_side=True,
        dynamic_branch_gate_mode=gate_mode,
        dynamic_branch_gate_hidden_dim=16,
    )


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
    )


def check_gate_module(model, obs):
    generator = model.dynamic_branch_gate
    assert generator is not None
    model.set_semantic_test_mode(True)
    eval_gate_1, eval_prob_1 = generator(obs, sample=False)
    eval_gate_2, eval_prob_2 = generator(obs, sample=False)
    expected_shape = (2,) + tuple(obs.shape)
    assert eval_gate_1.shape == expected_shape
    assert eval_prob_1.shape == expected_shape
    assert th.equal(eval_gate_1, eval_gate_2)
    assert th.equal(eval_prob_1, eval_prob_2)

    train_gate_1, _ = generator(obs, sample=True)
    train_gate_2, _ = generator(obs, sample=True)
    assert not th.equal(train_gate_1, train_gate_2)
    assert train_gate_1.min().item() >= 0.0
    assert train_gate_1.max().item() <= 1.0
    model.set_semantic_test_mode(False)


def check_smac(gate_mode):
    model = build_smac(gate_mode)
    batch_size, n_agents = 2, 5
    self_feat = th.randn(batch_size, n_agents, 5)
    ally_feat = th.randn(batch_size, n_agents, 4, 5)
    enemy_feat = th.randn(batch_size, n_agents, 6, 5)
    hidden = th.zeros(batch_size, n_agents, 16)
    flat_obs = model._flatten_semantic_slots(self_feat, ally_feat, enemy_feat)
    check_gate_module(model, flat_obs)

    condition, next_hidden = model(
        self_feat, ally_feat, enemy_feat, hidden
    )[:2]
    assert condition.shape == (batch_size, n_agents, 16)
    assert next_hidden.shape == (batch_size, n_agents, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None
    assert "dynamic_gate_linear_mean" in model.latest_aux_stats


def check_grf(gate_mode):
    model = build_grf(gate_mode)
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    check_gate_module(model, obs)

    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None
    assert "dynamic_gate_attention_mean" in model.latest_aux_stats


def main():
    th.manual_seed(17)
    for gate_mode in ("cstg", "bayesg"):
        check_smac(gate_mode)
        check_grf(gate_mode)
        print(
            "dual_branch dynamic_gate={} per_timestep_forward_backward=ok".format(
                gate_mode
            )
        )


if __name__ == "__main__":
    main()
