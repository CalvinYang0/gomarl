#!/usr/bin/env python3
"""Tensor checks for full, TD-benefit, and hyper-output dual branches."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRFPublicPrivateBiasTransformerCapturer,
    PublicTransformerRelationCapturer,
)


def build_smac(drop_mode):
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
        branch_drop_mode=drop_mode,
        branch_drop_warmup_steps=0,
        branch_drop_freeze_steps=10_000_000,
    )


def build_grf(drop_mode):
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="dual",
        semantic_router_share_by_side=True,
        branch_drop_mode=drop_mode,
        branch_drop_warmup_steps=0,
        branch_drop_freeze_steps=10_000_000,
    )


def check_smac(drop_mode):
    model = build_smac(drop_mode)
    batch_size, n_agents = 2, 5
    self_feat = th.randn(batch_size, n_agents, 5)
    ally_feat = th.randn(batch_size, n_agents, 4, 5)
    enemy_feat = th.randn(batch_size, n_agents, 6, 5)
    hidden = th.zeros(batch_size, n_agents, 16)
    condition, next_hidden = model(
        self_feat, ally_feat, enemy_feat, hidden
    )[:2]
    assert condition.shape == (batch_size, n_agents, 16)
    assert next_hidden.shape == (batch_size, n_agents, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dual_condition_fuser.weight.grad is not None

    if drop_mode is not None:
        scores = th.full((2, model.semantic_field_count), 1.0)
        if drop_mode == "td_benefit":
            scores[0, 0] = -1.0
        else:
            scores[0, 0] = 0.0
        assert model.update_branch_drop(1, scores)
        assert model.branch_group_keep_state()[0, 0].item() == 0.0
        model.set_branch_drop_audit(0, 0, keep=True)
        assert model._branch_keep_gates(self_feat)[0].max().item() == 1.0
        model.set_branch_drop_audit()


def check_grf(drop_mode):
    model = build_grf(drop_mode)
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dual_condition_fuser.weight.grad is not None

    if drop_mode is not None:
        scores = th.full((2, model.semantic_field_count), 1.0)
        if drop_mode == "td_benefit":
            scores[1, 0] = -1.0
        else:
            scores[1, 0] = 0.0
        assert model.update_branch_drop(1, scores)
        assert model.branch_group_keep_state()[1, 0].item() == 0.0


def main():
    th.manual_seed(13)
    for mode in (None, "td_benefit", "generated_parameters"):
        check_smac(mode)
        check_grf(mode)
        print("dual_branch mode={} forward_backward_drop=ok".format(mode))


if __name__ == "__main__":
    main()
