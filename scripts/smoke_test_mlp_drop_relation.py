#!/usr/bin/env python3
"""Exercise the lightweight MLP relation/drop paths before cluster submission."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRFPublicPrivateBiasTransformerCapturer,
    PublicTransformerRelationCapturer,
)


def check_smac(
    style,
    router_mode=None,
    learnable_threshold=False,
    l0_drop=False,
    soft_gate=False,
    independent_audit=False,
    binary_audit_mode=None,
    share_by_side=False,
    update_interval=0,
):
    capturer = PublicTransformerRelationCapturer(
        move_dim=4,
        own_dim=1,
        ally_feat_dim=5,
        enemy_feat_dim=5,
        relation_dim=16,
        output_dim=12,
        obs_own_health=True,
        obs_all_health=True,
        n_actions=10,
        n_allies=2,
        n_enemies=3,
        semantic_router_mode=router_mode,
        semantic_router_learnable_threshold=learnable_threshold,
        relation_encoder_style=style,
        l0_drop=l0_drop,
        mlp_soft_gate=soft_gate,
        mlp_independent_audit=independent_audit,
        mlp_binary_audit_mode=binary_audit_mode,
        semantic_router_share_fields=share_by_side,
        semantic_router_share_by_side=share_by_side,
        semantic_router_update_interval=update_interval,
        semantic_router_ema_up=0.5,
        semantic_router_ema_down=0.99,
    )
    capturer.train()
    self_feat = th.randn(2, 4, 5)
    ally_feat = th.randn(2, 4, 2, 5)
    enemy_feat = th.randn(2, 4, 3, 5)
    condition, hidden, _, _, enemy_tokens, enemy_mask = capturer(
        self_feat, ally_feat, enemy_feat, None
    )
    assert condition.shape == (2, 4, 12)
    assert hidden.shape == (2, 4, 16)
    assert enemy_tokens.shape == (2, 4, 3, 16)
    assert enemy_mask.shape == (2, 4, 3)
    loss = condition.square().mean()
    if capturer.latest_aux_loss is not None:
        loss = loss + capturer.latest_aux_loss
    loss.backward()
    if router_mode == "gradient_importance" and not independent_audit:
        assert capturer.semantic_probe_scale.grad is not None
    if independent_audit:
        assert capturer.semantic_probe_scale.grad is None
        capturer.zero_grad(set_to_none=True)
        capturer.set_semantic_full_input_audit(True)
        audit_condition, *_ = capturer(
            self_feat, ally_feat, enemy_feat, None
        )
        audit_condition.square().mean().backward()
        capturer.set_semantic_full_input_audit(False)
        assert capturer.semantic_probe_scale.grad is not None
    if l0_drop:
        assert capturer.l0_log_alpha.grad is not None
    if share_by_side:
        name_to_group = {
            name: int(group)
            for name, group in zip(
                capturer.semantic_names, capturer.semantic_field_ids.tolist()
            )
        }
        assert name_to_group["ally_0_health"] == name_to_group["ally_1_health"]
        assert name_to_group["enemy_0_health"] == name_to_group["enemy_2_health"]
        assert name_to_group["ally_0_health"] != name_to_group["enemy_0_health"]
        capturer.set_semantic_full_input_audit(True)
        dropped_group = name_to_group["enemy_0_health"]
        capturer.set_semantic_binary_audit_group(dropped_group)
        gate = capturer._mlp_relation_gate(self_feat)
        assert gate[capturer.semantic_names.index("enemy_0_health")] == 0
        assert gate[capturer.semantic_names.index("enemy_2_health")] == 0
        assert gate[capturer.semantic_names.index("ally_0_health")] == 1
        capturer.set_semantic_full_input_audit(False)


def check_grf(
    style,
    router_mode=None,
    learnable_threshold=False,
    l0_drop=False,
    soft_gate=False,
    independent_audit=False,
    binary_audit_mode=None,
    share_by_side=False,
    update_interval=0,
):
    capturer = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=3,
        relation_dim=16,
        output_dim=12,
        use_absolute_public=True,
        semantic_router_mode=router_mode,
        semantic_router_learnable_threshold=learnable_threshold,
        relation_encoder_style=style,
        l0_drop=l0_drop,
        mlp_soft_gate=soft_gate,
        mlp_independent_audit=independent_audit,
        mlp_binary_audit_mode=binary_audit_mode,
        semantic_router_share_fields=share_by_side,
        semantic_router_share_by_side=share_by_side,
        semantic_router_update_interval=update_interval,
        semantic_router_ema_up=0.5,
        semantic_router_ema_down=0.99,
    )
    capturer.train()
    obs = th.randn(2, 3, capturer.expected_obs_dim)
    condition, hidden = capturer(obs, None)
    assert condition.shape == (2, 3, 12)
    assert hidden.shape == (2, 3, 16)
    assert capturer.latest_self_token.shape == (2, 3, 16)
    loss = condition.square().mean()
    if capturer.latest_aux_loss is not None:
        loss = loss + capturer.latest_aux_loss
    loss.backward()
    if router_mode == "gradient_importance" and not independent_audit:
        assert capturer.semantic_probe_scale.grad is not None
    if independent_audit:
        assert capturer.semantic_probe_scale.grad is None
        capturer.zero_grad(set_to_none=True)
        capturer.set_semantic_full_input_audit(True)
        audit_condition, _ = capturer(obs, None)
        audit_condition.square().mean().backward()
        capturer.set_semantic_full_input_audit(False)
        assert capturer.semantic_probe_scale.grad is not None
    if l0_drop:
        assert capturer.l0_log_alpha.grad is not None
    if share_by_side:
        name_to_group = {
            name: int(group)
            for name, group in zip(
                capturer.semantic_names, capturer.semantic_field_ids.tolist()
            )
        }
        assert (
            name_to_group["ally_0_relative_x"]
            == name_to_group["ally_1_relative_x"]
        )
        assert (
            name_to_group["opponent_0_relative_x"]
            == name_to_group["opponent_1_relative_x"]
        )
        assert (
            name_to_group["ally_0_relative_x"]
            != name_to_group["opponent_0_relative_x"]
        )


def main():
    variants = (
        dict(style="mlp"),
        dict(
            style="mlp",
            router_mode="gradient_importance",
            learnable_threshold=True,
        ),
        dict(
            style="mlp",
            router_mode="gradient_importance",
            learnable_threshold=True,
            soft_gate=True,
            update_interval=8000,
        ),
        dict(
            style="mlp",
            router_mode="gradient_importance",
            independent_audit=True,
            update_interval=8000,
        ),
        dict(
            style="mlp",
            router_mode="binary_td_audit",
            binary_audit_mode="td_loss",
            share_by_side=True,
            update_interval=8000,
        ),
        dict(
            style="mlp",
            router_mode="binary_parameter_audit",
            binary_audit_mode="generated_parameters",
            share_by_side=True,
            update_interval=8000,
        ),
        dict(style="mlp", l0_drop=True),
    )
    for variant in variants:
        check_smac(**variant)
        check_grf(**variant)
    print("MLP relation/drop smoke test passed for SMAC and GRF.")


if __name__ == "__main__":
    main()
