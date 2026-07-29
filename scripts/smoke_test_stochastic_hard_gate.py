#!/usr/bin/env python3
"""Validate stochastic hard semantic routing for SMAC and GRF."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_MLP_GIMP_STOCHASTIC_HARD_VARIANTS,
    GRFPublicPrivateBiasTransformerCapturer,
    MLP_GIMP_STOCHASTIC_HARD_VARIANTS,
    PublicTransformerRelationCapturer,
)


def _check_route(capturer):
    th.manual_seed(7)
    scores = th.linspace(0.1, 2.0, len(capturer.semantic_names))
    capturer._apply_semantic_route_scores(scores, t_env=0)

    relative = scores / scores.mean()
    expected = 0.05 + 0.95 * th.sigmoid((relative - 1.0) / 0.2)
    assert th.allclose(capturer.semantic_deployed_probability, expected)
    assert bool(capturer.semantic_route_deployed.item())
    assert capturer.semantic_token_route.sum() >= 1
    assert th.all(
        (capturer.semantic_token_route == 0)
        | (capturer.semantic_token_route == 1)
    )

    reference = scores.new_zeros(1)
    capturer.set_semantic_test_mode(False)
    train_gate = capturer._mlp_relation_gate(reference)
    assert th.equal(train_gate, capturer.semantic_token_route)

    capturer.set_semantic_test_mode(True)
    eval_gate = capturer._mlp_relation_gate(reference)
    expected_eval = (
        capturer.semantic_deployed_probability > capturer.semantic_router_threshold
    ).to(eval_gate)
    assert th.equal(eval_gate, expected_eval)

    # A dropped slot produces zero input gradient. Its previous importance must
    # remain available so future exploratory samples can restore and re-score it.
    dropped = capturer.semantic_token_route < 0.5
    if not bool(dropped.any().item()):
        dropped[expected.argmin()] = True
        capturer.semantic_token_route[dropped] = 0.0
    previous = capturer.semantic_route_score.clone()
    capturer._apply_semantic_route_scores(th.zeros_like(scores), t_env=1)
    assert th.equal(capturer.semantic_route_score[dropped], previous[dropped])


def _check_smac():
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
        semantic_router_mode="gradient_importance",
        semantic_router_ema_up=0.5,
        semantic_router_ema_down=0.99,
        semantic_router_update_interval=8000,
        semantic_router_temperature=0.2,
        semantic_router_warmup_steps=0,
        semantic_router_freeze_steps=20000000,
        relation_encoder_style="mlp",
        mlp_stochastic_hard_gate=True,
        mlp_stochastic_exploration_floor=0.05,
    )
    _check_route(capturer)

    capturer.set_semantic_test_mode(False)
    condition, hidden, *_ = capturer(
        th.randn(2, 4, 5),
        th.randn(2, 4, 2, 5),
        th.randn(2, 4, 3, 5),
        None,
    )
    assert condition.shape == (2, 4, 12)
    assert hidden.shape == (2, 4, 16)
    condition.square().mean().backward()
    assert capturer.semantic_probe_scale.grad is not None


def _check_grf():
    capturer = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=3,
        relation_dim=16,
        output_dim=12,
        use_absolute_public=True,
        semantic_router_mode="gradient_importance",
        semantic_router_ema_up=0.5,
        semantic_router_ema_down=0.99,
        semantic_router_update_interval=8000,
        semantic_router_temperature=0.2,
        semantic_router_warmup_steps=0,
        semantic_router_freeze_steps=20000000,
        relation_encoder_style="mlp",
        mlp_stochastic_hard_gate=True,
        mlp_stochastic_exploration_floor=0.05,
    )
    _check_route(capturer)

    capturer.set_semantic_test_mode(False)
    condition, hidden = capturer(
        th.randn(2, 3, capturer.expected_obs_dim),
        None,
    )
    assert condition.shape == (2, 3, 12)
    assert hidden.shape == (2, 3, 16)
    condition.square().mean().backward()
    assert capturer.semantic_probe_scale.grad is not None


def main():
    assert (
        "rpg_gimp_lowfreq_stochastic_hard_mlp_relation_hypercond"
        in MLP_GIMP_STOCHASTIC_HARD_VARIANTS
    )
    assert (
        "grf_abs_gimp_lowfreq_stochastic_hard_mlp_relation_hypercond"
        in GRF_MLP_GIMP_STOCHASTIC_HARD_VARIANTS
    )
    _check_smac()
    _check_grf()
    print("Stochastic hard MLP gate smoke test passed for SMAC and GRF.")


if __name__ == "__main__":
    main()
