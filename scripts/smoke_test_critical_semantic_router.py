#!/usr/bin/env python3
"""Tensor checks for critical-state semantic gradient attribution."""

import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    PublicTransformerRelationCapturer,
)


def build_capturer():
    return PublicTransformerRelationCapturer(
        move_dim=4,
        own_dim=1,
        ally_feat_dim=5,
        enemy_feat_dim=5,
        relation_dim=16,
        output_dim=12,
        obs_own_health=True,
        obs_all_health=True,
        n_allies=1,
        n_enemies=1,
        mode="private_bias",
        num_heads=4,
        num_layers=1,
        merge_friendly_public_side=True,
        private_owner_side=True,
        private_bias_style="simple",
        semantic_router_mode="gradient_importance_critical",
        semantic_router_ema_up=0.5,
        semantic_router_ema_down=0.99,
        semantic_router_update_interval=8000,
        semantic_router_warmup_steps=0,
        semantic_router_freeze_steps=10_000_000,
    )


def check_per_state_tail_attribution():
    capturer = build_capturer()
    capturer.train()
    capturer.begin_semantic_critical_capture()

    batch_size, n_agents, time_count = 2, 3, 4
    slot_count = len(capturer.semantic_names)
    losses = []
    for time_index in range(time_count):
        flat_obs = th.ones(batch_size, n_agents, slot_count)
        scale = capturer._semantic_scales(flat_obs)
        weights = th.ones_like(flat_obs)
        weights[..., 1] = 0.0
        if time_index == time_count - 1:
            weights[0, :, 1] = 20.0
        losses.append((flat_obs * scale * weights).sum())
    th.stack(losses).sum().backward()

    transition_mask = th.ones(batch_size, time_count, 1)
    score = capturer.consume_semantic_critical_importance(
        transition_mask,
        tail_fraction=0.25,
        tail_weight=0.5,
    )
    assert score.shape == (slot_count,)
    assert th.isfinite(score).all()
    assert score[1] > score[0]
    assert (
        capturer._semantic_critical_stats["semantic_critical_tail_score"]
        > capturer._semantic_critical_stats["semantic_critical_mean_score"]
    )


def check_full_slot_capture():
    capturer = build_capturer()
    capturer.train()
    capturer.begin_semantic_critical_capture()

    batch_size, n_agents = 2, 3
    self_feat = th.rand(batch_size, n_agents, 5)
    ally_feat = th.rand(batch_size, n_agents, 1, 5)
    enemy_feat = th.rand(batch_size, n_agents, 1, 5)
    ally_mask = th.ones(batch_size, n_agents, 1, dtype=th.bool)
    enemy_mask = th.ones(batch_size, n_agents, 1, dtype=th.bool)

    routed = capturer._semantic_routed_embeddings(
        self_feat,
        ally_feat,
        enemy_feat,
        ally_mask,
        enemy_mask,
    )
    sum(value.sum() for value in routed if value is not None).backward()

    assert len(capturer._semantic_critical_probes) == 1
    probe = capturer._semantic_critical_probes[0]
    assert probe.shape == (
        batch_size,
        n_agents,
        len(capturer.semantic_names),
    )
    assert probe.grad is not None
    self_view, ally_view, enemy_view = capturer._semantic_slot_views(probe)
    assert self_view.shape == (batch_size, n_agents, 5)
    assert ally_view.shape == (batch_size, n_agents, 1, 5)
    assert enemy_view.shape == (batch_size, n_agents, 1, 5)


def check_soft_route_without_audit():
    capturer = build_capturer()
    capturer.semantic_router_temperature = 0.5
    scores = th.linspace(0.25, 2.0, len(capturer.semantic_names))
    capturer.update_semantic_router(t_env=1, external_score=scores)

    token_route, bias_route = capturer._current_semantic_routes(scores)
    assert th.all(token_route > 0.0)
    assert th.all(token_route < 1.0)
    assert th.allclose(token_route + bias_route, th.ones_like(token_route))
    assert th.allclose(capturer.semantic_keep_route, th.ones_like(token_route))
    assert not capturer.semantic_router_needs_counterfactual()
    assert not capturer.semantic_router_needs_independent_audit()
    assert not capturer.semantic_router_needs_binary_audit()


def main():
    th.manual_seed(7)
    check_per_state_tail_attribution()
    check_full_slot_capture()
    check_soft_route_without_audit()
    print("Critical-state semantic router smoke test passed.")


if __name__ == "__main__":
    main()
