#!/usr/bin/env python3
"""Fast tensor checks for the FiLM and DROP semantic-routing variants."""

import math

import torch as th

from modules.agents.clean_hyper_agent import PublicTransformerRelationCapturer


def build_capturer(private_bias_style, drop_mode):
    return PublicTransformerRelationCapturer(
        move_dim=4,
        own_dim=1,
        ally_feat_dim=5,
        enemy_feat_dim=5,
        relation_dim=16,
        output_dim=16,
        unit_type_bits=0,
        shield_bits_ally=0,
        shield_bits_enemy=0,
        obs_all_health=True,
        obs_own_health=True,
        n_allies=4,
        n_enemies=6,
        mode="private_bias",
        num_heads=4,
        num_layers=1,
        merge_friendly_public_side=True,
        private_owner_side=True,
        private_bias_style=private_bias_style,
        semantic_router_mode="gradient_importance",
        semantic_router_warmup_steps=0,
        semantic_router_freeze_steps=10_000_000,
        semantic_router_temperature=0.1,
        semantic_router_drop_mode=drop_mode,
        semantic_router_keep_threshold=0.35,
        semantic_router_keep_ratio=0.5,
    )


def check_case(private_bias_style, drop_mode):
    model = build_capturer(private_bias_style, drop_mode)
    model.train()

    slot_count = len(model.semantic_names)
    scores = th.linspace(0.1, 2.0, slot_count)
    model._apply_semantic_route_scores(scores, t_env=500_000)
    token_route, bias_route = model._current_semantic_routes(th.zeros(1))
    keep_route = model.semantic_keep_route

    assert th.allclose(token_route.detach() + bias_route.detach(), keep_route)
    assert th.all(token_route.detach() * bias_route.detach() == 0)
    if drop_mode in {"threshold", "topk"}:
        assert bias_route.sum().item() == 0
    if drop_mode == "topk":
        assert int(keep_route.sum().item()) == math.ceil(slot_count * 0.5)

    batch_size, n_agents = 2, 5
    self_feat = th.rand(batch_size, n_agents, 5)
    ally_feat = th.rand(batch_size, n_agents, 4, 5)
    enemy_feat = th.rand(batch_size, n_agents, 6, 5)
    prev_hidden = th.zeros(batch_size, n_agents, 16)

    ally_mask = ally_feat.abs().sum(dim=-1) > 0
    enemy_mask = enemy_feat.abs().sum(dim=-1) > 0
    routed = model._semantic_routed_embeddings(
        self_feat, ally_feat, enemy_feat, ally_mask, enemy_mask
    )
    mod_self, mod_ally, mod_enemy = routed[3:]
    if drop_mode in {"threshold", "topk"}:
        assert mod_self is None and mod_ally is None and mod_enemy is None
    if drop_mode == "hierarchical":
        assert th.allclose(mod_self, th.zeros_like(mod_self), atol=1e-7)
        assert th.allclose(mod_ally, th.zeros_like(mod_ally), atol=1e-7)
        assert th.allclose(mod_enemy, th.zeros_like(mod_enemy), atol=1e-7)
        attention_bias = model._build_attention_bias(
            mod_self,
            mod_ally,
            mod_enemy,
            batch_size,
            n_agents,
            ally_mask,
            enemy_mask,
        )
        assert th.allclose(
            attention_bias, th.zeros_like(attention_bias), atol=1e-7
        )

    outputs = model(self_feat, ally_feat, enemy_feat, prev_hidden)
    condition, hidden = outputs[:2]
    assert condition.shape == (batch_size, n_agents, 16)
    assert hidden.shape == (batch_size, n_agents, 16)

    (condition.sum() + hidden.sum()).backward()
    if private_bias_style == "film":
        assert model.film_modulation.weight.grad is not None
        assert th.isfinite(model.film_modulation.weight.grad).all()
    if drop_mode == "hierarchical":
        assert model.semantic_usage_logit.grad is not None
        assert th.isfinite(model.semantic_usage_logit.grad).all()

    print(
        "{}/{}: TOKEN={} BIAS={} DROP={} forward_backward=ok".format(
            private_bias_style,
            drop_mode,
            int(token_route.detach().sum().item()),
            int(bias_route.detach().sum().item()),
            int((1.0 - keep_route).sum().item()),
        )
    )


def main():
    th.manual_seed(7)
    for private_bias_style, drop_mode in (
        ("film", "none"),
        ("simple", "threshold"),
        ("simple", "hierarchical"),
        ("simple", "topk"),
    ):
        check_case(private_bias_style, drop_mode)


if __name__ == "__main__":
    main()
