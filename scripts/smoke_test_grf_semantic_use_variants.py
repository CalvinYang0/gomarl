#!/usr/bin/env python3
"""Exercise the three GRF semantic-use variants before Slurm submission."""

from pathlib import Path
import sys

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRFPublicPrivateBiasTransformerCapturer,
)


CASES = (
    ("film", "film", "none"),
    ("hierarchical_drop", "simple_bias", "learnable_hierarchical"),
    ("str_sparse", "token_only", "str_sparse"),
)


def run_case(name, use_mode, drop_mode):
    th.manual_seed(7)
    n_agents = 3
    relation_dim = 16
    obs_dim = 4 * n_agents + 14
    capturer = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=n_agents,
        relation_dim=relation_dim,
        output_dim=16,
        num_heads=4,
        num_layers=1,
        use_absolute_public=True,
        semantic_router_mode="gradient_importance",
        semantic_router_learnable_threshold=True,
        semantic_router_temperature=0.1,
        semantic_router_warmup_steps=0,
        semantic_router_freeze_steps=10_000_000,
        semantic_router_use_mode=use_mode,
        semantic_router_drop_mode=drop_mode,
        semantic_router_keep_threshold=0.35,
        semantic_router_sparse_coef=0.001,
    )

    scores = th.linspace(0.05, 2.0, len(capturer.semantic_names))
    capturer.update_semantic_router(t_env=1, external_score=scores)
    obs = th.randn(2, n_agents, obs_dim)
    previous_hidden = th.zeros(2, n_agents, relation_dim)
    condition, next_hidden = capturer(obs, previous_hidden)

    assert condition.shape == (2, n_agents, 16)
    assert next_hidden.shape == (2, n_agents, relation_dim)
    assert th.isfinite(condition).all()
    assert th.isfinite(next_hidden).all()

    loss = condition.square().mean() + next_hidden.square().mean()
    if capturer.latest_aux_loss is not None:
        loss = loss + capturer.latest_aux_loss
    loss.backward()

    threshold_grad = capturer.semantic_router_threshold_logit.grad
    assert threshold_grad is not None and th.isfinite(threshold_grad)
    if drop_mode == "learnable_hierarchical":
        drop_grad = capturer.semantic_router_drop_threshold_logit.grad
        assert drop_grad is not None and th.isfinite(drop_grad)
        assert th.all(capturer.semantic_token_route <= capturer.semantic_keep_route)
    if drop_mode == "str_sparse":
        assert capturer.latest_aux_loss is not None
        assert "semantic_sparse_zero_fraction" in capturer.latest_aux_stats

    stats = capturer.semantic_router_stats()
    print(
        "{}: condition={} token={} bias={} drop={} threshold_grad={:.6g}".format(
            name,
            tuple(condition.shape),
            int(stats["semantic_route_token_count"].item()),
            int(stats["semantic_route_bias_count"].item()),
            int(stats["semantic_route_drop_count"].item()),
            float(threshold_grad.item()),
        )
    )


def main():
    for case in CASES:
        run_case(*case)
    print("GRF semantic-use smoke test passed")


if __name__ == "__main__":
    main()
