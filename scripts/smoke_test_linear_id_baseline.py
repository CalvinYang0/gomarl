#!/usr/bin/env python3
"""Verify the matched Linear-ID semantics and real learner updates, no SC2 launch.

Default uses installed SMAC dimensions. --grf checks the same shared agent path
on a synthetic football layout when SMAC dependencies are unavailable locally.
"""
import sys
from unittest.mock import patch

import torch as th

from smoke_test_counter_transformer_nine import make_case


def check(scene, production_shapes=False):
    th.manual_seed(23)
    mac, learner, batch, logger = make_case(
        "linear_id_baseline", scene, production_shapes=production_shapes
    )
    agent = mac.agent
    assert agent.rpg_relation_capturer.relation_encoder_style == "linear_only"
    assert agent.counter_hyper_condition_source == "agent_id_linear"
    assert agent.counter_transformer_policy_projection is None
    assert isinstance(agent.counter_id_condition_encoder, th.nn.Linear)
    assert agent.rpg_relation_capturer.dynamic_branch_gate is None
    assert not agent.apply_hypermarl_init
    if production_shapes:
        assert agent.hidden_dim == agent.cond_dim == 64
        assert mac.args.obs_last_action and mac.args.obs_agent_id
        assert agent.fc1.in_features == mac.args.obs_shape + mac.args.n_actions + mac.args.n_agents
    for flag in ("mask_parameter_relation_active", "temporal_param_auxiliary_active",
                 "random_drop_auxiliary_active", "gate_regularization_active",
                 "mixer_kl80_auxiliary_active"):
        assert not getattr(learner, flag)
    captured = {}

    def capture_gru(module, inputs, output):
        captured["gru"] = output.detach().clone()

    original_head = agent._apply_dynamic_head

    def checked_head(hidden, condition, *args, **kwargs):
        assert th.equal(hidden.detach().reshape_as(captured["gru"]), captured["gru"])
        return original_head(hidden, condition, *args, **kwargs)

    handle = agent.rnn.register_forward_hook(capture_gru)
    try:
        with patch.object(agent, "_apply_dynamic_head", side_effect=checked_head):
            mac.init_hidden(batch.batch_size)
            with th.no_grad():
                q_a = mac.forward(batch, t=0, test_mode=True)
                condition_a = agent.latest_condition.clone()
                batch["obs"][:, 0] += th.randn_like(batch["obs"][:, 0])
                mac.init_hidden(batch.batch_size)
                q_b = mac.forward(batch, t=0, test_mode=True)
                condition_b = agent.latest_condition.clone()
                mac.forward(batch, t=1, test_mode=True)
                assert th.equal(condition_a, agent.latest_condition)
            assert th.equal(condition_a, condition_b)
            assert not th.allclose(condition_a[:, 0], condition_a[:, 1])
            assert not th.allclose(q_a, q_b)
            # Backprop must reach the GRU and ID generator, not a hidden
            # observation-conditioned policy replacement.
            agent.zero_grad(set_to_none=True)
            mac.init_hidden(batch.batch_size)
            mac.forward(batch, t=0).square().mean().backward()
            for module in (agent.rnn, agent.counter_id_condition_encoder):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
            assert all(p.grad is None for p in agent.rpg_relation_capturer.parameters())
            learner.train(batch, t_env=10, episode_num=1)
            learner.train(batch, t_env=300000, episode_num=2)
    finally:
        handle.remove()
    assert all(th.isfinite(p).all() for p in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(scene + ": Linear ID conditions + GRU head input + learner updates OK", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    scenes = ("academy_counterattack_easy",) if "--grf" in sys.argv else ("5m_vs_6m", "8m_vs_9m")
    for scene in scenes:
        check(scene)
