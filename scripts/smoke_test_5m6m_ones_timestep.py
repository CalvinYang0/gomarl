#!/usr/bin/env python3
"""Production-shape raw-input/GRU/gradient/target checks, without SC2 launch."""
from unittest.mock import patch
import torch as th

from smoke_test_counter_transformer_nine import make_case


def check(label, fill):
    th.manual_seed(31)
    base, _, _, _ = make_case("linear_baseline", "5m_vs_6m", production_shapes=True)
    th.manual_seed(31)
    mac, learner, batch, logger = make_case(label, "5m_vs_6m", production_shapes=True)
    agent, capturer = mac.agent, mac.agent.rpg_relation_capturer
    assert agent.hidden_dim == agent.cond_dim == 64
    assert mac.args.obs_shape == 55 and mac.args.n_agents == 5
    assert mac.args.obs_agent_id and mac.args.obs_last_action
    assert agent.fc1.in_features == 55 + mac.args.n_actions + 5
    assert capturer.relation_encoder_style == "linear_only"
    assert len(capturer.transformer_layers) == 0 and capturer.dynamic_branch_gate is None
    assert agent.counter_hyper_condition_source is None
    assert agent.counter_transformer_policy_projection is None
    assert not agent.apply_hypermarl_init
    # Same trainable modules AND same initial weights as the Obs baseline.
    weights = agent.state_dict()
    assert weights.keys() == base.agent.state_dict().keys()
    assert all(th.equal(value, base.agent.state_dict()[key]) for key, value in weights.items())
    for flag in ("mask_parameter_relation_active", "temporal_param_auxiliary_active",
                 "random_drop_auxiliary_active", "gate_regularization_active",
                 "mixer_kl80_auxiliary_active", "advantage_margin_auxiliary_active"):
        assert not getattr(learner, flag)
    assert getattr(mac.args, "double_q", None) == getattr(base.args, "double_q", None)
    raw_obs, raw_state = batch["obs"].clone(), batch["state"].clone()
    seen = {}
    def capture_gru(module, inputs, output):
        seen["gru"] = output.detach().clone()
    original_head = agent._apply_dynamic_head
    def check_head(hidden, condition, *args, **kwargs):
        assert th.equal(hidden.detach().reshape_as(seen["gru"]), seen["gru"])
        return original_head(hidden, condition, *args, **kwargs)
    handle = agent.rnn.register_forward_hook(capture_gru)
    try:
        with patch.object(agent, "_apply_dynamic_head", side_effect=check_head):
            conditions = []
            with th.no_grad():
                mac.init_hidden(batch.batch_size)
                for t in (0, 1, 2):
                    context = mac._build_model_context(batch, t)
                    value = 1. if fill == "ones" else float(t)
                    assert th.equal(context["obs"], th.full_like(raw_obs[:, t], value))
                    assert th.equal(context["state"], raw_state[:, t])
                    trunk = mac._build_inputs(batch, t)
                    assert th.equal(trunk[..., :55], raw_obs[:, t])
                    mac.forward(batch, t, test_mode=True)
                    conditions.append(agent.latest_condition.clone())
                    assert th.allclose(agent.latest_condition, capturer.dual_linear_encoder(context["obs"]))
                    for generated in agent.latest_generated_parameter_graph:
                        per_agent = generated.reshape(batch.batch_size, 5, -1)
                        assert th.allclose(per_agent, per_agent[:, :1].expand_as(per_agent), atol=1e-6)
                    target_context = learner.target_mac._build_model_context(batch, t)
                    assert th.equal(target_context["obs"], context["obs"])
                if fill == "ones":
                    assert th.equal(conditions[0], conditions[2])
                else:
                    assert not th.allclose(conditions[0], conditions[2])
                # Condition ignores real environment features, but GRU/Q do not.
                mac.init_hidden(batch.batch_size)
                q_before = mac.forward(batch, 2, test_mode=True).clone()
                condition_before = agent.latest_condition.clone()
                batch["obs"][:, 2] += th.randn_like(batch["obs"][:, 2])
                batch["state"][:, 2] += th.randn_like(batch["state"][:, 2])
                mac.init_hidden(batch.batch_size)
                q_after = mac.forward(batch, 2, test_mode=True)
                assert th.equal(agent.latest_condition, condition_before)
                assert not th.allclose(q_before, q_after)
                batch["obs"].copy_(raw_obs)
                batch["state"].copy_(raw_state)
                # A new rollout/reset at t=0 restarts the clock, not t_env.
                mac.init_hidden(batch.batch_size)
                mac.select_actions(batch, 0, 9000000, test_mode=True)
                assert th.equal(agent.latest_condition, conditions[0])
            agent.zero_grad(set_to_none=True)
            mac.init_hidden(batch.batch_size)
            mac.forward(batch, 2).square().mean().backward()
            for module in (agent.rnn, capturer.dual_linear_encoder):
                assert any(p.grad is not None and th.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in module.parameters())
            learner.train(batch, t_env=10, episode_num=1)
            learner.train(batch, t_env=9000000, episode_num=2)
    finally:
        handle.remove()
    assert th.equal(batch["obs"], raw_obs) and th.equal(batch["state"], raw_state)
    assert all(th.isfinite(p).all() for p in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print(label + ": 55D fill, identical initial Obs architecture, unchanged GRU/mixer inputs, shared generated heads, target timing, gradients and real TD updates PASS", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    check("linear_ones_baseline", "ones")
    check("linear_timestep_baseline", "episode_timestep")
