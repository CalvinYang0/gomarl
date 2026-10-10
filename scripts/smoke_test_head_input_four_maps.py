#!/usr/bin/env python3
"""Real production-shaped learner updates for this exact eight-cohort batch."""
import torch as th

from ozstar_submit_head_input_24jobs import SCENES
from smoke_test_counter_transformer_nine import make_case
from smoke_test_linear_id_baseline import check as check_id


def check(scene, label):
    th.manual_seed(67)
    mac, learner, batch, logger = make_case(label, scene, production_shapes=True)
    agent, args = mac.agent, mac.args
    assert agent.rpg_relation_capturer.relation_encoder_style == "linear_only"
    assert agent.rpg_relation_capturer.dynamic_branch_gate is None
    assert agent.counter_transformer_policy_projection is None
    assert agent.hidden_dim == agent.cond_dim == 64
    assert args.obs_last_action and args.obs_agent_id
    assert agent.fc1.in_features == args.obs_shape + args.n_actions + args.n_agents
    for flag in ("mask_parameter_relation_active", "temporal_param_auxiliary_active",
                 "random_drop_auxiliary_active", "gate_regularization_active",
                 "mixer_kl80_auxiliary_active", "advantage_margin_auxiliary_active"):
        assert not getattr(learner, flag)
    raw_obs, raw_state = batch["obs"].clone(), batch["state"].clone()
    mac.init_hidden(batch.batch_size)
    with th.no_grad():
        before = mac.forward(batch, 0, test_mode=True).clone()
        condition = agent.latest_condition.clone()
        assert th.equal(mac._build_inputs(batch, 0)[..., :args.obs_shape], raw_obs[:, 0])
        context = mac._build_model_context(batch, 0)
        assert th.equal(context["state"], raw_state[:, 0])
        if label == "linear_ones_baseline":
            assert th.equal(context["obs"], th.ones_like(context["obs"]))
        else:
            assert th.equal(context["obs"], raw_obs[:, 0])
        batch["obs"][:, 0] += th.randn_like(batch["obs"][:, 0])
        mac.init_hidden(batch.batch_size)
        after = mac.forward(batch, 0, test_mode=True)
        assert not th.allclose(before, after), "Main policy must still see actual local Obs"
        if label == "linear_ones_baseline":
            assert th.equal(condition, agent.latest_condition)
            assert th.allclose(condition, condition[:, :1].expand_as(condition), atol=1e-6)
        else:
            assert not th.allclose(condition, agent.latest_condition)
        batch["obs"].copy_(raw_obs)
        target_context = learner.target_mac._build_model_context(batch, 0)
        assert th.equal(target_context["obs"], context["obs"])
    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=9000000, episode_num=2)
    assert logger.stats["loss_td"][-1][1] >= 0
    assert not logger.stats.get("loss_aux")
    assert th.equal(raw_obs, batch["obs"]) and th.equal(raw_state, batch["state"])
    assert all(th.isfinite(p).all() for p in agent.parameters())
    print("PASS:", scene, label, "real local Obs -> GRU; head condition; unchanged mixer state; target and TD updates", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    for _, scene, labels in SCENES:
        for label in labels:
            if label == "linear_id_baseline":
                check_id(scene, production_shapes=True)
            else:
                check(scene, label)
    print("PASS: all eight requested map/model pairs; synthetic data, no SC2 launch or uploads")
