#!/usr/bin/env python3
"""Production 5m6m path/gradient/TD/diagnostic checks; no SC2 or upload."""
import torch as th

from smoke_test_counter_transformer_nine import make_case
from utils.hyper_obs_importance import analyze, head_jacobian, hyper_q, replay_samples, supported


def main():
    th.set_num_threads(1)
    th.manual_seed(31)
    base, base_learner, _, _ = make_case("linear_baseline", "5m_vs_6m", production_shapes=True)
    th.manual_seed(31)
    mac, learner, batch, logger = make_case("linear_health_baseline", "5m_vs_6m", production_shapes=True)
    agent, capturer = mac.agent, mac.agent.rpg_relation_capturer
    assert mac.args.obs_shape == 55 and mac.n_agents == 5
    names = list(capturer.semantic_names)
    health = [i for i, name in enumerate(names) if name == "self_health" or name.endswith("_health")]
    other = [i for i in range(55) if i not in health]
    assert health == [8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 54]
    assert {names[i] for i in health} == {"self_health"} | {
        "enemy_{}_health".format(i) for i in range(6)} | {"ally_{}_health".format(i) for i in range(4)}
    assert capturer.dual_linear_encoder.in_features == 55 and supported(agent)
    assert agent.hidden_dim == agent.cond_dim == 64
    assert capturer.relation_encoder_style == "linear_only" and not len(capturer.transformer_layers)
    assert capturer.dynamic_branch_gate is None
    assert agent.counter_hyper_condition_source is None and agent.counter_transformer_policy_projection is None
    assert agent.state_dict().keys() == base.agent.state_dict().keys()
    assert all(th.equal(v, base.agent.state_dict()[k]) for k, v in agent.state_dict().items())
    assert all(th.equal(v, base_learner.mixer.state_dict()[k]) for k, v in learner.mixer.state_dict().items())
    for flag in ("mask_parameter_relation_active", "temporal_param_auxiliary_active",
                 "random_drop_auxiliary_active", "gate_regularization_active",
                 "mixer_kl80_auxiliary_active", "advantage_margin_auxiliary_active"):
        assert not getattr(learner, flag)
    assert getattr(mac.args, "double_q", None) == getattr(base.args, "double_q", None)
    raw_obs, raw_state = batch["obs"].clone(), batch["state"].clone()
    # Per-field sentinels catch incorrect health offsets, including own HP at the tail.
    sentinel = th.arange(1, 56).to(raw_obs).expand_as(raw_obs[:, 0])
    filtered = capturer.health_only_hyper_input(sentinel)
    assert th.equal(filtered[..., health], sentinel[..., health])
    assert filtered[..., other].count_nonzero() == 0
    for mode in (False, True):
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            context = mac._build_model_context(batch, t)
            assert th.equal(context["obs"][..., health], raw_obs[:, t, :, health])
            assert context["obs"][..., other].count_nonzero() == 0
            for key, at in (("prev_obs", t - 1), ("next_obs", t + 1)):
                assert context[key][..., other].count_nonzero() == 0
                if 0 <= at < batch.max_seq_length:
                    assert th.equal(context[key][..., health], raw_obs[:, at, :, health])
                else:
                    assert context[key].count_nonzero() == 0
            assert th.equal(context["state"], raw_state[:, t])
            assert th.equal(mac._build_inputs(batch, t)[..., :55], raw_obs[:, t])
            assert th.equal(learner.target_mac._build_model_context(batch, t)["obs"], context["obs"])
            with th.no_grad():
                q = mac.forward(batch, t, test_mode=mode)
                assert th.allclose(agent.latest_condition, capturer.dual_linear_encoder(context["obs"]))
                pure = hyper_q(agent, raw_obs[:, t].reshape(-1, 55),
                               mac.hidden_states[..., :64].reshape(-1, 64)).reshape_as(q)
                assert th.allclose(q, pure, atol=1e-6)
    with th.no_grad():
        mac.init_hidden(batch.batch_size)
        before = mac.forward(batch, 2, test_mode=True).clone()
        condition = agent.latest_condition.clone()
        batch["obs"][:, 2, :, other] += .7
        mac.init_hidden(batch.batch_size)
        after = mac.forward(batch, 2, test_mode=True)
        assert th.equal(condition, agent.latest_condition) and not th.allclose(before, after)
        batch["obs"][:, 2, :, health] += .3
        mac.init_hidden(batch.batch_size)
        mac.forward(batch, 2, test_mode=True)
        assert not th.allclose(condition, agent.latest_condition)
        batch["obs"].copy_(raw_obs)
    agent.zero_grad(set_to_none=True)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, 2).square().mean().backward()
    gradient = capturer.dual_linear_encoder.weight.grad
    assert gradient[:, other].count_nonzero() == 0 and gradient[:, health].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.rnn.parameters())
    for jacobian in head_jacobian(agent):
        assert jacobian[:, other].count_nonzero() == 0
        assert jacobian[:, health].abs().sum() > 0
    result = analyze(agent, replay_samples(mac, batch, 2), 30)
    assert all(result["gap_gradient_abs_mean"][i] == 0 for i in other)
    assert all(row["action_flip_rate"] == 0 and row["original_pair_margin_abs_change"] == 0
               for row in result["group_mean_replacement"] if not row["group"].endswith("_health"))
    learner.train(batch, t_env=10, episode_num=1)
    learner.train(batch, t_env=9000000, episode_num=2)
    assert logger.stats["loss_td"][-1][1] >= 0
    assert th.equal(batch["obs"], raw_obs) and th.equal(batch["state"], raw_state)
    assert all(th.isfinite(p).all() for p in agent.parameters())
    print("PASS: health-only 11/55 exact slots; same initialization/GRU/mixer/Double-Q/main TD; behaviour/test/target; gradients and hyper-only diagnostics")


if __name__ == "__main__":
    main()
