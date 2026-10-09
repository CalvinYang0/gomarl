#!/usr/bin/env python3
"""Real SMAC layouts and production networks; no simulator or W&B launch."""
import copy
from unittest.mock import patch

import torch as th

from smoke_test_counter_transformer_nine import make_case
from modules.agents.counter_transformer_suite import model_type_for
from utils.hyper_obs_importance import supported
from utils.value_diagnostics import ValueDiagnosticSummary, collect_value_diagnostics


LABEL = "linear_obs_entity_id_baseline"


def check(scene):
    th.manual_seed(67)
    baseline, _, _, _ = make_case("linear_baseline", scene, production_shapes=True)
    th.manual_seed(67)
    mac, learner, batch, logger = make_case(LABEL, scene, production_shapes=True)
    agent, capturer = mac.agent, mac.agent.rpg_relation_capturer
    n, d, e = agent.n_agents, mac.args.obs_shape, capturer.observation_layout["n_enemies"]
    augmented_dim = d + (n + e) ** 2
    assert capturer.hyper_entity_ids and capturer.relation_encoder_style == "linear_only"
    assert len(capturer.transformer_layers) == 0 and capturer.dynamic_branch_gate is None
    assert agent.counter_hyper_condition_source is None
    assert agent.counter_transformer_policy_projection is None
    assert capturer.expected_obs_dim == d
    assert capturer.dual_linear_encoder.in_features == augmented_dim
    assert len(capturer.hyper_entity_input_names) == augmented_dim
    assert agent.fc1.in_features == d + mac.args.n_actions + n
    assert agent.hidden_dim == agent.cond_dim == 64 and not agent.apply_hypermarl_init
    assert not supported(agent)  # Old raw-Obs-only Jacobian diagnostic must not mislabel IDs.
    for flag in ("mask_parameter_relation_active", "temporal_param_auxiliary_active",
                 "random_drop_auxiliary_active", "gate_regularization_active",
                 "mixer_kl80_auxiliary_active", "advantage_margin_auxiliary_active"):
        assert not getattr(learner, flag)
    # All other seeded modules are bitwise unchanged. Same raw Obs columns too.
    base_weights = baseline.agent.state_dict()
    current = agent.state_dict()
    assert set(current) == set(base_weights)
    for key in current:
        if key == "rpg_relation_capturer.dual_linear_encoder.weight":
            assert th.equal(current[key][:, capturer.hyper_entity_raw_indices], base_weights[key])
        else:
            assert th.equal(current[key], base_weights[key]), key
    assert sum(p.numel() for p in agent.parameters()) - sum(p.numel() for p in baseline.agent.parameters()) == (n + e) ** 2 * 64
    raw_obs, raw_state = batch["obs"].clone(), batch["state"].clone()
    obs = batch["obs"][:, 0]
    augmented = capturer.entity_id_hyper_input(obs)
    assert th.equal(augmented[..., capturer.hyper_entity_raw_indices], obs)
    assert augmented.shape == (batch.batch_size, n, augmented_dim)
    names = capturer.hyper_entity_input_names
    for i in range(n):
        expected = {"self": i}
        expected.update({"ally_{}".format(slot): j for slot, j in enumerate(j for j in range(n) if j != i)})
        expected.update({"enemy_{}".format(j): n + j for j in range(e)})
        for label, identity in expected.items():
            columns = [names.index(label + "_id_{}".format(j)) for j in range(n + e)]
            values = augmented[:, i, columns]
            assert th.equal(values.argmax(-1), th.full((batch.batch_size,), identity))
            assert th.equal(values.sum(-1), th.ones(batch.batch_size))
    # IDs contain no visibility, life or state information; stay fixed after reset.
    id_columns = [i for i in range(augmented_dim) if i not in capturer.hyper_entity_raw_indices.tolist()]
    empty = capturer.entity_id_hyper_input(th.zeros_like(obs))
    assert th.equal(empty[..., id_columns], augmented[..., id_columns])
    assert th.equal(capturer.entity_id_hyper_input(batch["obs"][:, 3])[..., id_columns], augmented[..., id_columns])
    assert th.equal(capturer.entity_id_hyper_input(obs.double())[..., id_columns], augmented[..., id_columns].double())
    assert th.equal(learner.target_mac.agent.rpg_relation_capturer.entity_id_hyper_input(obs), augmented)
    # Zeroing ID columns must exactly reduce the generated head to the old Obs head.
    reduced = copy.deepcopy(mac)
    with th.no_grad():
        reduced.agent.rpg_relation_capturer.dual_linear_encoder.weight[:, id_columns] = 0
        baseline.init_hidden(batch.batch_size)
        reduced.init_hidden(batch.batch_size)
        for t in range(3):
            assert th.allclose(baseline.forward(batch, t, test_mode=True), reduced.forward(batch, t, test_mode=True), atol=1e-5, rtol=1e-5)
    seen = {}
    handle = agent.rnn.register_forward_hook(lambda module, inputs, output: seen.update(hidden=output.detach().clone()))
    original_head = agent._apply_dynamic_head

    def verify_head(hidden, condition, *args, **kwargs):
        assert th.equal(hidden.detach().reshape_as(seen["hidden"]), seen["hidden"])
        return original_head(hidden, condition, *args, **kwargs)

    try:
        with patch.object(agent, "_apply_dynamic_head", side_effect=verify_head):
            mac.init_hidden(batch.batch_size)
            with th.no_grad():
                for t in range(3):
                    context = mac._build_model_context(batch, t)
                    assert th.equal(context["obs"], raw_obs[:, t])
                    assert th.equal(mac._build_inputs(batch, t)[..., :d], raw_obs[:, t])
                    mac.forward(batch, t, test_mode=True)
                    expected = capturer.dual_linear_encoder(capturer.entity_id_hyper_input(context["obs"]))
                    assert th.allclose(agent.latest_condition, expected)
                # Hypercondition must respond to real Obs, unlike ID-only.
                before = agent.latest_condition.clone()
                altered = raw_obs[:, 2] + .2
                after, _ = capturer(altered, None)
                assert not th.allclose(before, after)
            agent.zero_grad(set_to_none=True)
            mac.init_hidden(batch.batch_size)
            mac.forward(batch, 0).square().mean().backward()
            gradient = capturer.dual_linear_encoder.weight.grad
            assert gradient[:, capturer.hyper_entity_raw_indices].abs().sum() > 0
            assert gradient[:, id_columns].abs().sum() > 0
            assert agent.rnn.weight_ih.grad.abs().sum() > 0
            learner.train(batch, t_env=10, episode_num=1)
            learner.train(batch, t_env=300000, episode_num=2)
    finally:
        handle.remove()
    # Existing value diagnostics remain compatible with the generated head.
    summary = ValueDiagnosticSummary()
    collect_value_diagnostics(mac, learner.mixer, batch, mac.args.gamma, summary)
    assert th.equal(batch["obs"], raw_obs) and th.equal(batch["state"], raw_state)
    assert all(th.isfinite(p).all() for p in mac.parameters())
    assert logger.stats["loss_td"][-1][1] >= 0
    print("PASS: {} raw={} augmented={}; absolute self/ally/enemy IDs, same trunk/head/mixer, gradients/target/TD/value diagnostics".format(scene, d, augmented_dim))


if __name__ == "__main__":
    th.set_num_threads(1)
    try:
        model_type_for(LABEL, "grf")
    except ValueError:
        pass
    else:
        raise AssertionError("Entity-ID profile must be SMAC-only")
    for scene in ("5m_vs_6m", "8m_vs_9m"):
        check(scene)
