#!/usr/bin/env python3
"""Tensor checks for corrected scalar hard gates and Transformer-only control."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    GRF_DUAL_BRANCH_FIXED_RANDOM_DROP_KEEP_BY_MODEL,
    GRF_DUAL_BRANCH_GATE_REGULARIZER_BY_MODEL,
    GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL,
    GRF_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS,
    GRF_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS,
    GRF_DUAL_BRANCH_SPLIT_HEAD_VARIANTS,
    GRF_DUAL_BRANCH_VARIANTS,
    GRF_MLP_RELATION_VARIANTS,
    GRF_SINGLE_LINEAR_BRANCH_VARIANTS,
    GRF_SINGLE_TRANSFORMER_BRANCH_VARIANTS,
    GRFPublicPrivateBiasTransformerCapturer,
    CleanHyperAgent,
    RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    RPG_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS,
    RPG_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS,
    RPG_DUAL_BRANCH_SPLIT_HEAD_VARIANTS,
    RPG_DUAL_BRANCH_VARIANTS,
)
from learners.clean_learner import CleanLearner  # noqa: E402
from controllers.clean_controller import CleanMAC  # noqa: E402


def build_grf(gate_mode, gate_scope="both", **kwargs):
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="dual",
        semantic_router_share_by_side=True,
        dynamic_branch_gate_mode=gate_mode,
        dynamic_branch_gate_hidden_dim=16,
        dynamic_branch_gate_warmup_steps=250000,
        dynamic_branch_gate_scope=gate_scope,
        **kwargs,
    )


def build_grf_transformer_only():
    return GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="attention_only",
    )


def check_gate_module(model, obs):
    generator = model.dynamic_branch_gate
    assert generator is not None
    eval_gate_1, eval_prob_1 = generator(obs, sample=False)
    expected_shape = (2,) + tuple(obs.shape)
    assert eval_gate_1.shape == expected_shape
    assert eval_prob_1.shape == expected_shape
    assert th.all((eval_gate_1 == 0.0) | (eval_gate_1 == 1.0))
    if generator.mode == "binary_concrete":
        train_gate_1, train_prob_1 = generator(obs, sample=True)
        train_gate_2, train_prob_2 = generator(obs, sample=True)
        assert th.equal(eval_prob_1, train_prob_1)
        assert th.equal(train_prob_1, train_prob_2)
        assert not th.equal(train_gate_1, train_gate_2)
        assert th.any((train_gate_1 > 0.0) & (train_gate_1 < 1.0))
        with th.no_grad():
            rollout_gate_1, _ = generator(obs, sample=True)
            rollout_gate_2, _ = generator(obs, sample=True)
            target_gate, target_prob = generator(
                obs, sample=True, deterministic_soft=True
            )
        assert not th.equal(rollout_gate_1, rollout_gate_2)
        assert th.equal(target_gate, target_prob)
    else:
        eval_gate_2, eval_prob_2 = generator(obs, sample=True)
        assert th.equal(eval_gate_1, eval_gate_2)
        assert th.equal(eval_prob_1, eval_prob_2)
    final_layer = generator.gate_network[-1]
    expected_out_features = (
        generator.obs_dim
        if generator.gate_scope == "shared"
        else 2 * generator.obs_dim
    )
    assert final_layer.out_features == expected_out_features
    if generator.gate_scope == "shared":
        shared_gates, _ = generator(obs, sample=False)
        assert th.equal(shared_gates[0], shared_gates[1])
        return
    with th.no_grad():
        saved_bias = final_layer.bias.clone()
        final_layer.bias[0] = -1.0
        final_layer.bias[1] = 1.0
        independent_gates, _ = generator(obs, sample=False)
        final_layer.bias.copy_(saved_bias)
    assert th.all(independent_gates[0, ..., 0] == 0.0)
    assert th.all(independent_gates[0, ..., 1] == 1.0)


def check_grf(gate_mode):
    model = build_grf(gate_mode)
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    check_gate_module(model, obs)

    model.set_dynamic_branch_gate_t_env(250000)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None
    assert (
        model.dynamic_branch_gate.gate_network[-1].weight.grad.abs().sum().item()
        > 0.0
    )
    assert "dynamic_gate_attention_mean" in model.latest_aux_stats
    assert model.latest_aux_stats["dynamic_gate_warmup_active"].item() == 0.0
    for branch in ("linear", "attention"):
        slot_values = []
        for slot_name in model.semantic_names:
            key = "dynamic_gate_{}_probability_slot_{}".format(
                branch, slot_name
            )
            assert key in model.latest_aux_stats
            value = model.latest_aux_stats[key]
            assert value.ndim == 0 and th.isfinite(value)
            slot_values.append(value)
        expected_std = th.stack(slot_values).std(unbiased=False)
        recorded_std = model.latest_aux_stats[
            "dynamic_gate_{}_probability_slot_std".format(branch)
        ]
        assert th.allclose(recorded_std, expected_std)

    model.zero_grad(set_to_none=True)
    model.set_dynamic_branch_gate_t_env(249999)
    model(obs, hidden)
    assert model.latest_aux_stats["dynamic_gate_warmup_active"].item() == 1.0
    assert model.latest_aux_stats["dynamic_gate_linear_mean"].item() == 1.0
    assert model.latest_aux_stats["dynamic_gate_attention_mean"].item() == 1.0

    model.set_dynamic_branch_gate_t_env(250000)
    model.set_dynamic_branch_gate_force_open(True)
    model(obs, hidden)
    assert model.latest_aux_stats["dynamic_gate_linear_mean"].item() == 1.0
    assert model.latest_aux_stats["dynamic_gate_attention_mean"].item() == 1.0
    model.set_dynamic_branch_gate_force_open(False)


def check_grf_transformer_only():
    model = build_grf_transformer_only()
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    assert model.dynamic_branch_gate is None
    assert model.dual_linear_encoder is None
    assert model.dual_condition_fuser is None
    (condition.mean() + next_hidden.mean()).backward()
    assert model.transformer_layers[0].self_attn.qkv.weight.grad is not None


def check_grf_fixed_random_drop80():
    model_name = "grf_abs_dual_branch_fixed_random_drop80_hypercond"
    assert model_name in GRF_DUAL_BRANCH_VARIANTS
    assert model_name not in GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL
    assert GRF_DUAL_BRANCH_FIXED_RANDOM_DROP_KEEP_BY_MODEL[model_name] == 0.80

    model = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="dual",
        dynamic_branch_gate_warmup_steps=250000,
        fixed_random_drop_keep_probability=0.80,
    )
    assert model.dynamic_branch_gate is None
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)

    model.set_dynamic_branch_gate_t_env(249999)
    warmup_gate = model._branch_keep_gates(obs)
    assert th.all(warmup_gate == 1.0)

    model.set_dynamic_branch_gate_t_env(250000)
    sampled_gate_1 = model._branch_keep_gates(obs)
    sampled_gate_2 = model._branch_keep_gates(obs)
    assert th.all((sampled_gate_1 == 0.0) | (sampled_gate_1 == 1.0))
    assert not th.equal(sampled_gate_1, sampled_gate_2)

    model.set_dynamic_branch_gate_target_mode(True)
    target_gate = model._branch_keep_gates(obs)
    assert th.allclose(target_gate, th.full_like(target_gate, 0.80))
    model.set_dynamic_branch_gate_target_mode(False)

    model.set_semantic_test_mode(True)
    test_gate = model._branch_keep_gates(obs)
    assert th.all(test_gate == 1.0)
    model.set_semantic_test_mode(False)

    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dual_linear_encoder.weight.grad is not None
    assert model.transformer_layers[0].self_attn.qkv.weight.grad is not None


def check_single_transformer_observation_gate():
    gated_name = (
        "grf_abs_single_transformer_branch_binary_concrete_gate_hypercond"
    )
    random_aux_name = (
        "grf_abs_single_transformer_branch_binary_concrete_gate_"
        "random_drop_aux_hypercond"
    )
    assert {gated_name, random_aux_name} <= GRF_SINGLE_TRANSFORMER_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[gated_name] == (
        "binary_concrete"
    )
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[random_aux_name] == (
        "binary_concrete"
    )

    model = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="attention_only",
        dynamic_branch_gate_mode="binary_concrete",
        dynamic_branch_gate_hidden_dim=16,
        dynamic_branch_gate_warmup_steps=250000,
    )
    assert model.dual_linear_encoder is None
    assert model.dual_condition_fuser is None
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    model.set_dynamic_branch_gate_t_env(300000)
    condition, _ = model(obs, hidden)
    condition.mean().backward()
    gate_weight = model.dynamic_branch_gate.gate_network[-1].weight
    assert gate_weight.grad is not None
    assert gate_weight.grad.abs().sum().item() > 0.0

    learned_mask = th.ones(2, 2, 4, 30)
    learned_mask[..., 2] = 0.0
    random_mask = th.ones_like(learned_mask)
    random_mask[..., 3] = 0.0
    model.set_dynamic_branch_gate_override(learned_mask)
    model.set_dynamic_branch_gate_random_aux_combine_mode("multiply")
    model.set_dynamic_branch_gate_random_aux_mask(random_mask)
    model(obs, hidden)
    assert th.equal(
        model.latest_dynamic_branch_gates_graph,
        learned_mask * random_mask,
    )
    model.set_dynamic_branch_gate_override(None)
    model.set_dynamic_branch_gate_random_aux_mask(None)


def check_single_linear_observation_gate():
    gated_name = "grf_abs_single_linear_branch_binary_concrete_gate_hypercond"
    random_aux_name = (
        "grf_abs_single_linear_branch_binary_concrete_gate_"
        "random_drop_aux_hypercond"
    )
    assert {gated_name, random_aux_name} <= GRF_SINGLE_LINEAR_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[gated_name] == (
        "binary_concrete"
    )
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[random_aux_name] == (
        "binary_concrete"
    )

    model = GRFPublicPrivateBiasTransformerCapturer(
        n_agents=4,
        relation_dim=16,
        output_dim=16,
        num_heads=4,
        use_absolute_public=True,
        relation_encoder_style="linear_only",
        dynamic_branch_gate_mode="binary_concrete",
        dynamic_branch_gate_hidden_dim=16,
        dynamic_branch_gate_warmup_steps=250000,
    )
    assert model.dual_linear_encoder is not None
    assert model.dual_condition_fuser is None
    assert len(model.transformer_layers) == 0
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    model.set_dynamic_branch_gate_t_env(300000)
    condition, _ = model(obs, hidden)
    condition.mean().backward()
    assert model.dual_linear_encoder.weight.grad is not None
    gate_weight = model.dynamic_branch_gate.gate_network[-1].weight
    assert gate_weight.grad is not None
    assert gate_weight.grad.abs().sum().item() > 0.0

    learned_mask = th.ones(2, 2, 4, 30)
    learned_mask[..., 4] = 0.0
    random_mask = th.ones_like(learned_mask)
    random_mask[..., 5] = 0.0
    model.set_dynamic_branch_gate_override(learned_mask)
    model.set_dynamic_branch_gate_random_aux_combine_mode("multiply")
    model.set_dynamic_branch_gate_random_aux_mask(random_mask)
    model(obs, hidden)
    assert th.equal(
        model.latest_dynamic_branch_gates_graph,
        learned_mask * random_mask,
    )
    model.set_dynamic_branch_gate_override(None)
    model.set_dynamic_branch_gate_random_aux_mask(None)


def check_single_branch_random_drop_auxiliary():
    mlp_name = "grf_abs_mlp_relation_random_drop_aux_hypercond"
    transformer_name = (
        "grf_abs_single_transformer_branch_random_drop_aux_hypercond"
    )
    corridor_transformer_name = (
        "rpg_public_transformer_random_drop_aux_hypercond"
    )
    assert mlp_name in GRF_MLP_RELATION_VARIANTS
    assert transformer_name in GRF_SINGLE_TRANSFORMER_BRANCH_VARIANTS
    assert corridor_transformer_name in CleanHyperAgent.MODEL_SPECS

    # Exercise the controller-level observation mask used by both controls.
    mac = CleanMAC.__new__(CleanMAC)
    mac._random_drop_auxiliary_input_keep_probability = None
    mac._random_drop_auxiliary_input_mask = None
    observation = th.ones(8, 4, 30)
    mac.set_random_drop_auxiliary_input_keep_probability(0.2)
    first = mac._random_drop_auxiliary_observation(observation)
    repeated = mac._random_drop_auxiliary_observation(observation)
    assert th.equal(first, repeated)
    assert 0.0 < first.mean().item() < 0.5

    # Resetting the probability is how timestep scope requests a fresh mask.
    mac.set_random_drop_auxiliary_input_keep_probability(0.2)
    resampled = mac._random_drop_auxiliary_observation(observation)
    assert not th.equal(first, resampled)
    mac.set_random_drop_auxiliary_input_keep_probability(None)
    assert th.equal(mac._random_drop_auxiliary_observation(observation), observation)


def check_qmix_minimal_no_hypernetwork():
    model_name = "qmix_minimal"
    args = SimpleNamespace(
        clean_model_type=model_name,
        env="academy_counterattack_easy",
        n_agents=4,
        n_actions=19,
        rnn_hidden_dim=16,
        hypernet_embed=16,
        obs_last_action=False,
        obs_agent_id=False,
        clean_hard_gate_initial_keep_probability=0.95,
    )
    agent = CleanHyperAgent(input_shape=30, args=args)
    assert agent.MODEL_SPECS[model_name]["uses_hypernet"] is False
    assert agent.hyper_bottleneck_w is None
    assert agent.hyper_out_w is None
    assert agent.fixed_head is not None
    assert getattr(agent, "rpg_relation_capturer", None) is None
    obs = th.randn(2, 4, 30)
    q, hidden = agent(obs, None, context=None)
    assert q.shape == (2, 4, 19)
    assert hidden.shape == (2, 4, 16)
    q.mean().backward()
    assert agent.fixed_head[0].weight.grad is not None


def check_shared_slot_gate():
    model = build_grf("binary_concrete", gate_scope="shared")
    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    check_gate_module(model, obs)
    model.set_dynamic_branch_gate_t_env(250000)
    condition, next_hidden = model(obs, hidden)
    assert condition.shape == (2, 4, 16)
    assert next_hidden.shape == (2, 4, 16)
    (condition.mean() + next_hidden.mean()).backward()
    assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None


def check_sparse_gate_variants():
    variants = {
        "grf_abs_dual_branch_binary_concrete_bayesg_kl20_hypercond": (
            "binary_concrete",
            "bernoulli_kl",
            0.20,
        ),
        "grf_abs_dual_branch_binary_concrete_bayesg_kl80_hypercond": (
            "binary_concrete",
            "bernoulli_kl",
            0.80,
        ),
        "grf_abs_dual_branch_binary_concrete_bayesg_kl80_threshold70_hypercond": (
            "binary_concrete",
            "bernoulli_kl",
            0.80,
        ),
        "grf_abs_dual_branch_binary_concrete_bayesg_kl70_hypercond": (
            "binary_concrete",
            "bernoulli_kl",
            0.70,
        ),
        "grf_abs_dual_branch_binary_concrete_bimodal_budget80_hypercond": (
            "binary_concrete",
            "bimodal_budget",
            0.80,
        ),
        "grf_abs_dual_branch_hard_concrete_l0_hypercond": (
            "hard_concrete",
            "l0",
            0.0,
        ),
        "grf_abs_dual_branch_binary_concrete_perturb_param_importance_hypercond": (
            "binary_concrete",
            None,
            None,
        ),
        "grf_abs_dual_branch_binary_concrete_gradient_importance_hypercond": (
            "binary_concrete",
            None,
            None,
        ),
        "grf_abs_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond": (
            "binary_concrete",
            None,
            None,
        ),
        "grf_abs_dual_branch_binary_concrete_temporal_param_stability_hypercond": (
            "binary_concrete",
            None,
            None,
        ),
        "grf_abs_dual_branch_binary_concrete_temporal_param_small_change_hypercond": (
            "binary_concrete",
            None,
            None,
        ),
    }
    for model_name, (mode, regularizer, prior) in variants.items():
        assert model_name in GRF_DUAL_BRANCH_VARIANTS
        assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == mode
        if regularizer is not None:
            assert GRF_DUAL_BRANCH_GATE_REGULARIZER_BY_MODEL[model_name] == (
                regularizer,
                prior,
            )
        if model_name in GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL:
            assert 0.0 < GRF_DUAL_BRANCH_HARD_GATE_THRESHOLD_BY_MODEL[model_name] < 1.0

    for model_name in (
        "rpg_dual_branch_binary_concrete_perturbed_head_td_quality_hypercond",
        "rpg_dual_branch_binary_concrete_temporal_param_stability_hypercond",
        "rpg_dual_branch_binary_concrete_temporal_param_small_change_hypercond",
    ):
        assert model_name in RPG_DUAL_BRANCH_VARIANTS
        assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == (
            "binary_concrete"
        )

    obs = th.randn(2, 4, 30)
    hidden = th.zeros(2, 4, 16)
    for regularizer, prior, mode in (
        ("bernoulli_kl", 0.20, "binary_concrete"),
        ("bernoulli_kl", 0.80, "binary_concrete"),
        ("bimodal_budget", 0.80, "binary_concrete"),
        ("l0", 0.0, "hard_concrete"),
    ):
        model = build_grf(
            mode,
            dynamic_branch_gate_regularizer=regularizer,
            dynamic_branch_gate_prior_keep=prior,
        )
        model.set_dynamic_branch_gate_t_env(250000)
        condition, _ = model(obs, hidden)
        assert model.latest_aux_loss is not None
        assert th.isfinite(model.latest_aux_loss)
        model.latest_aux_loss.backward()
        assert model.dynamic_branch_gate.gate_network[-1].weight.grad is not None
        if mode == "hard_concrete":
            assert "dynamic_gate_expected_l0" in model.latest_aux_stats
        elif regularizer == "bimodal_budget":
            assert "dynamic_gate_bimodal_entropy" in model.latest_aux_stats
            assert "dynamic_gate_bimodal_budget_error" in model.latest_aux_stats
        else:
            assert "dynamic_gate_bernoulli_kl" in model.latest_aux_stats

    for model_name in variants:
        args = SimpleNamespace(
            clean_model_type=model_name,
            env="academy_counterattack_easy",
            n_agents=4,
            n_actions=19,
            rnn_hidden_dim=16,
            hypernet_embed=16,
            obs_last_action=False,
            obs_agent_id=False,
            clean_hard_gate_initial_keep_probability=0.95,
        )
        agent = CleanHyperAgent(input_shape=30, args=args)
        agent.set_dynamic_branch_gate_t_env(250000)
        context = {
            "obs": obs,
            "prev_action": th.zeros(2, 4, 19),
        }
        q, hidden_out = agent(obs, None, context=context)
        assert q.shape == (2, 4, 19)
        assert hidden_out.shape == (2, 4, 32)
        assert agent.latest_dynamic_branch_gates_graph is not None
        assert agent.latest_dynamic_branch_probabilities_graph is not None
        if "perturbed_head_td_quality" in model_name:
            assert agent.latest_generated_parameter_graph is not None
            assert agent.latest_policy_hidden_graph is not None
            unperturbed_q = agent._apply_generated_dynamic_head(
                agent.latest_policy_hidden_graph,
                agent.latest_generated_parameter_graph,
            )
            zero_noise_q = agent.perturbed_q_from_generated_parameters(
                agent.latest_policy_hidden_graph,
                agent.latest_generated_parameter_graph,
                relative_std=0.0,
            )
            assert th.allclose(unperturbed_q, zero_noise_q)
            perturbed_q = agent.perturbed_q_from_generated_parameters(
                agent.latest_policy_hidden_graph,
                agent.latest_generated_parameter_graph,
                relative_std=0.05,
            )
            assert perturbed_q.shape == q.shape
            perturbed_q.mean().backward()
            gate_parameter = (
                agent.rpg_relation_capturer.dynamic_branch_gate.gate_network[-1].weight
            )
            assert gate_parameter.grad is not None
        if "perturb_param_importance" in model_name:
            assert agent.latest_generated_parameter_graph is not None
            base_parameters = agent.latest_generated_parameter_graph
            base_gates = agent.latest_dynamic_branch_gates_graph
            agent.set_dynamic_branch_gate_override(base_gates)
            perturbed_context = dict(context)
            perturbed_context["obs"] = obs + 0.05 * th.randn_like(obs)
            agent(obs, None, context=perturbed_context)
            agent.set_dynamic_branch_gate_override(None)
            perturbed_parameters = agent.latest_generated_parameter_graph
            total, count = CleanLearner._generated_parameter_stability_pair(
                base_parameters,
                perturbed_parameters,
                th.ones(2, dtype=th.bool),
                2,
                4,
            )
            perturb_loss = total / count.clamp(min=1.0)
            assert th.isfinite(perturb_loss) and perturb_loss.item() > 0.0


def check_gradient_importance_gate_loss():
    learner = CleanLearner.__new__(CleanLearner)
    learner.adaptive_auxiliary_eps = 1e-8
    logits = th.randn(2, 3, 5, requires_grad=True)
    probability = th.sigmoid(logits)
    sampled_gate = probability
    td_loss = (sampled_gate * th.linspace(0.1, 2.0, 5)).sum().square()
    loss, stats = learner._gradient_importance_gate_loss(
        td_loss, [(probability, sampled_gate)]
    )
    assert th.isfinite(loss)
    assert stats["dynamic_gate_gradient_importance_max"].item() > 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0


def check_importance_auxiliary_gate_only_backward():
    learner = CleanLearner.__new__(CleanLearner)
    learner.use_amp = False
    gate_parameter = th.nn.Parameter(th.tensor(2.0))
    main_parameter = th.nn.Parameter(th.tensor(3.0))
    learner.importance_gate_parameters = (gate_parameter,)

    main_loss = (gate_parameter + main_parameter).square()
    auxiliary_loss = (2.0 * gate_parameter + 3.0 * main_parameter).square()
    learner._backward_main_and_gate_only_auxiliary(main_loss, auxiliary_loss)

    # Main loss contributes 10 to both parameters. The auxiliary contributes
    # another 52 only to the gate; its 78-gradient must not reach the main net.
    assert th.allclose(gate_parameter.grad, th.tensor(62.0))
    assert th.allclose(main_parameter.grad, th.tensor(10.0))


def check_temporal_parameter_change():
    previous = tuple(
        th.ones(6, width, requires_grad=True) for width in (4, 3, 5, 2)
    )
    current = tuple(
        th.full((6, width), 1.1, requires_grad=True)
        for width in (4, 3, 5, 2)
    )
    change, valid = CleanLearner._normalized_generated_parameter_change(
        previous,
        current,
        pair_valid=th.tensor([True, False, True]),
        batch_size=3,
        n_agents=2,
        scale_eps=1e-6,
    )
    assert change.shape == (3, 2)
    assert valid.sum().item() == 4.0
    assert th.allclose(
        change,
        th.full_like(change, 0.1 / 1.05),
        atol=1e-6,
    )
    small_change = (change < 0.1 / 2.0).detach()
    loss = th.where(small_change, change.square(), th.zeros_like(change)).mean()
    assert loss.item() == 0.0
    change.mean().backward()
    assert all(parameter.grad is not None for parameter in current)


def check_importance_alternating_training():
    learner = CleanLearner.__new__(CleanLearner)
    learner.importance_alternating_training = True
    learner.importance_auxiliary_warmup_steps = 250000
    learner.importance_non_gate_phase_steps = 80000
    learner.importance_gate_phase_steps = 20000

    assert learner._importance_training_phase(249999) == "non_gate_td"
    assert learner._importance_training_phase(250000) == "non_gate_td"
    assert learner._importance_training_phase(329999) == "non_gate_td"
    assert learner._importance_training_phase(330000) == "gate_td_aux"
    assert learner._importance_training_phase(349999) == "gate_td_aux"
    assert learner._importance_training_phase(350000) == "non_gate_td"

    learner.importance_auxiliary_warmup_steps = 0
    assert learner._importance_training_phase(0) == "non_gate_td"
    assert learner._importance_training_phase(79999) == "non_gate_td"
    assert learner._importance_training_phase(80000) == "gate_td_aux"
    assert learner._importance_training_phase(99999) == "gate_td_aux"
    assert learner._importance_training_phase(100000) == "non_gate_td"

    gate_parameter = th.nn.Parameter(th.tensor(2.0))
    main_parameter = th.nn.Parameter(th.tensor(3.0))
    learner.use_amp = False

    gate_phase_loss = (
        (gate_parameter + main_parameter).square()
        + (2.0 * gate_parameter + 3.0 * main_parameter).square()
    )
    learner._backward_parameters_only(gate_phase_loss, (gate_parameter,))
    assert gate_parameter.grad is not None
    assert main_parameter.grad is None

    gate_parameter.grad = None
    main_phase_loss = (gate_parameter + main_parameter).square()
    learner._backward_parameters_only(main_phase_loss, (main_parameter,))
    assert gate_parameter.grad is None
    assert th.allclose(main_parameter.grad, th.tensor(10.0))


def check_test_slot_probability_summary():
    mac = CleanMAC.__new__(CleanMAC)
    mac._test_gate_probability_sum = None
    mac._test_gate_probability_count = 0
    mac.n_agents = 1
    mac.agent = SimpleNamespace(
        rpg_relation_capturer=SimpleNamespace(semantic_names=("slot_x", "slot_y"))
    )
    mac.latest_dynamic_branch_probabilities_graph = th.tensor(
        [
            [[[0.2, 0.4]], [[0.6, 0.8]]],
            [[[0.1, 0.3]], [[0.5, 0.7]]],
        ]
    )
    # The live agent stores [branch, batch * agent, slot], so this also
    # verifies that the controller restores the environment/agent axes before
    # applying the active-environment selection.
    mac.latest_dynamic_branch_probabilities_graph = (
        mac.latest_dynamic_branch_probabilities_graph.reshape(2, 2, 2)
    )
    mac._accumulate_test_gate_probabilities([0, 1], batch_size=2)
    summary = mac.pop_test_gate_probability_summary()
    assert summary["slot_names"] == ["slot_x", "slot_y"]
    assert summary["sample_count"] == 2
    assert th.allclose(th.tensor(summary["linear"]), th.tensor([0.4, 0.6]))
    assert th.allclose(th.tensor(summary["attention"]), th.tensor([0.3, 0.5]))
    assert mac.pop_test_gate_probability_summary() is None


def check_condition_gradient_consistency():
    model_name = "grf_abs_dual_branch_hard_gate_grad_consistency_hypercond"
    assert model_name in GRF_DUAL_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"

    learner = CleanLearner.__new__(CleanLearner)
    learner.condition_gradient_consistency_pairs = 2
    source = th.randn(3, 4, 2, 5, requires_grad=True)
    conditions = [source[:, time_index] for time_index in range(4)]
    predictions = th.stack(
        [
            (condition.pow(2).sum(dim=-1)).sum(dim=-1, keepdim=True)
            for condition in conditions
        ],
        dim=1,
    )
    td_error = predictions - 1.0
    td_mask = th.ones_like(td_error)
    loss, stats = learner._condition_gradient_consistency_loss(
        td_error, td_mask, conditions
    )
    assert th.isfinite(loss)
    assert stats["condition_grad_consistency_pairs"].item() == 2.0
    loss.backward()
    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0


def check_generated_parameter_stability():
    model_name = "grf_abs_dual_branch_hard_gate_param_stability_hypercond"
    assert model_name in GRF_DUAL_BRANCH_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"

    batch_size = 3
    n_agents = 2
    source = th.randn(batch_size * n_agents, 7, requires_grad=True)
    previous = (source[:, :4] * 0.5, source[:, 4:] * 0.25)
    current = (source[:, :4], source[:, 4:])
    pair_valid = th.tensor([True, False, True])
    total, count = CleanLearner._generated_parameter_stability_pair(
        previous, current, pair_valid, batch_size, n_agents
    )
    loss = total / count.clamp(min=1.0)
    assert th.isfinite(loss)
    assert count.item() == 4.0
    loss.backward()
    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0

    corridor_model_name = "rpg_dual_branch_hard_gate_param_stability_hypercond"
    assert corridor_model_name in RPG_DUAL_BRANCH_VARIANTS
    assert (
        RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[corridor_model_name]
        == "hard_st"
    )


def check_corridor_gradient_consistency_registration():
    model_name = "rpg_dual_branch_hard_gate_grad_consistency_hypercond"
    assert model_name in RPG_DUAL_BRANCH_VARIANTS
    assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[model_name] == "hard_st"


def check_new_variant_registration():
    for prefix, variants, modes in (
        ("grf_abs", GRF_DUAL_BRANCH_VARIANTS, GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
        ("rpg", RPG_DUAL_BRANCH_VARIANTS, RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
    ):
        for auxiliary in ("param_stability", "grad_consistency"):
            concrete = f"{prefix}_dual_branch_binary_concrete_{auxiliary}_hypercond"
            adaptive = f"{prefix}_dual_branch_hard_gate_adaptive_{auxiliary}_hypercond"
            assert concrete in variants
            assert modes[concrete] == "binary_concrete"
            assert adaptive in variants
            assert modes[adaptive] == "hard_st"

    for auxiliary in ("param_stability", "grad_consistency"):
        attention_only = (
            f"rpg_dual_branch_attention_only_hard_gate_{auxiliary}_hypercond"
        )
        split_head = f"rpg_dual_branch_split_head_hard_gate_{auxiliary}_hypercond"
        assert attention_only in RPG_DUAL_BRANCH_VARIANTS
        assert attention_only in RPG_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS
        assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[attention_only] == "hard_st"
        assert split_head in RPG_DUAL_BRANCH_VARIANTS
        assert split_head in RPG_DUAL_BRANCH_SPLIT_HEAD_VARIANTS
        assert RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[split_head] == "hard_st"

    for prefix, variants, modes in (
        ("grf_abs", GRF_DUAL_BRANCH_VARIANTS, GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
        ("rpg", RPG_DUAL_BRANCH_VARIANTS, RPG_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL),
    ):
        base = f"{prefix}_dual_branch_binary_concrete_adaptive"
        for suffix in ("", "_slot", "_attention_only", "_split_head"):
            model_name = f"{base}{suffix}_grad_consistency_hypercond"
            assert model_name in variants
            assert modes[model_name] == "binary_concrete"

        for suffix in (
            "param_stability",
            "attention_only_param_stability",
            "parameter_likelihood",
            "attention_only_parameter_likelihood",
        ):
            model_name = f"{base}_{suffix}_hypercond"
            assert model_name in variants
            assert modes[model_name] == "binary_concrete"

        td_weighted = (
            f"{base}_td_weighted_param_likelihood_hypercond"
        )
        assert td_weighted in variants
        assert modes[td_weighted] == "binary_concrete"
        trajectory = (
            f"{base}_trajectory_parameter_likelihood_hypercond"
        )
        assert trajectory in variants
        assert modes[trajectory] == "binary_concrete"

    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS
    )
    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_attention_only_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_ATTENTION_ONLY_GATE_VARIANTS
    )
    assert (
        "grf_abs_dual_branch_binary_concrete_adaptive_split_head_grad_consistency_hypercond"
        in GRF_DUAL_BRANCH_SPLIT_HEAD_VARIANTS
    )
    assert (
        "rpg_dual_branch_binary_concrete_adaptive_slot_grad_consistency_hypercond"
        in RPG_DUAL_BRANCH_SLOT_SHARED_GATE_VARIANTS
    )


def check_adaptive_auxiliary_ratio_detached():
    learner = CleanLearner.__new__(CleanLearner)
    learner.adaptive_auxiliary_target_ratio = 0.1
    learner.adaptive_auxiliary_ema_decay = 0.9
    learner.adaptive_auxiliary_eps = 1e-8
    learner.adaptive_auxiliary_max_coef = 100.0
    learner.adaptive_auxiliary_ema_td = None
    learner.adaptive_auxiliary_ema_aux = None
    learner.latest_adaptive_auxiliary_stats = {}

    td_source = th.tensor(2.0, requires_grad=True)
    aux_source = th.tensor(0.5, requires_grad=True)
    td_loss = td_source.square()
    auxiliary_loss = aux_source.square()
    coefficient = learner._adaptive_auxiliary_coefficient(td_loss, auxiliary_loss)
    assert isinstance(coefficient, float)
    assert abs(coefficient * auxiliary_loss.item() / td_loss.item() - 0.1) < 1e-6
    total = td_loss + coefficient * auxiliary_loss
    total.backward()
    assert abs(td_source.grad.item() - 4.0) < 1e-6
    assert abs(aux_source.grad.item() - coefficient) < 1e-6

    learner.adaptive_auxiliary_target_ratio = 0.0
    zero_coefficient = learner._adaptive_auxiliary_coefficient(
        td_loss.detach(), auxiliary_loss.detach()
    )
    assert zero_coefficient == 0.0


def check_generated_parameter_conditional_nll():
    learner = CleanLearner.__new__(CleanLearner)
    learner.generated_parameter_likelihood_std = 1.0
    learner.adaptive_auxiliary_eps = 1e-8
    batch_size = 2
    n_agents = 3
    source = th.randn(batch_size * n_agents, 5, requires_grad=True)
    target = th.randn(batch_size * n_agents, 5)
    valid = th.tensor([True, False])
    total, count = learner._generated_parameter_conditional_nll(
        (source,), (target,), valid, batch_size, n_agents
    )
    loss = total / count.clamp(min=1.0)
    assert th.isfinite(loss)
    assert count.item() == 3.0
    loss.backward()
    assert source.grad is not None
    assert source.grad.abs().sum().item() > 0.0


def check_td_weighted_parameter_score():
    agent = CleanHyperAgent.__new__(CleanHyperAgent)
    th.nn.Module.__init__(agent)
    agent.model_type = (
        "grf_abs_dual_branch_binary_concrete_adaptive_"
        "td_weighted_param_likelihood_hypercond"
    )
    agent.n_agents = 2
    agent.td_parameter_relative_std = 0.02
    agent.td_parameter_minimum_rms = 0.01
    agent._td_parameter_sampling_enabled = True
    agent._generated_parameter_log_prob_sum = None
    agent._generated_parameter_log_prob_count = 0
    agent.latest_generated_parameter_log_prob = None
    mean = th.randn(6, 4, requires_grad=True)
    sample = agent._sample_td_weighted_generated_parameter(mean)
    assert sample.shape == mean.shape
    assert not th.equal(sample, mean)
    assert agent.latest_generated_parameter_log_prob.shape == (3, 2)
    score_loss = -agent.latest_generated_parameter_log_prob.mean()
    score_loss.backward()
    assert mean.grad is not None
    assert mean.grad.abs().sum().item() > 0.0


def check_trajectory_parameter_projection():
    first = th.randn(6, 4, requires_grad=True)
    second = th.randn(6, 3, 2, requires_grad=True)
    projected_1 = CleanMAC._fixed_parameter_projection((first, second), 64)
    projected_2 = CleanMAC._fixed_parameter_projection((first, second), 64)
    assert projected_1.shape == (6, 64)
    assert th.equal(projected_1, projected_2)
    projected_1.square().mean().backward()
    assert first.grad is not None and first.grad.abs().sum().item() > 0.0
    assert second.grad is not None and second.grad.abs().sum().item() > 0.0


def check_structured_perturbed_head_replay():
    batch_size, n_agents = 2, 3
    hidden_dim, n_ego_actions, n_enemies = 4, 5, 2
    flat_size = batch_size * n_agents
    hidden = th.randn(batch_size, n_agents, hidden_dim)
    parameters = (
        th.randn(flat_size, hidden_dim, hidden_dim),
        th.randn(flat_size, 1, hidden_dim),
        th.randn(flat_size, hidden_dim, n_ego_actions),
        th.randn(flat_size, 1, n_ego_actions),
        th.randn(flat_size, hidden_dim + 3, 1),
        th.randn(flat_size, 1, 1),
    )
    interaction_input = th.randn(flat_size, n_enemies, hidden_dim + 3)
    enemy_mask = th.tensor(
        [
            [[True, False], [True, True], [False, True]],
            [[True, True], [False, False], [True, False]],
        ]
    )
    dummy_agent = SimpleNamespace(hidden_dim=hidden_dim)
    q = CleanHyperAgent.perturbed_q_from_generated_parameters(
        dummy_agent,
        hidden,
        parameters,
        relative_std=0.0,
        interaction_input=interaction_input,
        enemy_mask=enemy_mask,
    )
    assert q.shape == (
        batch_size,
        n_agents,
        n_ego_actions + n_enemies,
    )
    assert th.all(q[..., n_ego_actions:].masked_select(~enemy_mask) == 0.0)


def main():
    th.manual_seed(17)
    check_grf("hard_st")
    print("dual_branch hard_st scalar_forward_backward=ok")
    check_grf("binary_concrete")
    print("dual_branch binary_concrete stochastic_recovery=ok")
    check_grf_transformer_only()
    print("single_transformer_branch forward_backward=ok")
    check_grf_fixed_random_drop80()
    print("dual_branch fixed_random_drop80 train_sample_target_mean_test_open=ok")
    check_single_transformer_observation_gate()
    print("single_transformer obs_gate_and_random_multiply=ok")
    check_single_linear_observation_gate()
    print("single_linear obs_gate_and_random_multiply=ok")
    check_single_branch_random_drop_auxiliary()
    print("single_branch random_drop_auxiliary observation_mask=ok")
    check_qmix_minimal_no_hypernetwork()
    print("qmix_minimal fixed_head_no_hypernetwork=ok")
    check_shared_slot_gate()
    print("shared_slot_binary_concrete forward_backward=ok")
    check_sparse_gate_variants()
    print("sparse_gate_five_variants registration_and_aux=ok")
    check_gradient_importance_gate_loss()
    print("gradient_importance first_order_weighted_sparsity=ok")
    check_importance_auxiliary_gate_only_backward()
    print("importance_auxiliary gate_only_backward=ok")
    check_temporal_parameter_change()
    print("temporal_parameter normalized_change_and_cutoff=ok")
    check_importance_alternating_training()
    print("importance_auxiliary alternating_80k_20k=ok")
    check_test_slot_probability_summary()
    print("dynamic_gate test_slot_probability_summary=ok")
    check_condition_gradient_consistency()
    print("condition_gradient_consistency second_order=ok")
    check_generated_parameter_stability()
    print("generated_parameter_stability exact_l1=ok")
    check_corridor_gradient_consistency_registration()
    print("corridor_stability_variants registration=ok")
    check_new_variant_registration()
    print("new_gate_and_branch_role_variants registration=ok")
    check_adaptive_auxiliary_ratio_detached()
    print("adaptive_auxiliary detached_ema_ratio=ok")
    check_generated_parameter_conditional_nll()
    print("generated_parameter_conditional_nll detached_target=ok")
    check_td_weighted_parameter_score()
    print("td_weighted_parameter_likelihood score_gradient=ok")
    check_trajectory_parameter_projection()
    print("trajectory_parameter_projection deterministic_gradient=ok")
    check_structured_perturbed_head_replay()
    print("corridor_structured_perturbed_head replay_and_mask=ok")


if __name__ == "__main__":
    main()
