#!/usr/bin/env python3
"""Preflight the self-first SMAC condition and the exact Linear KL80 objective."""
import math
from pathlib import Path
import sys

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_test_counter_transformer_nine import make_case
from modules.agents.counter_transformer_suite import (
    ALL_PROFILES,
    model_type_for,
)


SCENE = "5m_vs_6m"


def check_cyclic_condition():
    th.manual_seed(91)
    mac, learner, batch, logger = make_case("linear_cyclic_obs_baseline", SCENE)
    agent = mac.agent
    capturer = agent.rpg_relation_capturer
    flags = ALL_PROFILES["linear_cyclic_obs_baseline"]
    assert flags == {"branch": "linear", "cyclic_self_first": True}
    assert model_type_for("linear_cyclic_obs_baseline", "smac") == (
        "smac_single_linear_suite_cyclic_obs_baseline_hypercond"
    )
    try:
        model_type_for("linear_cyclic_obs_baseline", "grf")
    except ValueError as exc:
        assert "SMAC-only" in str(exc)
    else:
        raise AssertionError("SMAC-only cyclic profile resolved to a GRF model")
    assert capturer.relation_encoder_style == "linear_only"
    assert len(capturer.transformer_layers) == 0
    assert capturer.dynamic_branch_gate is None
    assert capturer.cyclic_self_first
    assert not learner.gate_regularization_active
    assert not learner.random_drop_auxiliary_active

    layout = capturer.observation_layout
    obs = th.zeros(2, mac.args.n_agents, mac.args.obs_shape)
    move_end = layout["move_dim"]
    enemy_end = move_end + layout["n_enemies"] * layout["enemy_feat_dim"]
    ally_end = enemy_end + layout["n_allies"] * layout["ally_feat_dim"]
    obs[..., :move_end] = th.arange(move_end).view(1, 1, -1)
    obs[..., move_end:enemy_end] = 7.0
    obs[..., ally_end:] = 9.0
    for agent_id in range(mac.args.n_agents):
        roster = [j for j in range(mac.args.n_agents) if j != agent_id]
        values = th.tensor([agent_id * 100 + j for j in roster], dtype=obs.dtype)
        ally_rows = values.view(1, -1, 1).expand(2, -1, layout["ally_feat_dim"])
        obs[:, agent_id, enemy_end:ally_end] = ally_rows.reshape(2, -1)

    canonical = capturer.canonicalize_cyclic_self_first(obs)
    assert canonical.shape == obs.shape
    assert th.equal(canonical[..., :move_end], obs[..., :move_end])
    assert th.equal(canonical[..., move_end:enemy_end], obs[..., move_end:enemy_end])
    assert th.equal(canonical[..., enemy_end:enemy_end + layout["own_dim"]],
                    obs[..., ally_end:])
    canonical_ally_start = enemy_end + layout["own_dim"]
    for agent_id in range(mac.args.n_agents):
        desired = [(agent_id + offset) % mac.args.n_agents
                   for offset in range(1, mac.args.n_agents)]
        expected = th.tensor([agent_id * 100 + j for j in desired], dtype=obs.dtype)
        actual = canonical[:, agent_id, canonical_ally_start:].reshape(
            2, layout["n_allies"], layout["ally_feat_dim"]
        )[:, :, 0]
        assert th.equal(actual, expected.view(1, -1).expand_as(actual))

    # Only condition/capturer observation is canonicalized; the policy trunk's
    # observation prefix remains byte-for-byte the original SMAC observation.
    batch["obs"][:, 0] = obs
    trunk_input = mac._build_inputs(batch, 0)
    context = mac._build_model_context(batch, 0)
    assert th.equal(trunk_input[..., :mac.args.obs_shape], obs)
    assert th.equal(context["obs"], canonical)

    # The new order is also reflected in diagnostic/gate semantic slot labels.
    names = capturer.semantic_names
    first_ally = next(i for i, name in enumerate(names) if name.startswith("ally_"))
    first_own = next(i for i, name in enumerate(names) if name.startswith("self_health")
                     or name.startswith("self_unit_type"))
    assert first_own < first_ally

    learner.train(batch, t_env=10, episode_num=1)
    assert math.isfinite(float(logger.stats["loss_td"][-1][1]))
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    print(
        "cyclic-observation: self-first/cyclic roster, unchanged raw GRU observation, "
        "and learner update OK",
        flush=True,
    )


def check_linear_direct_kl80():
    th.manual_seed(92)
    mac, learner, batch, logger = make_case("linear_bayesg_kl80_keep", SCENE)
    capturer = mac.agent.rpg_relation_capturer
    flags = ALL_PROFILES["linear_bayesg_kl80_keep"]
    assert flags.get("branch") == "linear"
    assert flags.get("gate") and flags.get("kl")
    assert not flags.get("aux")
    assert model_type_for("linear_bayesg_kl80_keep", "smac") == (
        "smac_single_linear_suite_bayesg_kl80_keep_hypercond"
    )
    assert capturer.relation_encoder_style == "linear_only"
    assert len(capturer.transformer_layers) == 0
    assert capturer.dynamic_branch_gate is not None
    assert capturer.dynamic_branch_gate_regularizer == "bernoulli_kl"
    assert capturer.dynamic_branch_gate_prior_keep == 0.8
    assert not capturer.kl80_auxiliary_enabled
    assert getattr(capturer, "kl80_auxiliary_gate", None) is None
    assert learner.main_td_coef == 1.0
    assert learner.nomask_td_auxiliary_coef == 0.0
    assert learner.gate_regularization_active
    assert learner.adaptive_auxiliary_ratio_active
    assert math.isclose(learner.adaptive_auxiliary_target_ratio, 0.1)
    assert not learner.random_drop_auxiliary_active

    # At t_env=300k the gate warmup is over. KL regularizes only the linear
    # gate logits; the unused attention-output logits must have zero gradient.
    mac.init_hidden(batch.batch_size)
    mac.set_dynamic_branch_gate_t_env(300000)
    mac.forward(batch, t=0)
    final_gate_layer = capturer.dynamic_branch_gate.gate_network[-1]
    kl_grad = th.autograd.grad(mac.latest_aux_loss, final_gate_layer.bias,
                               retain_graph=True)[0]
    group_count = capturer.dynamic_branch_gate.group_count
    assert kl_grad[:group_count].abs().sum() > 0
    assert kl_grad[group_count:].abs().sum() == 0

    learner.train(batch, t_env=300000, episode_num=1)
    td = float(logger.stats["loss_td"][-1][1])
    kl = float(logger.stats["loss_aux"][-1][1])
    weighted_kl = float(logger.stats["weighted_loss_dynamic_gate_regularizer"][-1][1])
    assert math.isfinite(td) and td >= 0
    assert math.isfinite(kl) and kl > 0
    ratio = float(logger.stats["dynamic_gate_regularizer_to_td_ratio"][-1][1])
    assert math.isclose(ratio, 0.1, rel_tol=1e-5, abs_tol=1e-6)
    assert math.isclose(weighted_kl, ratio * td, rel_tol=1e-6, abs_tol=1e-7)
    assert not logger.stats.get("loss_random_drop_td_auxiliary")
    assert all(th.isfinite(parameter).all() for parameter in mac.parameters())
    print("linear-kl80-direct: one Linear TD path + EMA-scaled KL(Bernoulli(p)||.8) "
          "at 10% target TD ratio, no attention branch / no KL80 auxiliary TD; "
          "gradients verified", flush=True)


if __name__ == "__main__":
    th.set_num_threads(1)
    check_cyclic_condition()
    check_linear_direct_kl80()
