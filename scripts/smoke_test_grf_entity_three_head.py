#!/usr/bin/env python3
"""Smoke-test the TD-only GRF raw-entity three-head action model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_DECISION_MAKER_VARIANTS,
    GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    GRF_DUAL_BRANCH_GRAD_CONSISTENCY_VARIANTS,
    GRF_DUAL_BRANCH_PARAMETER_LIKELIHOOD_VARIANTS,
    GRF_DUAL_BRANCH_PARAMETER_STABILITY_VARIANTS,
    GRF_INDEPENDENT_ENTITY_THREE_HEAD_VARIANTS,
    CleanHyperAgent,
)


MODEL_TYPE = (
    "grf_abs_dual_branch_binary_concrete_td_only_entity_three_head_hypercond"
)


def build_agent(env="academy_counterattack_easy", n_agents=4):
    args = SimpleNamespace(
        clean_model_type=MODEL_TYPE,
        env=env,
        n_agents=n_agents,
        n_actions=19,
        rnn_hidden_dim=16,
        hypernet_embed=16,
        obs_last_action=False,
        obs_agent_id=False,
    )
    return CleanHyperAgent(input_shape=4 * n_agents + 14, args=args)


def check_scene_shape(env, n_agents):
    agent = build_agent(env=env, n_agents=n_agents)
    obs_dim = 4 * n_agents + 14
    obs = th.randn(2, n_agents, obs_dim)
    context = {
        "obs": obs,
        "prev_action": th.zeros(2, n_agents, 19),
    }
    agent.set_dynamic_branch_gate_t_env(250000)
    q, next_hidden = agent(obs, None, context=context)
    assert q.shape == (2, n_agents, 19)
    assert next_hidden.shape == (2, n_agents, 32)


def main():
    th.manual_seed(31)
    assert MODEL_TYPE in GRF_DECISION_MAKER_VARIANTS
    assert MODEL_TYPE in GRF_INDEPENDENT_ENTITY_THREE_HEAD_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[MODEL_TYPE] == "binary_concrete"
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_GRAD_CONSISTENCY_VARIANTS
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_PARAMETER_STABILITY_VARIANTS
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_PARAMETER_LIKELIHOOD_VARIANTS

    agent = build_agent()
    assert agent.grf_ego_action_idx.tolist() == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18
    ]
    assert agent.grf_ally_action_idx.tolist() == [9, 10, 11]
    assert agent.grf_opponent_action_idx.tolist() == [16]
    all_actions = th.cat(
        [
            agent.grf_ego_action_idx,
            agent.grf_ally_action_idx,
            agent.grf_opponent_action_idx,
        ]
    )
    assert sorted(all_actions.tolist()) == list(range(19))

    batch_size = 2
    obs = th.randn(batch_size, 4, 30)
    context = {
        "obs": obs,
        "prev_action": th.zeros(batch_size, 4, 19),
    }
    agent.set_dynamic_branch_gate_t_env(250000)
    q, next_hidden = agent(obs, None, context=context)
    assert q.shape == (batch_size, 4, 19)
    assert next_hidden.shape == (batch_size, 4, 32)
    assert agent.latest_aux_loss is None

    q.square().mean().backward()
    for encoder in (
        agent.grf_head_self_encoder,
        agent.grf_head_ball_encoder,
        agent.grf_head_ally_encoder,
        agent.grf_head_opponent_encoder,
    ):
        grad = encoder[0].weight.grad
        assert grad is not None and grad.abs().sum().item() > 0.0

    # The action encoders must use raw obs rather than cached Transformer tokens.
    agent.eval()
    hidden = th.randn(batch_size, 4, 16)
    condition = th.randn(batch_size, 4, 16)
    q_before = agent._apply_grf_decision_maker_head(
        hidden, condition, context=context
    )
    agent.rpg_relation_capturer.latest_self_token = th.randn(batch_size, 4, 16)
    agent.rpg_relation_capturer.latest_ally_tokens = th.randn(batch_size, 4, 3, 16)
    agent.rpg_relation_capturer.latest_opponent_tokens = th.randn(
        batch_size, 4, 2, 16
    )
    q_after = agent._apply_grf_decision_maker_head(
        hidden, condition, context=context
    )
    assert th.equal(q_before, q_after)

    changed_context = dict(context)
    changed_context["obs"] = obs + 1.0
    q_changed = agent._apply_grf_decision_maker_head(
        hidden, condition, context=changed_context
    )
    assert not th.equal(q_before, q_changed)

    check_scene_shape("academy_pass_and_shoot_with_keeper", 2)
    check_scene_shape("academy_3_vs_1_with_keeper", 3)
    print("GRF TD-only raw-entity three-head smoke test passed")


if __name__ == "__main__":
    main()
