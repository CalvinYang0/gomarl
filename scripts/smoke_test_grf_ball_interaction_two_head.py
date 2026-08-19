#!/usr/bin/env python3
"""Smoke-test the GRF shared-Q plus ball-interaction two-head residual model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.clean_hyper_agent import (  # noqa: E402
    GRF_BALL_INTERACTION_TWO_HEAD_VARIANTS,
    GRF_DECISION_MAKER_VARIANTS,
    GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL,
    GRF_DUAL_BRANCH_GRAD_CONSISTENCY_VARIANTS,
    GRF_DUAL_BRANCH_PARAMETER_LIKELIHOOD_VARIANTS,
    GRF_DUAL_BRANCH_PARAMETER_STABILITY_VARIANTS,
    CleanHyperAgent,
)


MODEL_TYPE = (
    "grf_abs_dual_branch_binary_concrete_td_only_ball_interaction_two_head_hypercond"
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


def build_context(batch_size, n_agents):
    obs_dim = 4 * n_agents + 14
    return {
        "obs": th.randn(batch_size, n_agents, obs_dim),
        "prev_action": th.zeros(batch_size, n_agents, 19),
    }


def enable_nonzero_residual_weights(agent):
    with th.no_grad():
        for branch in ("self_control", "ball_interaction"):
            generator = agent.grf_two_head_residual_hyper[f"{branch}_w"]
            generator.bias.copy_(
                th.linspace(-0.02, 0.02, generator.bias.numel())
            )


def check_scene_shape(env, n_agents):
    agent = build_agent(env=env, n_agents=n_agents)
    context = build_context(batch_size=2, n_agents=n_agents)
    agent.set_dynamic_branch_gate_t_env(250000)
    q, next_hidden = agent(context["obs"], None, context=context)
    assert q.shape == (2, n_agents, 19)
    assert next_hidden.shape == (2, n_agents, 32)


def main():
    th.manual_seed(37)
    assert MODEL_TYPE in GRF_DECISION_MAKER_VARIANTS
    assert MODEL_TYPE in GRF_BALL_INTERACTION_TWO_HEAD_VARIANTS
    assert GRF_DUAL_BRANCH_DYNAMIC_GATE_MODE_BY_MODEL[MODEL_TYPE] == "binary_concrete"
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_GRAD_CONSISTENCY_VARIANTS
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_PARAMETER_STABILITY_VARIANTS
    assert MODEL_TYPE not in GRF_DUAL_BRANCH_PARAMETER_LIKELIHOOD_VARIANTS

    agent = build_agent()
    assert agent.grf_self_control_action_idx.tolist() == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15
    ]
    assert agent.grf_ball_interaction_action_idx.tolist() == [
        9, 10, 11, 12, 16, 17, 18
    ]
    all_actions = th.cat(
        [
            agent.grf_self_control_action_idx,
            agent.grf_ball_interaction_action_idx,
        ]
    )
    assert sorted(all_actions.tolist()) == list(range(19))

    # Every residual generator is exactly zero at initialization.
    for module in agent.grf_two_head_residual_hyper.values():
        for parameter in module.parameters():
            assert th.count_nonzero(parameter).item() == 0

    batch_size = 2
    n_agents = 4
    context = build_context(batch_size, n_agents)
    hidden = th.randn(batch_size, n_agents, 16)
    condition = th.randn(batch_size, n_agents, 16)
    agent.eval()
    q_base = agent._apply_dynamic_head(hidden, condition)
    q_two_head = agent._apply_grf_decision_maker_head(
        hidden, condition, context=context
    )
    assert th.equal(q_base, q_two_head)

    # Each residual changes only its own action partition.
    with th.no_grad():
        agent.grf_two_head_residual_hyper["self_control_b"].bias.fill_(0.25)
    q_self = agent._apply_grf_decision_maker_head(hidden, condition, context=context)
    assert th.allclose(
        q_self[:, :, agent.grf_self_control_action_idx],
        q_base[:, :, agent.grf_self_control_action_idx] + 0.25,
    )
    assert th.equal(
        q_self[:, :, agent.grf_ball_interaction_action_idx],
        q_base[:, :, agent.grf_ball_interaction_action_idx],
    )
    with th.no_grad():
        agent.grf_two_head_residual_hyper["self_control_b"].bias.zero_()
        agent.grf_two_head_residual_hyper["ball_interaction_b"].bias.fill_(0.25)
    q_ball = agent._apply_grf_decision_maker_head(hidden, condition, context=context)
    assert th.equal(
        q_ball[:, :, agent.grf_self_control_action_idx],
        q_base[:, :, agent.grf_self_control_action_idx],
    )
    assert th.allclose(
        q_ball[:, :, agent.grf_ball_interaction_action_idx],
        q_base[:, :, agent.grf_ball_interaction_action_idx] + 0.25,
    )

    # Once residual weights depart from zero, all independent raw-entity
    # encoders receive gradients and observation changes affect the Q values.
    with th.no_grad():
        agent.grf_two_head_residual_hyper["ball_interaction_b"].bias.zero_()
    enable_nonzero_residual_weights(agent)
    agent.zero_grad(set_to_none=True)
    q = agent._apply_grf_decision_maker_head(hidden, condition, context=context)
    q.square().mean().backward()
    for encoder in (
        agent.grf_head_self_encoder,
        agent.grf_head_ball_encoder,
        agent.grf_head_ally_encoder,
        agent.grf_head_opponent_encoder,
    ):
        grad = encoder[0].weight.grad
        assert grad is not None and grad.abs().sum().item() > 0.0

    changed_context = dict(context)
    changed_context["obs"] = context["obs"] + 1.0
    q_changed = agent._apply_grf_decision_maker_head(
        hidden, condition, context=changed_context
    )
    assert not th.equal(q.detach(), q_changed.detach())

    check_scene_shape("academy_counterattack_easy", 4)
    check_scene_shape("academy_pass_and_shoot_with_keeper", 2)
    check_scene_shape("academy_3_vs_1_with_keeper", 3)
    print("GRF ball-interaction two-head residual smoke test passed")


if __name__ == "__main__":
    main()
